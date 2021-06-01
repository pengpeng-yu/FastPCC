from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import compressai
from compressai.models.utils import update_registered_buffers

from lib import loss_function
from lib.points_layers import TransitionDown, NeighborFeatureGenerator, RandLANeighborFea, \
    LocalFeatureAggregation as LFA
from lib.torch_utils import MLPBlock
from models.exp4.model_config import ModelConfig


# TODO: anchor points based coordinates offset prediction, LFA upsample, intermediate supervision
class GenerativeTransitionUp(nn.Module):
    def __init__(self, in_channels, neighbor_feature_generator: NeighborFeatureGenerator,
                 raw_neighbor_fea_out_chnls, out_channels, cache_out_feature=False):
        super(GenerativeTransitionUp, self).__init__()
        self.lfa = LFA(in_channels, neighbor_feature_generator,
                       raw_neighbor_fea_out_chnls, out_channels, cache_out_feature)
        self.mlp_offset = nn.Linear()

    def forward(self, x):
        x = self.lfa(x)
        xyz, cached_feature, raw_neighbors_feature, neighbors_idx, cached_sample_indexes = x
        pass
        return None


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        neighbor_fea_generator = RandLANeighborFea(cfg.neighbor_num)

        self.encoder = [LFA(3, neighbor_fea_generator, 8, 16),
                        LFA(16, neighbor_fea_generator, 8, 16),

                        TransitionDown(cfg.sample_method, 0.25),
                        LFA(16, neighbor_fea_generator, 16, 32),
                        LFA(32, neighbor_fea_generator, 16, 32),

                        TransitionDown(cfg.sample_method, 0.25),
                        LFA(32, neighbor_fea_generator, 32, 64),
                        LFA(64, neighbor_fea_generator, 32, 64),

                        TransitionDown(cfg.sample_method, 0.25),
                        LFA(64, neighbor_fea_generator, 32, 64),
                        LFA(64, neighbor_fea_generator, 32, 32)]

        self.encoded_points_num = reduce(lambda x, y: x * y,
                                         [t.sample_rate for t in self.encoder
                                          if isinstance(t, TransitionDown)] + [1])
        self.encoder = nn.Sequential(*self.encoder)
        self.mlp_enc_out = nn.Sequential(MLPBlock(32, 32, activation=None, batchnorm='nn.bn1d'))

        self.entropy_bottleneck = compressai.entropy_models.EntropyBottleneck(self.mlp_enc_out[-1].out_channels)

        self.decoder = nn.Sequential(
            GenerativeTransitionUp(32, neighbor_fea_generator, 32, 64),
            MLPBlock(2048, self.cfg.input_points_num * 3, activation=None, batchnorm='nn.bn1d'))

    def forward(self, xyz):
        batch_size = xyz.shape[0]

        # encode
        sampled_xyz, fea, _, _, _ = self.encoder((xyz, xyz, None, None, None))
        fea = self.mlp_enc_out(fea)
        fea = fea * self.cfg.bottleneck_scaler

        if self.training:
            fea, likelihoods = self.entropy_bottleneck(fea.permute(0, 2, 1).unsqueeze(3).contiguous())
            fea = fea / self.cfg.bottleneck_scaler
            fea = fea.squeeze(3).permute(0, 2, 1).contiguous()
            likelihoods = likelihoods.squeeze(3).permute(0, 2, 1).contiguous()
            fea = self.decoder(fea)
            # fea = fea.reshape(batch_size, self.cfg.input_points_num, 3)

            bpp_loss = torch.log2(likelihoods).sum() * (-self.cfg.bpp_loss_factor / (xyz.shape[0] * xyz.shape[1]))
            reconstruct_loss = loss_function.chamfer_loss(fea, xyz)
            aux_loss = self.entropy_bottleneck.loss() * self.cfg.aux_loss_factor
            loss = reconstruct_loss + bpp_loss + aux_loss

            return {'aux_loss': aux_loss.detach().cpu().item(),
                    'bpp_loss': bpp_loss.detach().cpu().item(),
                    'reconstruct_loss': reconstruct_loss.detach().cpu().item(),
                    'loss': loss}
        else:
            compressed_strings = self.entropy_bottleneck_compress(fea)
            decompressed_tensors = self.entropy_bottleneck_decompress(compressed_strings)
            fea = decompressed_tensors / self.cfg.bottleneck_scaler
            fea = self.decoder(fea)
            fea = fea.reshape(batch_size, self.cfg.input_points_num, 3)

            return {'compressed_strings': compressed_strings,
                    'decompressed_tensors': decompressed_tensors,
                    'decoder_output': fea}

    def load_state_dict(self, state_dict, strict: bool = True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    def entropy_bottleneck_compress(self, encoder_output):
        assert not self.training
        encoder_output = encoder_output.permute(0, 2, 1).unsqueeze(3).contiguous()
        return self.entropy_bottleneck.compress(encoder_output)

    def entropy_bottleneck_decompress(self, compressed_strings):
        assert not self.training
        decompressed_tensors = self.entropy_bottleneck.decompress(compressed_strings, size=(self.encoded_points_num, 1))
        decompressed_tensors = decompressed_tensors.squeeze(3).permute(0, 2, 1)
        return decompressed_tensors


def main_t():
    from thop import profile
    from thop import clever_format

    cfg = ModelConfig()
    torch.cuda.set_device('cuda:3')
    model = PointCompressor(cfg).cuda()
    model.train()
    batch_points = torch.rand(2, cfg.input_points_num, 3).cuda()
    out = model(batch_points)
    model.entropy_bottleneck.update()
    model.eval()
    test_out = model(batch_points)

    macs, params = profile(model, inputs=(batch_points,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f'macs: {macs}, params: {params}')  # macs: 10.924G, params: 67.639M

    print('Done')


if __name__ == '__main__':
    main_t()
