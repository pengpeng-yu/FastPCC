import torch
import torch.nn as nn
import torch.nn.functional as F
import compressai
from compressai.models.utils import update_registered_buffers

from lib import loss_function
from lib.points_layers import TransitionDown, RandLANeighborFea, LocalFeatureAggregation as LFA
from lib.torch_utils import MLPBlock
from models.exp4.model_config import ModelConfig


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        neighbor_fea_generator = RandLANeighborFea(cfg.neighbor_num)

        self.encoder_layers = [LFA(3, neighbor_fea_generator, 16, 24),
                               LFA(24, neighbor_fea_generator, 16, 32),
                               LFA(32, neighbor_fea_generator, 16, 48),
                               LFA(48, neighbor_fea_generator, 24, 48),
                               LFA(48, neighbor_fea_generator, 24, 64),
                               LFA(64, neighbor_fea_generator, 24, 64),
                               TransitionDown(None, 0.5, 'uniform'),
                               LFA(64, neighbor_fea_generator, 24, 128),
                               LFA(128, neighbor_fea_generator, 32, 128),
                               TransitionDown(None, 0.5, 'uniform'),
                               LFA(128, neighbor_fea_generator, 32, 256),
                               LFA(256, neighbor_fea_generator, 32, 256),
                               TransitionDown(None, 0.5, 'uniform'),
                               LFA(256, neighbor_fea_generator, 32, 512),
                               LFA(512, neighbor_fea_generator, 32, 512),
                               TransitionDown(None, 0.5, 'uniform'),
                               LFA(512, neighbor_fea_generator, 64, 1024),
                               LFA(1024, neighbor_fea_generator, 128, 1024),
                               TransitionDown(None, 0.5, 'uniform')]

        self.encoded_points_num = cfg.input_points_num // 32
        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        self.mlp_enc_out = nn.Sequential(MLPBlock(1024, 1024, activation='leaky_relu(0.2)', batchnorm='nn.bn1d'),
                                         MLPBlock(1024, 1024, activation=None, batchnorm='nn.bn1d'))

        self.entropy_bottleneck = compressai.entropy_models.EntropyBottleneck(1024)

        self.decoder_layers = [LFA(1024, neighbor_fea_generator, 128, 1024),
                               LFA(1024, neighbor_fea_generator, 128, 1024)]
        self.decoder_layers = nn.Sequential(*self.decoder_layers)
        self.mlp_dec_out = nn.Sequential(MLPBlock(1024, 256, activation='leaky_relu(0.2)', batchnorm='nn.bn1d'),
                                         MLPBlock(256, 96, activation=None, batchnorm='nn.bn1d'))

        self.init_weights()

    def forward(self, fea):
        if self.training: ori_fea = fea
        batch_size = fea.shape[0]
        xyz = fea[..., :3]  # B, N, C
        # encode
        xyz, fea = self.encoder_layers((xyz, fea, None, None))[:2]
        fea = self.mlp_enc_out(fea)

        if self.training:
            fea, likelihoods = self.entropy_bottleneck(fea.permute(0, 2, 1).unsqueeze(3).contiguous())
            fea = fea.squeeze(3).permute(0, 2, 1).contiguous()
            likelihoods = likelihoods.squeeze(3).permute(0, 2, 1).contiguous()
            fea = self.decoder_layers((xyz, fea, None, None))[1]
            fea = self.mlp_dec_out(fea)
            fea = fea.reshape(batch_size, self.cfg.input_points_num, 3)

            bpp_loss = torch.log2(likelihoods).sum() * (-self.cfg.bpp_loss_factor / (ori_fea.shape[0] * ori_fea.shape[1]))
            reconstruct_loss = loss_function.chamfer_loss(fea, ori_fea)
            aux_loss = self.entropy_bottleneck.loss() * self.cfg.aux_loss_factor
            loss = reconstruct_loss + bpp_loss + aux_loss
            return {'aux_loss': aux_loss.detach().cpu().item(),
                    'bpp_loss': bpp_loss.detach().cpu().item(),
                    'reconstruct_loss': reconstruct_loss.detach().cpu().item(),
                    'loss': loss}
        else:
            compressed_strings = self.entropy_bottleneck_compress(fea)
            decompressed_tensors = self.entropy_bottleneck_decompress(compressed_strings)
            fea = self.decoder_layers((xyz, decompressed_tensors, None, None))[1]
            fea = self.mlp_dec_out(fea)
            fea = fea.reshape(batch_size, self.cfg.input_points_num, 3)

            return {'compressed_strings': compressed_strings,
                    'decompressed_tensors': decompressed_tensors,
                    'decoder_output': fea}

    def init_weights(self):
        torch.nn.init.uniform_(self.mlp_enc_out[-1].bn.weight, -10, 10)
        torch.nn.init.uniform_(self.mlp_enc_out[-1].bn.bias, -10, 10)

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
    cfg = ModelConfig()
    cfg.input_points_num = 1024
    torch.cuda.set_device('cuda:1')
    model = PointCompressor(cfg).cuda()
    model.train()
    batch_points = torch.rand(1, cfg.input_points_num, 3).cuda()
    out =  model(batch_points)
    print('Done')

if __name__ == '__main__':
    main_t()