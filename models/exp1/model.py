import torch
import torch.nn as nn
import torch.nn.functional as F
import compressai
from compressai.models.utils import update_registered_buffers

from lib import loss_function
from lib.torch_utils import MLPBlock
from lib.points_layers import PointLayerMessage, TransitionDown, TransformerBlock
from models.exp1.model_config import ModelConfig


class PointEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(PointEncoder, self).__init__()
        self.top_down_blocks = nn.Sequential(TransformerBlock(3, 64, cfg.neighbor_num),
                                             TransformerBlock(64, 128, cfg.neighbor_num),
                                             TransitionDown('uniform', 0.5),

                                             TransformerBlock(128, 256, cfg.neighbor_num),
                                             TransitionDown('uniform', 0.5),

                                             TransformerBlock(256, 512, cfg.neighbor_num),
                                             TransitionDown('uniform', 0.25,
                                                            cache_sampled_xyz=True, cache_sampled_feature=True),

                                             TransformerBlock(512, 1024, cfg.neighbor_num),
                                             TransitionDown('uniform', 0.25,
                                                            cache_sampled_xyz=True, cache_sampled_feature=True),

                                             TransformerBlock(1024, 1024, cfg.neighbor_num),
                                             TransitionDown('uniform', 0.25))

        self.transition_blocks = nn.ModuleList()
        self.transition_blocks.extend((nn.Sequential(TransformerBlock(512, 1024, cfg.neighbor_num),
                                                     TransitionDown('uniform', nsample=cfg.input_points_num // 16)),
                                       TransformerBlock(1024, 512, cfg.neighbor_num),
                                       TransformerBlock(1024, 1024, cfg.neighbor_num)))

        self.init_weights()

    def forward(self, msg: PointLayerMessage):
        msg = self.top_down_blocks(msg)  # type: PointLayerMessage

        msg = self.transition_blocks[2](msg)

        msg.xyz = torch.cat((msg.cached_xyz[1], msg.xyz), dim=1)
        msg.feature = torch.cat((msg.cached_feature[1], msg.feature), dim=1)
        msg.raw_neighbors_feature = msg.neighbors_idx = None

        msg = self.transition_blocks[1](msg)
        msg.xyz = torch.cat((msg.cached_xyz[0], msg.xyz), dim=1)
        msg.feature = torch.cat((msg.cached_feature[0], msg.feature), dim=1)
        msg.raw_neighbors_feature = msg.neighbors_idx = None

        msg = self.transition_blocks[0](msg)
        return msg

    def init_weights(self):
        torch.nn.init.uniform_(self.transition_blocks[0][0].fc2.weight, -10, 10)
        torch.nn.init.uniform_(self.transition_blocks[0][0].fc2.bias, -10, 10)
        torch.nn.init.uniform_(self.transition_blocks[0][0].shortcut_fc.weight, -10, 10)
        torch.nn.init.uniform_(self.transition_blocks[0][0].shortcut_fc.bias, -10, 10)


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = PointEncoder(cfg)
        self.encoded_points_num = cfg.input_points_num // 16
        self.encoded_points_dim = 1024

        self.entropy_bottleneck = compressai.entropy_models.EntropyBottleneck(self.encoded_points_dim)

        self.decoder_in_mlp = MLPBlock(1024, 1024, bn='nn.bn1d', act='leaky_relu(0.2)')
        self.decoder = nn.Sequential(
            TransformerBlock(1024, 512, cfg.neighbor_num, cache_out_feature=True),
            TransformerBlock(512, 256, cfg.neighbor_num, cache_out_feature=True),
            TransformerBlock(256, 48, cfg.neighbor_num, cache_out_feature=True))

    def forward(self, fea):
        if self.training: ori_fea = fea
        batch_size = fea.shape[0]
        xyz = fea[..., :3]  # B, N, C only coordinate supported
        # encode
        msg = self.encoder(PointLayerMessage(xyz=xyz, feature=fea))
        xyz, fea = msg.xyz, msg.feature

        if self.training:
            fea, likelihoods = self.entropy_bottleneck(fea.permute(0, 2, 1).unsqueeze(3).contiguous())
            fea = fea.squeeze(3).permute(0, 2, 1).contiguous()
            likelihoods = likelihoods.squeeze(3).permute(0, 2, 1).contiguous()
            fea = self.decoder_in_mlp(fea)

            msg = self.decoder(PointLayerMessage(xyz=xyz, feature=fea))
            fea_list = [msg.cached_feature[0].reshape(batch_size, self.cfg.input_points_num // 8, -1)[:, :, :3],
                        msg.cached_feature[1].reshape(batch_size, self.cfg.input_points_num // 4, -1)[:, :, :3],
                        msg.cached_feature[2].reshape(batch_size, self.cfg.input_points_num, self.cfg.input_points_dim)]

            bpp_loss = torch.log2(likelihoods).sum() * \
                (-self.cfg.bpp_loss_factor / (ori_fea.shape[0] * ori_fea.shape[1]))

            reconstruct_regular_loss = loss_function.chamfer_loss(fea_list[0], ori_fea[:, :, :3]) * 0.05 + \
                                       loss_function.chamfer_loss(fea_list[1], ori_fea[:, :, :3]) * 0.1

            reconstruct_loss = loss_function.chamfer_loss(fea_list[2], ori_fea)

            aux_loss = self.entropy_bottleneck.loss()

            loss = reconstruct_regular_loss + reconstruct_loss + bpp_loss + aux_loss

            return {'aux_loss': aux_loss.detach().cpu().item(),
                    'bpp_loss': bpp_loss.detach().cpu().item(),
                    'reconstruct_loss': reconstruct_loss.detach().cpu().item(),
                    'reconstruct_regular_loss': reconstruct_regular_loss.detach().cpu().item(),
                    'loss': loss}
        else:
            compressed_strings = self.entropy_bottleneck_compress(fea)
            decompressed_fea = self.entropy_bottleneck_decompress(compressed_strings)
            decompressed_fea = self.decoder_in_mlp(decompressed_fea)

            decoder_output = self.decoder(PointLayerMessage(xyz=xyz, feature=decompressed_fea)).feature

            decoder_output = decoder_output.reshape(batch_size, self.cfg.input_points_num, self.cfg.input_points_dim)

            return {'encoder_output': fea,
                    'compressed_strings': compressed_strings,
                    'decompressed_fea': decompressed_fea,
                    'decoder_output': decoder_output}

    def load_state_dict(self, state_dict, strict: bool = True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        return super().load_state_dict(state_dict, strict=strict)

    def entropy_bottleneck_compress(self, encoder_output):
        assert not self.training
        encoder_output = encoder_output.permute(0, 2, 1).unsqueeze(3).contiguous()
        return self.entropy_bottleneck.compress(encoder_output)

    def entropy_bottleneck_decompress(self, compressed_strings):
        assert not self.training
        decompressed_fea = self.entropy_bottleneck.decompress(compressed_strings, size=(self.encoded_points_num, 1))
        decompressed_fea = decompressed_fea.squeeze(3).permute(0, 2, 1)
        return decompressed_fea


def main_t():
    torch.cuda.set_device('cuda:2')
    cfg = ModelConfig()
    cfg.input_points_num = 2048
    model = PointCompressor(cfg)
    model = model.cuda()
    point_cloud = torch.rand(4, cfg.input_points_num, cfg.input_points_dim, device='cuda')
    out = model(point_cloud)  # 4, 16, 256
    out['loss'].backward()
    model.eval()
    model.entropy_bottleneck.update()
    val_out = model(point_cloud)


def point_encoder_t():
    cfg = ModelConfig()
    point_encoder = PointEncoder(cfg)
    pc = torch.rand(2, 8192, 3)
    output = point_encoder(pc)
    print('Done')


if __name__ == '__main__':
    main_t()
