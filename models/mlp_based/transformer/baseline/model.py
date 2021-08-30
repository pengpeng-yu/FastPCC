import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.entropy_models.continuous_batched import NoisyDeepFactorizedEntropyModel

from lib.loss_functions import chamfer_loss
from lib.torch_utils import MLPBlock
from lib.data_utils import PCData
from lib.points_layers import PointLayerMessage, TransitionDown, TransformerBlock
from models.mlp_based.transformer.baseline.model_config import ModelConfig


class PCC(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.encoder = nn.Sequential(
            *[nn.Sequential(
                TransformerBlock(cfg.encoder_channels[idx - 1]
                                 if idx != 0 else 3,
                                 ch, cfg.neighbor_num),
                TransitionDown(cfg.sample_method, cfg.sample_rate))
                for idx, ch in enumerate(cfg.encoder_channels)],

            TransformerBlock(cfg.encoder_channels[-1],
                             cfg.compressed_channels,
                             cfg.neighbor_num)
        )

        self.encoder_out_mlp = MLPBlock(
            cfg.compressed_channels, cfg.compressed_channels,
            bn='nn.bn1d', act=None)

        self.entropy_bottleneck = NoisyDeepFactorizedEntropyModel(
            batch_shape=torch.Size([cfg.compressed_channels]),
            coding_ndim=2,
            init_scale=10
        )

        self.decoder = nn.Sequential(
            *[TransformerBlock(
                cfg.decoder_channels[idx - 1]
                if idx != 0 else cfg.compressed_channels,
                ch, cfg.neighbor_num
            ) for idx, ch in enumerate(cfg.decoder_channels)],
        )

        self.decoder_out_mlp = MLPBlock(
            cfg.decoder_channels[-1], cfg.decoder_channels[-1],
            bn='nn.bn1d', act=None)

        self.cfg = cfg
        self.init_weights()

    def forward(self, pc_data: PCData):
        if not (pc_data.colors is None and pc_data.normals is None):
            raise NotImplementedError

        encoder_msg: PointLayerMessage = self.encoder(
            PointLayerMessage(xyz=pc_data.xyz, feature=pc_data.xyz)
        )
        feature = self.encoder_out_mlp(encoder_msg.feature)
        feature = feature.contiguous()

        if self.training:
            fea_tilde, loss_dict = self.entropy_bottleneck(feature)
            decoder_msg: PointLayerMessage = self.decoder(
                PointLayerMessage(
                    xyz=encoder_msg.xyz,
                    feature=fea_tilde))
            pc_recon = self.decoder_out_mlp(decoder_msg.feature)
            pc_recon = pc_recon.reshape(pc_data.xyz.shape)

            loss_dict['reconstruct_loss'] = chamfer_loss(pc_recon, pc_data.xyz) * \
                self.cfg.reconstruct_loss_factor
            loss_dict['bits_loss'] = loss_dict['bits_loss'] * \
                (self.cfg.bpp_loss_factor / pc_data.xyz.shape[0] / pc_data.xyz.shape[1])

            loss_dict['loss'] = sum(loss_dict.values())
            for key in loss_dict:
                if key != 'loss':
                    loss_dict[key] = loss_dict[key].detach().cpu().item()

            return loss_dict

        else:
            fea_recon, loss_dict, compressed_strings = self.entropy_bottleneck(feature)
            decoder_msg = self.decoder(
                PointLayerMessage(
                    xyz=encoder_msg.xyz,
                    feature=fea_recon)
            )
            pc_recon = self.decoder_out_mlp(decoder_msg.feature)
            pc_recon = pc_recon.reshape(pc_data.xyz.shape)

            return pc_recon, loss_dict['bits_loss'], compressed_strings

    def init_weights(self):
        torch.nn.init.uniform_(self.encoder_out_mlp.bn.weight, -10, 10)
        torch.nn.init.uniform_(self.encoder_out_mlp.bn.bias, -10, 10)


def main_t():
    torch.cuda.set_device('cuda:2')
    cfg = ModelConfig()
    model = PCC(cfg).cuda()
    pc_data = PCData(xyz=torch.rand(4, cfg.input_points_num, 3).cuda())
    out = model(pc_data)
    out['loss'].backward()
    model.eval()
    val_out = model(pc_data)
    print('Done')


if __name__ == '__main__':
    main_t()
