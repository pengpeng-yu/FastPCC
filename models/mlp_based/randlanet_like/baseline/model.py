from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.data_utils import PCData
from lib.entropy_models.continuous_batched import NoisyDeepFactorizedEntropyModel
from lib.loss_function import chamfer_loss
from lib.points_layers import PointLayerMessage, TransitionDown, RandLANeighborFea, \
    LocalFeatureAggregation as LFA
from lib.torch_utils import MLPBlock
from models.mlp_based.randlanet_like.baseline.model_config import ModelConfig


class GenerativeTransitionUp(nn.Module):
    def __init__(self, lfa: LFA, upsample_rate: int = 2):
        super(GenerativeTransitionUp, self).__init__()
        self.lfa = lfa
        self.mlp_pred = MLPBlock(lfa.out_channels // upsample_rate, 3, bn='nn.bn1d', act=None)
        self.upsample_rate = upsample_rate

    def forward(self, msg: PointLayerMessage):
        msg: PointLayerMessage = self.lfa(msg)
        batch_size, points_num, channels = msg.feature.shape

        msg.raw_neighbors_feature = msg.neighbors_idx = None

        msg.feature = msg.feature.contiguous().view(
            batch_size, points_num, self.upsample_rate, channels // self.upsample_rate
        )
        pred_offset = self.mlp_pred(msg.feature)

        pred_coord = msg.xyz.unsqueeze(2) + pred_offset
        pred_coord = pred_coord.reshape(batch_size, points_num * self.upsample_rate, 3)
        msg.cached_feature.append(pred_coord)

        msg.xyz = pred_coord.detach()
        msg.feature = msg.feature.reshape(
            batch_size, points_num * self.upsample_rate, channels // self.upsample_rate
        )

        return msg


class PCC(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        neighbor_fea_generator = RandLANeighborFea(cfg.neighbor_num)

        self.encoder = nn.Sequential(
            *[nn.Sequential(
                LFA(cfg.encoder_channels[idx - 1] if idx != 0 else 3,
                    neighbor_fea_generator,
                    cfg.encoder_neighbor_feature_channels[idx],
                    ch),

                LFA(ch,
                    neighbor_fea_generator,
                    cfg.encoder_neighbor_feature_channels[idx],
                    ch),

                TransitionDown(cfg.sample_method, cfg.sample_rate)

            ) for idx, ch in enumerate(cfg.encoder_channels)],

            nn.Sequential(
                *[LFA(cfg.encoder_channels[-1],
                      neighbor_fea_generator,
                      cfg.encoder_neighbor_feature_channels[-1],
                      cfg.encoder_channels[-1]
                      ) for _ in range(2)]
            )
        )

        # pooling to one point?
        self.encoder_out_mlp = MLPBlock(
            cfg.encoder_channels[-1], cfg.compressed_channels,
            bn='nn.bn1d', act=None
        )

        self.entropy_bottleneck = NoisyDeepFactorizedEntropyModel(
            batch_shape=torch.Size([cfg.compressed_channels]),
            coding_ndim=2,
            init_scale=10
        )

        # use conv1d or local-info mlp?
        self.decoder = nn.Sequential(
            *[GenerativeTransitionUp(
                LFA(int(cfg.decoder_channels[idx - 1] * cfg.sample_rate)
                    if idx != 0 else cfg.compressed_channels,
                    neighbor_fea_generator,
                    cfg.decoder_neighbor_feature_channels[idx],
                    ch),
                upsample_rate=int(1 / cfg.sample_rate)
            ) for idx, ch in enumerate(cfg.decoder_channels)]
        )

    def forward(self, pc_data: PCData):
        if not (pc_data.colors is None and pc_data.normals is None):
            raise NotImplementedError

        encoder_msg: PointLayerMessage = self.encoder(
            PointLayerMessage(xyz=pc_data.xyz, feature=pc_data.xyz)
        )
        feature = self.encoder_out_mlp(encoder_msg.feature)
        feature = feature.contiguous()
        feature = feature * self.cfg.bottleneck_scaler

        if self.training:
            fea_tilde, loss_dict = self.entropy_bottleneck(feature)
            fea_tilde = fea_tilde / self.cfg.bottleneck_scaler

            decoder_msg = self.decoder(
                PointLayerMessage(xyz=encoder_msg.xyz, feature=fea_tilde)
            )

            loss_dict['reconstruct_loss'] = sum(
                [chamfer_loss(p, pc_data.xyz) for p in decoder_msg.cached_feature]
            ) * self.cfg.reconstruct_loss_factor

            loss_dict['bits_loss'] = loss_dict['bits_loss'] * \
                (self.cfg.bpp_loss_factor / pc_data.xyz.shape[0] / pc_data.xyz.shape[1])

            loss_dict['loss'] = sum(loss_dict.values())
            for key in loss_dict:
                if key != 'loss':
                    loss_dict[key] = loss_dict[key].detach().cpu().item()

            return loss_dict

        else:
            fea_recon, loss_dict, compressed_strings = self.entropy_bottleneck(feature)
            fea_recon = fea_recon / self.cfg.bottleneck_scaler
            decoder_msg: PointLayerMessage = self.decoder(
                PointLayerMessage(xyz=encoder_msg.xyz, feature=fea_recon)
            )
            pc_recon = decoder_msg.cached_feature[-1]

            return pc_recon, loss_dict['bits_loss'], compressed_strings


def main_t():
    try:
        from thop import profile
        from thop import clever_format
        thop = True
    except ModuleNotFoundError:
        thop = False

    cfg = ModelConfig()
    torch.cuda.set_device('cuda:3')
    model = PCC(cfg).cuda()
    model.train()
    pc_data = PCData(torch.rand(2, 1024, 3).cuda())
    out = model(pc_data)
    model.eval()
    test_out = model(pc_data)

    if thop is True:
        macs, params = profile(model, inputs=(pc_data,))
        macs, params = clever_format([macs, params], "%.3f")
        print(f'macs: {macs}, params: {params}')

    print('Done')


if __name__ == '__main__':
    main_t()
