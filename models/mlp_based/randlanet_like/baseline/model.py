from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.data_utils import PCData
from lib.entropy_models.continuous_batched import NoisyDeepFactorizedEntropyModel
from lib.loss_functions import chamfer_loss
from lib.points_layers import PointLayerMessage, TransitionDown, RandLANeighborFea, \
    LocalFeatureAggregation as LFA
from lib.torch_utils import MLPBlock
from models.mlp_based.randlanet_like.baseline.model_config import ModelConfig


class GenerativeTransitionUp(nn.Module):
    def __init__(self, lfa: LFA, upsample_rate: int):
        super(GenerativeTransitionUp, self).__init__()
        self.lfa = lfa
        self.mlp_pred = MLPBlock(lfa.out_channels // upsample_rate, 3, bn='nn.bn1d', act=None)
        self.upsample_rate = upsample_rate

    def forward(self, msg: PointLayerMessage):
        msg: PointLayerMessage = self.lfa(msg)
        msg.raw_neighbors_feature = msg.neighbors_idx = None
        batch_size, points_num, channels = msg.feature.shape
        assert channels == self.lfa.out_channels

        msg.feature = msg.feature.contiguous().view(
            batch_size, points_num, self.upsample_rate, channels // self.upsample_rate
        )
        pred_offset = self.mlp_pred(msg.feature)

        pred_coord = msg.xyz.unsqueeze(2) + pred_offset
        pred_coord = pred_coord.reshape(batch_size, points_num * self.upsample_rate, 3)
        msg.cached_xyz.append(pred_coord)

        msg.xyz = pred_coord.detach()
        msg.feature = msg.feature.reshape(
            batch_size, points_num * self.upsample_rate, channels // self.upsample_rate
        )
        msg.cached_feature.append(msg.feature)
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
                    ch, cache_out_feature=True),
                TransitionDown(
                    cfg.sample_method, cfg.sample_rate,
                    cache_sampled_xyz=True)
            ) for idx, ch in enumerate(cfg.encoder_channels)],

            nn.Sequential(
                *[LFA(cfg.encoder_channels[-1],
                      neighbor_fea_generator,
                      cfg.encoder_neighbor_feature_channels[-1],
                      cfg.encoder_channels[-1]
                      ) for _ in range(2)]
            )
        )

        self.encoder_out_mlp = MLPBlock(
            cfg.encoder_channels[-1], cfg.compressed_channels,
            bn='nn.bn1d', act=None
        )

        self.em = NoisyDeepFactorizedEntropyModel(
            batch_shape=torch.Size([cfg.compressed_channels]),
            broadcast_shape_bytes=(3,),
            coding_ndim=2,
            init_scale=5
        )

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
        encoder_msg.feature = self.encoder_out_mlp(encoder_msg.feature).contiguous()
        encoder_msg.feature *= self.cfg.bottleneck_scaler

        if self.training:
            encoder_msg.feature, loss_dict = self.em(encoder_msg.feature)
            encoder_msg.feature /= self.cfg.bottleneck_scaler
            decoder_msg: PointLayerMessage = self.decoder(
                PointLayerMessage(
                    xyz=encoder_msg.xyz, feature=encoder_msg.feature,
                    raw_neighbors_feature=encoder_msg.raw_neighbors_feature,
                    neighbors_idx=encoder_msg.neighbors_idx)
            )

            loss_dict['reconstruct_loss'] = sum(
                [chamfer_loss(p, pc_data.xyz) for p in decoder_msg.cached_xyz]
            ) * self.cfg.reconstruct_loss_factor
            loss_dict['bits_loss'] = loss_dict['bits_loss'] * \
                (self.cfg.bpp_loss_factor / pc_data.xyz.shape[0] / pc_data.xyz.shape[1])
            loss_dict['loss'] = sum(loss_dict.values())
            for key in loss_dict:
                if key != 'loss':
                    loss_dict[key] = loss_dict[key].item()
            return loss_dict

        else:
            fea_recon, compressed_strings, coding_batch_shape = self.em(encoder_msg.feature)
            fea_recon = fea_recon / self.cfg.bottleneck_scaler
            decoder_msg: PointLayerMessage = self.decoder(
                PointLayerMessage(xyz=encoder_msg.xyz, feature=fea_recon)
            )
            pc_recon = decoder_msg.cached_feature[-1]
            return pc_recon, compressed_strings


def main_t():
    cfg = ModelConfig()
    torch.cuda.set_device('cuda:0')
    model = PCC(cfg).cuda()
    model.train()
    pc_data = PCData(torch.rand(2, 1024, 3).cuda())
    out = model(pc_data)
    out['loss'].backward()
    model.eval()
    test_out = model(pc_data)

    print('Done')


if __name__ == '__main__':
    main_t()
