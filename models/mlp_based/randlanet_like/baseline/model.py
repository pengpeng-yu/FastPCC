import io
from typing import Tuple, List, Dict, Iterable

import torch
import torch.nn as nn

from lib.data_utils import PCData
from lib.entropy_models.continuous_batched import NoisyDeepFactorizedEntropyModel
from lib.entropy_models.continuous_indexed import ContinuousNoisyDeepFactorizedIndexedEntropyModel
from lib.loss_functions import chamfer_loss
from lib.evaluators import PCGCEvaluator
from lib.points_layers import PointLayerMessage, TransitionDown, RandLANeighborFea, \
    LocalFeatureAggregation as LFA
from lib.torch_utils import MLPBlock, concat_loss_dicts
from models.mlp_based.randlanet_like.baseline.model_config import ModelConfig


class Decoder(nn.Module):
    def __init__(self, channels: Tuple[int], neighbor_feature_channels: Tuple[int],
                 neighbor_fea_generator: RandLANeighborFea,
                 sample_rate: float, res_em_index_ranges: Tuple[int]):
        super(Decoder, self).__init__()
        self.upsample_rate = int(1 / sample_rate)
        self.blocks_num = len(channels) - 1

        self.lfa_list = nn.ModuleList([
            LFA(channels[idx],
                neighbor_fea_generator,
                neighbor_feature_channels[idx],
                channels[idx] * self.upsample_rate)
            for idx in range(self.blocks_num)
        ])
        self.res_lfa_list = nn.ModuleList([
            LFA(channels[idx + 1],
                neighbor_fea_generator,
                neighbor_feature_channels[idx + 1],
                channels[idx + 1],
                use_shortcut=False)
            for idx in range(self.blocks_num - 1)
        ])
        self.mlp_pred_coord_list = nn.ModuleList([
            MLPBlock(
                channels[idx], 3, bn='nn.bn1d', act=None
            ) for idx in range(self.blocks_num)
        ])
        self.mlp_pred_fea_list = nn.ModuleList([
            MLPBlock(
                channels[idx],
                channels[idx + 1] * len(res_em_index_ranges),
                bn='nn.bn1d', act=None
            ) for idx in range(self.blocks_num - 1)
        ])

        def parameter_fns_factory(in_channels, out_channels):
            return nn.Sequential(
                MLPBlock(in_channels, out_channels,
                         bn=None, act='leaky_relu(0.2)'),
                nn.Linear(out_channels, out_channels,
                          bias=True)
            )
        self.res_em = ContinuousNoisyDeepFactorizedIndexedEntropyModel(
            res_em_index_ranges, 2,
            parameter_fns_factory=parameter_fns_factory,
            quantize_indexes=True
        )

    def forward(self, msg: PointLayerMessage,
                encoder_cached_feature: List[torch.Tensor],
                encoder_cached_xyz: List[torch.Tensor],
                loss_dict: Dict[str, torch.Tensor]):

        for idx in range(self.blocks_num):
            self.update_msg(msg, idx)
            if idx != self.blocks_num - 1:
                res_feature = self.res_lfa_list[idx](PointLayerMessage(
                    torch.tensor(()), encoder_cached_feature.pop(),
                    *self.res_lfa_list[idx].neighbor_feature_generator(encoder_cached_xyz.pop(), msg.xyz)
                )).feature
                pred_feature = self.mlp_pred_fea_list[idx](msg.feature)

                msg.feature, sub_loss_dict = self.res_em(res_feature, pred_feature)
                concat_loss_dicts(loss_dict, sub_loss_dict, lambda k: f'{msg.feature.shape[2]}c_' + k)
        return msg

    def compress(self, msg: PointLayerMessage,
                 encoder_cached_feature: List[torch.Tensor],
                 encoder_cached_xyz: List[torch.Tensor]):
        bytes_dict = {}

        for idx in range(self.blocks_num - 1):
            self.update_msg(msg, idx)
            res_feature = self.res_lfa_list[idx](PointLayerMessage(
                torch.tensor(()), encoder_cached_feature.pop(),
                *self.res_lfa_list[idx].neighbor_feature_generator(encoder_cached_xyz.pop(), msg.xyz)
            )).feature
            pred_feature = self.mlp_pred_fea_list[idx](msg.feature)
            bytes_strings, msg.feature = self.res_em.compress(
                res_feature, pred_feature, return_dequantized=True
            )
            bytes_dict[f'{msg.feature.shape[2]}c'] = bytes_strings[0]
        return msg, self.concat_strings(bytes_dict.values())

    def decompress(self, concat_string: bytes,
                   msg: PointLayerMessage,
                   target_device: torch.device):
        fea_bytes_list = self.split_strings(concat_string)
        for idx in range(self.blocks_num):
            self.update_msg(msg, idx)
            if idx != self.blocks_num - 1:
                pred_feature = self.mlp_pred_fea_list[idx](msg.feature)
                msg.feature = self.res_em.decompress([fea_bytes_list[idx]], pred_feature, target_device)
        return msg

    def update_msg(self, msg: PointLayerMessage, decoder_block_idx: int):
        """
        Update cached xyz, xyz and feature
        """
        msg: PointLayerMessage = self.lfa_list[decoder_block_idx](msg)
        batch_size, points_num, channels = msg.feature.shape
        assert channels == self.lfa_list[decoder_block_idx].out_channels
        msg.feature = msg.feature.contiguous().view(
            batch_size, points_num, self.upsample_rate, channels // self.upsample_rate
        )

        pred_offset = self.mlp_pred_coord_list[decoder_block_idx](msg.feature)
        pred_coord = msg.xyz.unsqueeze(2) + pred_offset
        pred_coord = pred_coord.reshape(batch_size, points_num * self.upsample_rate, 3)
        msg.cached_xyz.append(pred_coord)
        msg.xyz = pred_coord.detach()
        msg.feature = msg.feature.reshape(
            batch_size, points_num * self.upsample_rate, channels // self.upsample_rate
        )
        msg.neighbors_idx = msg.raw_neighbors_feature = None

    @staticmethod
    def concat_strings(bytes_list: Iterable[bytes]) -> bytes:
        with io.BytesIO() as bs:
            for bytes_string in bytes_list:
                bs.write(len(bytes_string).to_bytes(4, 'little', signed=False))  # TODO 2
                bs.write(bytes_string)
            concat_bytes = bs.getvalue()
        return concat_bytes

    @staticmethod
    def split_strings(concat_bytes: bytes) -> List[bytes]:
        concat_bytes_len = len(concat_bytes)
        bytes_list = []
        with io.BytesIO(concat_bytes) as bs:
            while bs.tell() != concat_bytes_len:
                length = int.from_bytes(bs.read(4), 'little', signed=False)
                bytes_list.append(bs.read(length))
        return bytes_list


class PCC(nn.Module):

    @staticmethod
    def params_divider(s: str) -> int:
        if s.endswith("aux_param"): return 2
        else:
            if 'em' not in s: return 0
            else: return 1

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.evaluator = PCGCEvaluator(
            cfg.mpeg_pcc_error_command,
            cfg.mpeg_pcc_error_threads,
            cfg.chamfer_dist_test_phase
        )
        neighbor_fea_generator = RandLANeighborFea(cfg.neighbor_num)
        blocks_num = len(cfg.channels) - 1

        self.encoder = nn.Sequential(
            *[nn.Sequential(
                LFA(cfg.channels[idx],
                    neighbor_fea_generator,
                    cfg.neighbor_feature_channels[idx],
                    cfg.channels[idx + 1]),
                LFA(cfg.channels[idx + 1],
                    neighbor_fea_generator,
                    cfg.neighbor_feature_channels[idx],
                    cfg.channels[idx + 1]),
                TransitionDown(
                    cfg.sample_method, cfg.sample_rate,
                    cache_sampled_xyz=idx != blocks_num - 1,
                    cache_sampled_feature=idx != blocks_num - 1
                )
            ) for idx in range(blocks_num)],
        )
        self.encoder_out_mlp = MLPBlock(
            cfg.channels[-1], cfg.compressed_channels,
            bn='nn.bn1d', act=None
        )

        self.em = NoisyDeepFactorizedEntropyModel(
            batch_shape=torch.Size([cfg.compressed_channels]),
            broadcast_shape_bytes=(3,),
            coding_ndim=2,
            init_scale=5
        )

        self.decoder_in_mlp = MLPBlock(
            cfg.compressed_channels, cfg.channels[-1],
            bn='nn.bn1d', act='leaky_relu(0.2)'
        )
        self.decoder = Decoder(
            cfg.channels[::-1],
            cfg.neighbor_feature_channels[::-1],
            neighbor_fea_generator,
            cfg.sample_rate,
            cfg.res_em_index_ranges
        )

    def forward(self, pc_data: PCData):
        if not (pc_data.color is None and pc_data.normal is None):
            raise NotImplementedError

        if self.training:
            enc_msg: PointLayerMessage = self.encoder(
                PointLayerMessage(xyz=pc_data.xyz, feature=pc_data.xyz)
            )
            enc_msg.feature = self.encoder_out_mlp(enc_msg.feature).contiguous()
            enc_msg.feature, loss_dict = self.em(enc_msg.feature)

            dec_msg = PointLayerMessage(
                xyz=enc_msg.xyz, feature=enc_msg.feature,
                raw_neighbors_feature=enc_msg.raw_neighbors_feature,
                neighbors_idx=enc_msg.neighbors_idx
            )
            dec_msg.feature = self.decoder_in_mlp(dec_msg.feature).contiguous()
            dec_msg: PointLayerMessage = self.decoder(
                dec_msg, list(enc_msg.cached_feature),
                enc_msg.cached_xyz.copy(), loss_dict
            )

            for idx, recon_p in enumerate(dec_msg.cached_xyz):
                loss_dict[f'reconstruct_{idx}_loss'] = chamfer_loss(
                    recon_p,
                    enc_msg.cached_xyz[-idx - 1] if idx != len(dec_msg.cached_xyz) - 1 else pc_data.xyz
                ) * self.cfg.reconstruct_loss_factor
            for key in loss_dict:
                if key.endswith('bits_loss'):
                    loss_dict[key] = loss_dict[key] * (self.cfg.bpp_loss_factor / pc_data.xyz.shape[1])

            loss_dict['loss'] = sum(loss_dict.values())
            for key in loss_dict:
                if key != 'loss':
                    loss_dict[key] = loss_dict[key].item()
            return loss_dict

        else:
            assert pc_data.batch_size == 1, 'Only supports batch size == 1 during testing.'
            if isinstance(pc_data.xyz, torch.Tensor):
                raise NotImplementedError

            else:
                assert isinstance(pc_data.xyz, List)
                pc_recon_list = []
                bottom_bytes_list = []
                res_bytes_list = []
                for sub_xyz in pc_data.xyz[1:]:
                    enc_msg: PointLayerMessage = self.encoder(
                        PointLayerMessage(xyz=sub_xyz, feature=sub_xyz)
                    )
                    enc_msg.feature = self.encoder_out_mlp(enc_msg.feature).contiguous()
                    (bottom_bytes, ), coding_batch_shape, bottom_fea_recon = self.em.compress(
                        enc_msg.feature, return_dequantized=True
                    )
                    bottom_bytes_list.append(bottom_bytes)

                    dec_msg = PointLayerMessage(xyz=enc_msg.xyz, feature=bottom_fea_recon)
                    dec_msg.feature = self.decoder_in_mlp(dec_msg.feature).contiguous()
                    dec_msg, res_bytes = self.decoder.compress(
                        dec_msg, list(enc_msg.cached_feature), enc_msg.cached_xyz
                    )
                    res_bytes_list.append(res_bytes)

                    torch.cuda.empty_cache()
                    # TODO: split enc and dec into individual loops

                    target_device = next(self.parameters()).device
                    bottom_fea_recon_ = self.em.decompress([bottom_bytes], coding_batch_shape, target_device)
                    dec_msg = PointLayerMessage(xyz=enc_msg.xyz, feature=bottom_fea_recon_)
                    dec_msg.feature = self.decoder_in_mlp(dec_msg.feature).contiguous()
                    dec_msg = self.decoder.decompress(
                        res_bytes, dec_msg, target_device
                    )
                    pc_recon_list.append(dec_msg.xyz)

                pc_recon = torch.cat([pc_recon for pc_recon in pc_recon_list], 1)
                with io.BytesIO() as bs:
                    for bytes_string in bottom_bytes_list:
                        bs.write(len(bytes_string).to_bytes(4, 'little', signed=False))
                        bs.write(bytes_string)
                    for bytes_string in res_bytes_list:
                        bs.write(len(bytes_string).to_bytes(4, 'little', signed=False))
                        bs.write(bytes_string)
                    concat_bytes = bs.getvalue()
                ret = self.evaluator.log_batch(
                    preds=(pc_recon * pc_data.ori_resolution[0]).round().to('cpu', torch.int32),
                    targets=[pc_data.xyz[0]],
                    compressed_bytes_list=[concat_bytes],
                    pc_data=pc_data
                )
                return ret

    def train_forward(self):
        pass

    def test_forward(self):
        pass

    def test_partitions_forward(self):
        pass

    def compress(self):
        pass

    def decompress(self):
        pass

    def train(self, mode: bool = True):
        """
        Use model.train() to reset evaluator.
        """
        if mode is True:
            self.evaluator.reset()
        return super(PCC, self).train(mode=mode)


def main_t():
    cfg = ModelConfig()
    torch.cuda.set_device('cuda:0')
    model = PCC(cfg).cuda()
    model.train()
    pc_data = PCData(torch.rand(2, 1024, 3).cuda(), batch_size=2)
    out = model(pc_data)
    out['loss'].backward()
    model.eval()
    pc_data = PCData(torch.rand(1, 1024, 3).cuda(), batch_size=1)
    pc_data.xyz = [pc_data.xyz, pc_data.xyz[:, :512], pc_data.xyz[:, 512:]]
    test_out = model(pc_data)

    print('Done')


if __name__ == '__main__':
    main_t()
