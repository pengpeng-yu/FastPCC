from typing import List, Tuple, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from pytorch3d.ops import knn_points

from lib.torch_utils import MLPBlock
from lib.sparse_conv_layers import \
    ConvBlock, ConvTransBlock, GenConvTransBlock, MEMLPBlock, ResBlock, InceptionResBlock, \
    NNSequentialWithConvTransBlockArgs, NNSequentialWithConvBlockArgs
from lib.torch_utils import minkowski_tensor_wrapped_fn


BLOCKS_LIST = [ResBlock, InceptionResBlock]
BLOCKS_DICT = {_.__name__: _ for _ in BLOCKS_LIST}
residuals_num_per_scale = 1
non_shared_scales_num = 6


def make_downsample_blocks(
        in_channels: int,
        out_channels: int,
        intra_channels: int,
        basic_block_type: str,
        region_type: str,
        basic_block_num: int,
        use_batch_norm: bool,
        act: Optional[str]) -> nn.ModuleList:
    basic_block = partial(BLOCKS_DICT[basic_block_type],
                          region_type=region_type,
                          bn=use_batch_norm, act=act)
    blocks = nn.ModuleList()
    blocks.append(nn.Sequential(
        ConvBlock(
            in_channels, intra_channels, 2, 2,
            region_type='HYPER_CUBE', bn=use_batch_norm, act=act
        ),

        *[basic_block(intra_channels) for _ in range(basic_block_num)],

        ConvBlock(
            intra_channels, out_channels, 3, 1,
            region_type=region_type, bn=use_batch_norm, act=act
        ),
    ))
    return blocks


def make_upsample_block(
        generative: bool,
        in_channels: int,
        out_channels: int,
        basic_block_type: str,
        region_type: str,
        basic_block_num: int,
        use_batch_norm: bool,
        act: Optional[str]):
    basic_block = partial(BLOCKS_DICT[basic_block_type],
                          region_type=region_type,
                          bn=use_batch_norm, act=act)
    if generative is True:
        upsample_block = GenConvTransBlock
    else:
        upsample_block = ConvTransBlock

    ret = [
        upsample_block(
            in_channels, out_channels, 2, 2,
            region_type='HYPER_CUBE', bn=use_batch_norm, act=act
        ),

        ConvBlock(
            out_channels, out_channels, 3, 1,
            region_type=region_type, bn=use_batch_norm, act=act
        ),

        *[basic_block(out_channels) for _ in range(basic_block_num)]
    ]

    if generative is True:
        return nn.Sequential(*ret)
    else:
        return NNSequentialWithConvTransBlockArgs(*ret)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 intra_channels: Tuple[int, ...],
                 requires_points_num_list: bool,
                 points_num_scaler: float,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(Encoder, self).__init__()
        self.requires_points_num_list = requires_points_num_list
        self.points_num_scaler = points_num_scaler
        self.first_block = ConvBlock(
            in_channels,
            (intra_channels[0] if len(intra_channels) > 1 else out_channels) - in_channels, 3, 1,
            region_type=region_type, bn=use_batch_norm, act=act
        )
        assert len(intra_channels) == 2
        self.blocks = make_downsample_blocks(
            intra_channels[0], out_channels, intra_channels[1],
            basic_block_type, region_type, basic_block_num,
            use_batch_norm, act
        )

    def forward(self, x) -> Tuple[List[ME.SparseTensor], Optional[List[List[int]]]]:
        points_num_list = [[_.shape[0] for _ in x.decomposed_coordinates]]
        strided_fea_list = []
        x = ME.cat([x, self.first_block(x)])
        strided_fea_list.append(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            strided_fea_list.append(x)
            if idx != len(self.blocks) - 1:
                points_num_list.append([_.shape[0] for _ in x.decomposed_coordinates])

        if not self.requires_points_num_list:
            points_num_list = None
        else:
            points_num_list = [[int(n * self.points_num_scaler) for n in _]
                               for _ in points_num_list]

        return strided_fea_list, points_num_list


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 intra_channels: int,
                 coord_recon_loss_factor: float,
                 color_recon_loss_factor: float,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(Decoder, self).__init__()
        self.coord_recon_loss_factor = coord_recon_loss_factor
        self.color_recon_loss_factor = color_recon_loss_factor
        self.upsample_block = make_upsample_block(
            True,
            in_channels, intra_channels,
            basic_block_type, region_type, basic_block_num,
            use_batch_norm, act
        )
        self.classify_block = ConvBlock(
            intra_channels, 1, 3, 1,
            region_type=region_type, bn=use_batch_norm, act=None
        )
        self.predict_block = nn.Sequential(
            MEMLPBlock(
                intra_channels + 2, intra_channels + 2, bn=use_batch_norm, act=act
            ),
            MEMLPBlock(
                intra_channels + 2, intra_channels + 2, bn=use_batch_norm, act=act
            ),
            MEMLPBlock(
                intra_channels + 2, out_channels, bn=use_batch_norm, act=None
            )
        )
        self.pruning = ME.MinkowskiPruning()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_forward(*args, **kwargs)
        else:
            return self.test_forward(*args, **kwargs)

    def train_forward(self, fea, points_num_list, target_key, target_rgb):
        assert len(points_num_list) == 1
        fea = self.upsample_block(fea)
        pred = self.classify_block(fea)
        keep = self.get_keep(pred, points_num_list[0])

        loss_dict = {}
        keep_target = self.get_target(pred, target_key)
        fea = ME.cat(
            fea, ME.SparseTensor(
                torch.stack((keep, torch.ones_like(keep, dtype=torch.float)), dim=1),
                coordinate_manager=fea.coordinate_manager,
                coordinate_map_key=fea.coordinate_map_key
            ))
        fea = self.inverse_transform_for_color(self.predict_block(fea))
        rgb_loss = self.batched_recolor(
            pred, fea.F, keep, target_key, target_rgb,
        )
        loss_dict['coord_recon_loss'] = self.get_coord_recon_loss(pred, keep_target)
        loss_dict['color_recon_loss'] = rgb_loss * self.color_recon_loss_factor
        return loss_dict

    def test_forward(self, fea, points_num_list):
        fea = self.upsample_block(fea)
        pred = self.classify_block(fea)
        keep = self.get_keep(pred, points_num_list[0])
        fea = ME.cat(
            fea, ME.SparseTensor(
                torch.stack((keep, torch.ones_like(keep, dtype=torch.float)), dim=1),
                coordinate_manager=fea.coordinate_manager,
                coordinate_map_key=fea.coordinate_map_key
            ))
        return self.inverse_transform_for_color(self.pruning(self.predict_block(fea), keep))

    @torch.no_grad()
    def get_keep(self, pred: ME.SparseTensor, points_num_list: List[int]) -> torch.Tensor:
        local_max_mask = torch.zeros((pred.shape[0]), device=pred.device, dtype=torch.bool).reshape(-1, 2 ** 3)
        local_max_mask.scatter_(1, pred.F.reshape(-1, 2 ** 3).argmax(1, keepdim=True), True)
        local_max_mask = ~local_max_mask.reshape(-1)
        if points_num_list is not None:
            sample_threshold = []
            for sample_tgt, sample_permutation in zip(points_num_list, pred.decomposition_permutations):
                sample = pred.F[sample_permutation]
                assert sample.shape[0] > sample_tgt
                sample_masked = sample[local_max_mask[sample_permutation]]
                sample_threshold.append(
                    torch.kthvalue(sample_masked, sample.shape[0] - sample_tgt, dim=0).values)
            threshold = torch.tensor(sample_threshold, device=pred.F.device, dtype=pred.F.dtype)
            threshold = threshold[pred.C[:, 0].to(torch.long)]
        else:
            threshold = 0
        keep = (pred.F.squeeze(dim=1) > threshold)
        keep.logical_or_(~local_max_mask)
        return keep

    @torch.no_grad()
    def get_target(self, pred: ME.SparseTensor, target_key: ME.CoordinateMapKey) -> torch.Tensor:
        cm = pred.coordinate_manager
        strided_target_key = cm.stride(target_key, pred.tensor_stride)
        kernel_map = cm.kernel_map(pred.coordinate_map_key, strided_target_key, kernel_size=1)
        keep_target = torch.zeros(pred.shape[0], dtype=torch.bool, device=pred.device)
        for _, curr_in in kernel_map.items():
            keep_target[curr_in[0].type(torch.long)] = 1
        return keep_target

    def get_coord_recon_loss(self, pred: ME.SparseTensor, target: torch.Tensor):
        recon_loss = F.binary_cross_entropy_with_logits(
            pred.F.squeeze(dim=1),
            target.type(pred.F.dtype)
        )

        if self.coord_recon_loss_factor != 1:
            recon_loss *= self.coord_recon_loss_factor
        return recon_loss

    @minkowski_tensor_wrapped_fn({1: 0})
    def inverse_transform_for_color(self, x):
        return x.clip_(0, 1).mul_(255)

    def batched_recolor(
            self,
            batched_pred: ME.SparseTensor,
            batched_pred_rgb: torch.Tensor,
            batched_pred_keep_mask: torch.Tensor,
            batched_tgt_coord_key: ME.CoordinateMapKey,
            batched_tgt_rgb: torch.Tensor
    ):
        mg = batched_pred.coordinate_manager
        rgb_loss_list = []
        batched_tgt_coord = mg.get_coordinates(batched_tgt_coord_key)
        for pred_row_ids, tgt_coord_row_ids in zip(
            batched_pred._batchwise_row_indices, mg.origin_map(batched_tgt_coord_key)[1]
        ):
            rgb_loss = self.sample_wise_recolor(
                batched_pred.C[pred_row_ids, 1:].to(torch.float),
                batched_pred_rgb[pred_row_ids],
                batched_pred_keep_mask[pred_row_ids],
                batched_tgt_coord[tgt_coord_row_ids, 1:].to(torch.float),
                batched_tgt_rgb[tgt_coord_row_ids]
            )
            rgb_loss_list.append(rgb_loss)
        sum_rgb_loss = sum([_.sum() for _ in rgb_loss_list]) / batched_tgt_rgb.shape[0]

        return sum_rgb_loss

    def sample_wise_recolor(
            self,
            cand_xyz: torch.Tensor,
            cand_rgb: torch.Tensor,
            pred_mask: torch.Tensor,
            tgt_xyz: torch.Tensor,
            tgt_rgb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cand_xyz: L x 3 float
            cand_rgb: L x 3 float RGB, **with gard**
            pred_mask: L bool, sum(pred_mask) == M
            tgt_xyz: N x 3 float
            tgt_rgb: L x 3 float RGB
            Note that cand_xyz[tgt_true_mask] != tgt_xyz
        """
        device = cand_xyz.device
        search_range = 8

        pred_xyz = cand_xyz[pred_mask]
        pred_rgb = cand_rgb[pred_mask]
        recolored_pred_rgb = torch.full_like(pred_xyz, 0)

        @torch.no_grad()
        def recolor_backward():
            tgt_to_pt_dist, tgt_to_pt_idx = knn_points(
                tgt_xyz[None], pred_xyz[None], K=search_range, return_sorted=False
            )[:2]
            tgt_to_pt_dist, tgt_to_pt_idx = tgt_to_pt_dist[0], tgt_to_pt_idx[0]
            tgt_to_pt_zero_mask = tgt_to_pt_dist == 0
            tgt_to_pt_mask = tgt_to_pt_dist == tgt_to_pt_dist.amin(1, keepdim=True)
            tgt_to_pt_mask.logical_and_(~tgt_to_pt_zero_mask.any(1).unsqueeze(1))
            expanded_tgt_rgb = tgt_rgb.unsqueeze(1).expand(-1, search_range, -1)
            masked_rec_tgt_to_pt_dist = tgt_to_pt_dist[tgt_to_pt_mask].sqrt().reciprocal_()
            recolored_pred_rgb.index_add_(
                0, tgt_to_pt_idx[tgt_to_pt_mask],
                (expanded_tgt_rgb[tgt_to_pt_mask]).mul(
                    masked_rec_tgt_to_pt_dist.unsqueeze(1))
            )
            recolored_pred_rgb_denominator = torch.zeros(
                (pred_xyz.shape[0],), dtype=cand_rgb.dtype, device=device).index_add_(
                0, tgt_to_pt_idx[tgt_to_pt_mask], masked_rec_tgt_to_pt_dist
            )
            tmp_mask = recolored_pred_rgb_denominator != 0
            recolored_pred_rgb[tmp_mask] = recolored_pred_rgb[tmp_mask].div(
                recolored_pred_rgb_denominator[tmp_mask].unsqueeze(1))
            recolored_pred_rgb[tgt_to_pt_idx[tgt_to_pt_zero_mask]] = \
                expanded_tgt_rgb[tgt_to_pt_zero_mask]
            tmp_mask.logical_not_()
            tmp_mask[tgt_to_pt_idx[tgt_to_pt_zero_mask]] = False
            return tmp_mask

        empty_rgb_mask = recolor_backward()

        @torch.no_grad()
        def recolor_forward():
            if torch.any(empty_rgb_mask):
                empty_rgb_xyz = pred_xyz[empty_rgb_mask]
                empty_rgb_to_tgt_dist, empty_rgb_to_tgt_idx = knn_points(
                    empty_rgb_xyz[None], tgt_xyz[None], K=search_range, return_sorted=False
                )[:2]
                empty_rgb_to_tgt_dist, empty_rgb_to_tgt_idx = empty_rgb_to_tgt_dist[0], empty_rgb_to_tgt_idx[0]
                empty_rgb_to_tgt_mask = empty_rgb_to_tgt_dist == empty_rgb_to_tgt_dist.amin(1, keepdim=True)
                recolored_pred_rgb[empty_rgb_mask] = torch.sum(
                    tgt_rgb[empty_rgb_to_tgt_idx] * empty_rgb_to_tgt_mask[:, :, None], dim=1
                ) / empty_rgb_to_tgt_mask.sum(1, keepdim=True)

        recolor_forward()

        rgb_loss = F.l1_loss(
            pred_rgb, recolored_pred_rgb, reduction='none'
        )
        return rgb_loss


class HyperDecoderUpsample(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 intra_channels: int,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 skip_encoding_fea: int,
                 out_channels2: int):
        super(HyperDecoderUpsample, self).__init__()
        basic_block = partial(BLOCKS_DICT[basic_block_type],
                              region_type=region_type,
                              bn=use_batch_norm, act=act)

        def make_block(up, out_ch):
            args = (2, 2) if up else (3, 1)
            seq_cls = NNSequentialWithConvTransBlockArgs if up else NNSequentialWithConvBlockArgs
            conv_cls = ConvTransBlock if up else ConvBlock
            return seq_cls(
                conv_cls(in_channels, intra_channels, *args,
                         region_type=region_type, bn=use_batch_norm, act=act),
                *(basic_block(intra_channels) for _ in range(basic_block_num)),
                ConvBlock(intra_channels, out_ch, 3, 1,
                          region_type=region_type, bn=use_batch_norm, act=act))

        self.non_shared_blocks = nn.ModuleList()
        for _ in range(non_shared_scales_num):
            for __ in range(residuals_num_per_scale - 1):
                self.non_shared_blocks.append(make_block(False, out_channels2 if _ <= skip_encoding_fea else out_channels))
            self.non_shared_blocks.append(make_block(True, out_channels2 if _ <= skip_encoding_fea else out_channels))

        self.shared_blocks = nn.ModuleList()
        for _ in range(residuals_num_per_scale - 1):
            self.shared_blocks.append(make_block(False, out_channels))
        self.shared_blocks.append(make_block(True, out_channels))

    def __getitem__(self, idx):
        if idx < len(self.non_shared_blocks):
            return self.non_shared_blocks[idx]
        else:
            return self.shared_blocks[(idx - len(self.non_shared_blocks)) % residuals_num_per_scale]


class HyperDecoderGenUpsample(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 intra_channels: int,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(HyperDecoderGenUpsample, self).__init__()
        basic_block = partial(BLOCKS_DICT[basic_block_type],
                              region_type=region_type,
                              bn=use_batch_norm, act=act)

        def make_block():
            return nn.Sequential(
                GenConvTransBlock(in_channels, intra_channels, 2, 2,
                                  region_type=region_type, bn=use_batch_norm, act=act),
                *(basic_block(intra_channels) for _ in range(basic_block_num)),
                ConvBlock(intra_channels, out_channels, 3, 1,
                          region_type=region_type, bn=use_batch_norm, act=None))

        self.non_shared_blocks = nn.ModuleList()
        for _ in range(non_shared_scales_num):
            self.non_shared_blocks.append(make_block())

        self.shared_blocks = make_block()

    def __getitem__(self, idx):
        tgt_idx = (idx % residuals_num_per_scale) - (residuals_num_per_scale - 1)
        if 0 == tgt_idx:
            if idx // residuals_num_per_scale < len(self.non_shared_blocks):
                return self.non_shared_blocks[idx // residuals_num_per_scale]
            else:
                return self.shared_blocks
        else:
            return None


class ResidualRecurrent(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 skip_encoding_fea: int):
        super(ResidualRecurrent, self).__init__()

        def make_block():
            return nn.Sequential(
                MLPBlock(
                    in_channels, in_channels,
                    bn='nn.bn1d' if use_batch_norm else None, act=act),
                MLPBlock(
                    in_channels, out_channels,
                    bn='nn.bn1d' if use_batch_norm else None, act=None)
            )

        self.non_shared_blocks = nn.ModuleList()
        for _ in range(non_shared_scales_num):
            if _ <= skip_encoding_fea:
                self.non_shared_blocks.append(nn.Module())
            else:
                for __ in range(residuals_num_per_scale):
                    self.non_shared_blocks.append(make_block())

        self.shared_blocks = nn.ModuleList()
        for __ in range(residuals_num_per_scale):
            self.shared_blocks.append(make_block())

    def __getitem__(self, idx):
        if idx < len(self.non_shared_blocks):
            return self.non_shared_blocks[idx]
        else:
            return self.shared_blocks[(idx - len(self.non_shared_blocks)) % residuals_num_per_scale]


class DecoderRecurrent(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 use_batch_norm: bool,
                 act: Optional[str],
                 skip_encoding_fea: int,
                 in_channels2: int):
        super(DecoderRecurrent, self).__init__()

        def make_block(in_ch):
            return (nn.Sequential(
                MLPBlock(
                    in_ch, in_ch,
                    bn='nn.bn1d' if use_batch_norm else None, act=act),
                MLPBlock(
                    in_ch, out_channels,
                    bn='nn.bn1d' if use_batch_norm else None, act=act)
            ))

        self.non_shared_blocks = nn.ModuleList()
        for _ in range(non_shared_scales_num):
            for __ in range(residuals_num_per_scale):
                if _ <= skip_encoding_fea:
                    self.non_shared_blocks.append(make_block(in_channels2))
                else:
                    self.non_shared_blocks.append(make_block(in_channels))

        self.shared_blocks = nn.ModuleList()
        for __ in range(residuals_num_per_scale):
            self.shared_blocks.append(make_block(in_channels))

    def __getitem__(self, idx):
        if idx < len(self.non_shared_blocks):
            return self.non_shared_blocks[idx]
        else:
            return self.shared_blocks[(idx - len(self.non_shared_blocks)) % residuals_num_per_scale]


class EncoderRecurrent(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 basic_block_type: str,
                 region_type: str,
                 basic_block_num: int,
                 use_batch_norm: bool,
                 act: Optional[str]):
        super(EncoderRecurrent, self).__init__()
        hidden_channels = in_channels
        basic_block = partial(BLOCKS_DICT[basic_block_type],
                              region_type=region_type,
                              bn=use_batch_norm, act=act)

        def make_block(down):
            args = (2, 2) if down else (3, 1)
            return nn.Sequential(
                ConvBlock(hidden_channels, hidden_channels, *args,
                          region_type=region_type, bn=use_batch_norm, act=act),
                *(basic_block(hidden_channels) for _ in range(basic_block_num)),
                ConvBlock(hidden_channels, hidden_channels, 3, 1,
                          region_type=region_type, bn=use_batch_norm, act=act)
            ), MEMLPBlock(
                hidden_channels, out_channels, bn=use_batch_norm, act=None
            )

        self.non_shared_blocks_out_first = MEMLPBlock(
            hidden_channels, out_channels, bn=use_batch_norm, act=None
        )
        self.non_shared_blocks = nn.ModuleList()
        self.non_shared_blocks_out = nn.ModuleList()
        for _ in range(non_shared_scales_num):
            for __ in range(residuals_num_per_scale):
                block, block_out = make_block(False if __ != residuals_num_per_scale - 1 else True)
                self.non_shared_blocks.append(block)
                self.non_shared_blocks_out.append(block_out)

        self.shared_blocks = nn.ModuleList()
        self.shared_blocks_out = nn.ModuleList()
        for _ in range(residuals_num_per_scale):
            block, block_out = make_block(False if _ != residuals_num_per_scale - 1 else True)
            self.shared_blocks.append(block)
            self.shared_blocks_out.append(block_out)

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

    def forward(self, x: ME.SparseTensor, batch_size: int):
        if not self.training: assert batch_size == 1
        strided_fea_list = [self.non_shared_blocks_out_first(x)]

        idx = 0
        while True:
            if idx < len(self.non_shared_blocks):
                block = self.non_shared_blocks[idx]
                block_out = self.non_shared_blocks_out[idx]
            else:
                tmp_idx = (idx - len(self.non_shared_blocks)) % len(self.shared_blocks)
                block = self.shared_blocks[tmp_idx]
                block_out = self.shared_blocks_out[tmp_idx]
            idx += 1
            x = block(x)
            strided_fea_list.append(block_out(x))
            if x.C.shape[0] == batch_size:
                break
        if idx < len(self.non_shared_blocks) + 1:
            print(f'Warning: EncoderRecurrent: '
                  f'Downsample steps ({idx}) < '
                  f'len(self.non_shared_blocks) + 1 ({len(self.non_shared_blocks) + 1})')

        return strided_fea_list
