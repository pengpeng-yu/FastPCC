from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from lib.sparse_conv_layers import \
    ConvBlock, ConvTransBlock, GenConvTransBlock, MEMLPBlock, \
    NNSequentialWithConvTransBlockArgs, NNSequentialWithConvBlockArgs


class BoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return torch.clip(x, -bound, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        grad_output = grad_output.clone()
        grad_output[x > bound] = 1
        grad_output[x < -bound] = -1
        return grad_output, None


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 intra_channels: Tuple[int, ...],
                 requires_points_num_list: bool,
                 points_num_scaler: float,
                 region_type: str,
                 act: Optional[str]):
        super(Encoder, self).__init__()
        self.requires_points_num_list = requires_points_num_list
        self.points_num_scaler = points_num_scaler
        self.blocks = nn.ModuleList((
            ConvBlock(
                in_channels, intra_channels[0], 3, 1,
                region_type=region_type, act=act
            ),))
        last_ch = intra_channels[0]
        for ch in intra_channels[1:]:
            self.blocks.append(nn.Sequential(
                ConvBlock(
                    last_ch, ch, 2, 2,
                    region_type='HYPER_CUBE', act=act
                ),
                ConvBlock(
                    ch, ch, 3, 1,
                    region_type=region_type, act=act
                )
            ))
            last_ch = ch

    def forward(self, x) -> Tuple[List[ME.SparseTensor], Optional[List[List[int]]]]:
        points_num_list = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx != len(self.blocks) - 1:
                points_num_list.append([_.shape[0] for _ in x.decomposed_coordinates])

        scaler = self.points_num_scaler
        if not self.requires_points_num_list:
            points_num_list = None
        else:
            points_num_list = [[int(n * scaler) for n in _]
                               for _ in points_num_list]

        return x, points_num_list


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 intra_channels: Tuple[int, ...],
                 region_type: str,
                 act: Optional[str]):
        super(Decoder, self).__init__()
        self.upsample_blocks = nn.ModuleList()
        self.classify_blocks = nn.ModuleList()
        last_ch = in_channels
        for idx, ch in enumerate(intra_channels):
            blocks = nn.Sequential()
            if idx == len(intra_channels) - 1:
                blocks.append(ConvBlock(
                    last_ch, last_ch, 3, 1,
                    region_type=region_type, act=act
                ))
            blocks.append(GenConvTransBlock(
                last_ch, ch, 2, 2,
                region_type='HYPER_CUBE', act=act
            ))
            if idx != len(intra_channels) - 1:
                blocks.append(ConvBlock(
                    ch, ch, 3, 1,
                    region_type=region_type, act=act
                ))
            self.upsample_blocks.append(blocks)
            self.classify_blocks.append(nn.Sequential(
                ConvBlock(
                    ch, ch // 2, 1, 1, act=act
                ),
                ConvBlock(
                    ch // 2, 1, 1, 1, act=None
                )))
            last_ch = ch
        self.pruning = ME.MinkowskiPruning()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_forward(*args, **kwargs)
        else:
            return self.test_forward(*args, **kwargs)

    def train_forward(self, fea, points_num_list, target_key):
        loss_dict = {}
        for idx, (upsample_block, classify_block) in \
                enumerate(zip(self.upsample_blocks, self.classify_blocks)):
            fea = upsample_block(fea)
            pred = classify_block(fea)
            keep = self.get_keep(pred, points_num_list, [2 ** len(self.classify_blocks)] * 3)
            keep_target = self.get_target(pred, target_key)
            keep |= keep_target
            loss_dict[f'coord_{idx}_recon_loss'] = self.get_coord_recon_loss(pred, keep_target)
            fea = self.pruning(fea, keep)
        return loss_dict

    def test_forward(self, fea, points_num_list):
        for idx, (upsample_block, classify_block) in \
                enumerate(zip(self.upsample_blocks, self.classify_blocks)):
            fea = upsample_block(fea)
            keep = self.get_keep(classify_block(fea), points_num_list,
                                 [2 ** len(self.classify_blocks)] * 3)
            fea = self.pruning(fea, keep)
        return fea

    @torch.no_grad()
    def get_keep(self, pred: ME.SparseTensor, points_num_list: List[List[int]],
                 max_stride_lossy_recon: List[int]) -> torch.Tensor:
        _cm = pred.coordinate_manager._manager
        max_stride_coord_key = ME.CoordinateMapKey(
            max_stride_lossy_recon, '' if self.training or
                len(_cm.get_coordinate_map_keys(max_stride_lossy_recon)) == 1 else 'pruned')
        stride_scaler = [_ // __ for _, __ in zip(max_stride_coord_key.get_tensor_stride(), pred.tensor_stride)]
        pool = ME.MinkowskiMaxPooling(stride_scaler, stride_scaler, dimension=3).to(pred.device)
        un_pool = ME.MinkowskiPoolingTranspose(stride_scaler, stride_scaler, dimension=3).to(pred.device)
        pred_local_max = un_pool(pool(pred, max_stride_coord_key), pred.coordinate_map_key)
        local_max_mask = (pred.F - pred_local_max.F).squeeze(1) != 0

        if points_num_list is not None:
            target_points_num = points_num_list.pop()
            sample_threshold = []
            for sample_tgt, sample_permutation in zip(target_points_num, pred.decomposition_permutations):
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
            target.type(pred.F.dtype),
            reduction='sum'
        )
        return recon_loss


class HyperDecoderUpsample(nn.Module):
    def __init__(self, in_channels: Tuple[int, ...],
                 out_channels: Tuple[int, ...],
                 if_sample: Tuple[int, ...],
                 region_type: str,
                 act: Optional[str]):
        super(HyperDecoderUpsample, self).__init__()

        def make_block(up, in_ch, out_ch):
            args = (2, 2) if up else (3, 1)
            seq_cls = NNSequentialWithConvTransBlockArgs if up else NNSequentialWithConvBlockArgs
            conv_cls = ConvTransBlock if up else ConvBlock
            return seq_cls(
                conv_cls(max(in_ch, 1), out_ch, *args,
                         region_type=region_type, act=act),
                ConvBlock(out_ch, out_ch, 3, 1,
                          region_type=region_type, act=act))

        self.blocks = nn.ModuleList()
        for in_ch, out_ch, up in zip(in_channels, out_channels, if_sample):
            self.blocks.append(make_block(up, in_ch, out_ch))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


class HyperDecoderGenUpsample(nn.Module):
    def __init__(self, in_channels: Tuple[int, ...],
                 if_sample: Tuple[int, ...],
                 region_type: str,
                 act: Optional[str]):
        super(HyperDecoderGenUpsample, self).__init__()

        def make_block(in_ch):
            return nn.Sequential(
                GenConvTransBlock(in_ch, max(in_ch // 4, 1), 2, 2,
                                  region_type=region_type, act=act),
                ConvBlock(max(in_ch // 4, 1), 1, 3, 1,
                          region_type=region_type, act=None))

        self.blocks = nn.ModuleList()
        for in_ch, up in zip(in_channels, if_sample):
            self.blocks.append(make_block(in_ch) if up else None)

    def __getitem__(self, idx):
        return self.blocks[idx]


class SubResidualGeoLossl(nn.Module):
    def __init__(self, in_ch, out_ch, region_type, act,
                 bottleneck_value_bound: int):
        super(SubResidualGeoLossl, self).__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_ch + in_ch, in_ch, 3, 1, region_type=region_type, act=act),
            ConvBlock(in_ch, out_ch, 3, 1, region_type=region_type, act=None)
        )
        self.register_buffer('bound', torch.tensor(bottleneck_value_bound),
                             persistent=False)

    def forward(self, x, y):
        x = ME.cat((x, y))
        x = self.blocks(x)
        x = ME.SparseTensor(
            BoundFunction.apply(x.F, self.bound),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )
        return x


class ResidualGeoLossl(nn.Module):
    def __init__(self, in_channels: Tuple[int, ...],
                 out_channels: Tuple[int, ...],
                 region_type: str,
                 act: Optional[str],
                 bottleneck_value_bound: int,
                 skip_encoding_fea: int):
        super(ResidualGeoLossl, self).__init__()
        self.blocks = nn.ModuleList()
        for idx, (in_ch, out_ch) in enumerate(zip(in_channels, out_channels)):
            self.blocks.append(SubResidualGeoLossl(in_ch, out_ch, region_type, act, bottleneck_value_bound)
                               if idx > skip_encoding_fea else None)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


class SubDecoderGeoLossl(nn.Module):
    def __init__(self, in_ch, in_ch2, out_ch, region_type, act):
        super(SubDecoderGeoLossl, self).__init__()
        self.residual_decoder = nn.Sequential(
            MEMLPBlock(in_ch, out_ch // 2, act=act),
            MEMLPBlock(out_ch // 2, out_ch, act=act)
        )
        self.decoder = nn.Sequential(
            MEMLPBlock(out_ch + in_ch2, out_ch, act=act),
            MEMLPBlock(out_ch, out_ch, act=act)
        )

    def forward(self, x, y):
        assert isinstance(y, ME.SparseTensor)
        if isinstance(x, torch.Tensor):
            x = ME.SparseTensor(
                x, coordinate_map_key=y.coordinate_map_key,
                coordinate_manager=y.coordinate_manager
            )
        x = self.residual_decoder(x)
        x = self.decoder(ME.cat((x, y)))
        return x


class SubDecoderGeoLossl2(nn.Module):
    def __init__(self, in_ch, in_ch2, out_ch, region_type, act):
        super(SubDecoderGeoLossl2, self).__init__()
        self.decoder = nn.Sequential(
            MEMLPBlock(
                in_ch2, out_ch, act=act
            ),
            MEMLPBlock(
                out_ch, out_ch, act=act
            ))

    def forward(self, x):
        x = self.decoder(x)
        return x


class DecoderGeoLossl(nn.Module):
    def __init__(self, in_channels: Tuple[int, ...],
                 in_channels2: Tuple[int, ...],
                 out_channels: Tuple[int, ...],
                 region_type: str,
                 act: Optional[str],
                 skip_encoding_fea: int):
        super(DecoderGeoLossl, self).__init__()
        self.blocks = nn.ModuleList()
        for idx, (in_ch, out_ch, in_ch2) in enumerate(zip(in_channels, out_channels, in_channels2)):
            self.blocks.append(SubDecoderGeoLossl(
                in_ch, in_ch2, out_ch, region_type, act)
                if idx > skip_encoding_fea else SubDecoderGeoLossl2(
                in_ch, in_ch2, out_ch, region_type, act)
                )

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


class EncoderGeoLossl(nn.Module):
    def __init__(self,
                 in_channels: Tuple[int, ...],
                 out_channels: Tuple[int, ...],
                 if_sample: Tuple[int, ...],
                 region_type: str,
                 act: Optional[str],
                 bottleneck_value_bound: int,
                 skip_encoding_fea: int):
        super(EncoderGeoLossl, self).__init__()
        assert len(in_channels) + 1 == len(out_channels)

        def make_block(down, in_ch, out_ch):
            intra_out_ch = max(in_ch, out_ch)
            args = (2, 2) if down else (3, 1)
            return nn.Sequential(
                ConvBlock(in_ch, in_ch, *args,
                          region_type=region_type, act=act),
                ConvBlock(in_ch, intra_out_ch, 3, 1,
                          region_type=region_type, act=act)
            ), MEMLPBlock(
                intra_out_ch, out_ch, act=act
            )

        self.blocks_out_first = MEMLPBlock(
            in_channels[0], out_channels[0], act=act
        ) if skip_encoding_fea < 0 else None
        self.blocks = nn.ModuleList()
        self.blocks_out = nn.ModuleList()
        for idx, (in_ch, out_ch, down) in enumerate(zip(in_channels, out_channels[1:], if_sample)):
            block, block_out = make_block(
                down, in_ch, out_ch
            )
            self.blocks.append(block)
            self.blocks_out.append(block_out if idx >= skip_encoding_fea else None)

        self.out_channels = out_channels
        assert len(self.blocks) == len(self.blocks_out)
        self.register_buffer('bound', torch.tensor(bottleneck_value_bound),
                             persistent=False)

    def __len__(self):
        return len(self.blocks)

    def forward(self, x: ME.SparseTensor, batch_size: int):
        if not self.training: assert batch_size == 1
        strided_fea_list = [self.blocks_out_first(x) if self.blocks_out_first is not None else x]

        for block, block_out in zip(self.blocks, self.blocks_out):
            x = block(x)
            strided_fea_list.append(block_out(x) if block_out is not None else x)

        if strided_fea_list[-1] is not None:
            strided_fea_list[-1] = ME.SparseTensor(
                BoundFunction.apply(strided_fea_list[-1].F, self.bound),
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager
            )
        return strided_fea_list
