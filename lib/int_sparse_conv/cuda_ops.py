import math
from typing import Tuple, List, Optional, Dict, Any
import torch
import torch.nn as nn
import torchsparse.nn
from torchsparse import SparseTensor
try:
    from .build import int_sparse_conv_ext
except ImportError as e:
    int_sparse_conv_ext = None

from torch.ao.quantization import HistogramObserver


SharedFxpShift = 23  # Q8.23
WeightRange = (1 << 7) - 1  # [-WeightRange, WeightRange]
ActRange = (1 << 7) - 1  # [-ActRange, ActRange]


class SparseTensorHistogramObserver(HistogramObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: SparseTensor) -> SparseTensor:
        super().forward(input.F)
        return input

    def calculate_qparams(self, *args, **kwargs):
        return super().calculate_qparams(*args, **kwargs)

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}, {self.qscheme}"


def make_obs(qscheme=torch.per_tensor_symmetric):
    return SparseTensorHistogramObserver(
        bins=2048, dtype=torch.qint8, quant_min=-ActRange, quant_max=ActRange, qscheme=qscheme)


class SparseResBlockWithObs(nn.Module):  # float32
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.obs = make_obs(torch.per_tensor_symmetric)
        self.conv = torchsparse.nn.Conv3d(ch, ch, 3, 1, 1, bias=True)
        self.act = nn.PReLU()
        self.obs2 = make_obs(torch.per_tensor_symmetric)
        self.conv2 = torchsparse.nn.Conv3d(ch, ch, 3, 1, 1, bias=True)
        self.act2 = nn.PReLU()

    def forward(self, org: SparseTensor):
        org = self.obs(org)
        x = self.conv(org)
        x.F = self.act(x.F)
        x = self.obs2(x)
        x = self.conv2(x)
        x.F.add_(org.F)
        x.F = self.act2(x.F)
        return x


class SparseResBlockIn32W8Out32(nn.Module):
    def __init__(self, ch: int, eps=None):
        super().__init__()
        self.ch = ch
        self.eps = eps if eps is not None else torch.finfo(torch.float32).eps
        self.input_requant = RequantFxpToScaledInt8()
        self.conv_prelu = SparseConvPReLUIn8W8Out8(ch, ch, (3, 3, 3), (1, 1, 1))
        self.conv2 = SparseConvIn8W8Out32(ch, ch, (3, 3, 3), (1, 1, 1))
        self.prelu = PReLUIn32Out32()

    @torch.no_grad()
    def import_parameters(self, block: SparseResBlockWithObs):
        scale, zero_point = block.obs.calculate_qparams()
        scale2, zero_point2 = block.obs2.calculate_qparams()
        self.input_requant.import_parameters(scale, zero_point)
        self.conv_prelu.import_parameters(
            scale, zero_point, scale2, zero_point2, block.conv, block.act)  # scaled int -> scaled int
        self.conv2.import_parameters(scale2, zero_point2, block.conv2)  # scaled int -> fixed-point
        self.prelu.import_parameters(block.act2)  # fixed-point -> fixed-point

    def forward(
        self, input: SparseTensor,  # N1 x C1 fixed-point feats, N1 x 4(batch_idx, x, y, z) int32 coords
    ) -> SparseTensor:  # fixed-point feats
        input_feats_scaled_int = self.input_requant(input.F)
        x = SparseTensor(input_feats_scaled_int, input.C, input.stride, input.spatial_range)
        x._caches = input._caches
        x = self.conv2(self.conv_prelu(x))
        assert input.F.dtype == x.F.dtype == torch.int32
        x = SparseTensor(self.prelu(input.F + x.F), input.C, input.stride, input.spatial_range)
        x._caches = input._caches
        return x


def sparse_conv_in8w8out32(
    in_feats: torch.Tensor,             # N1 x C1, scaled int
    weight: torch.Tensor,               # kernel_volume x C2 x C1
    in_coords: torch.Tensor,            # N1 x 4(batch_idx, x, y, z)
    out_coords: torch.Tensor,           # N2 x 4
    kernel_size: Tuple[int, int, int],  # 3
    stride: Tuple[int, int, int],       # 3
    in_out_maps: Optional[List[torch.Tensor]] = None,
    hashmap_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    zero_point_comp: Optional[torch.Tensor] = None,
    if_in_coords_equals_out_coords: bool = False,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
    # output: N2 x C2 scaled int32, hashmap_kv, in_out_maps
    device = in_feats.device
    kernel_volume = math.prod(kernel_size)
    if if_in_coords_equals_out_coords is True and all(ks % 2 == 1 for ks in kernel_size):
        idx_omit_map = kernel_volume >> 1
    else:
        idx_omit_map = -1

    if in_out_maps is None:
        if hashmap_kv is None:
            hashmap_keys = torch.zeros(2 * in_coords.shape[0], dtype=torch.int64, device=device)
            hashmap_vals = torch.zeros(2 * in_coords.shape[0], dtype=torch.int32, device=device)
            hashmap = int_sparse_conv_ext.GPUHashTable(hashmap_keys, hashmap_vals)
            hashmap.insert_coords(in_coords[:, [1, 2, 3, 0]].contiguous())
            hashmap_kv = hashmap_keys, hashmap_vals
        else:
            hashmap = int_sparse_conv_ext.GPUHashTable(hashmap_kv[0], hashmap_kv[1])
        out_in_map = hashmap.lookup_coords(
            out_coords[:, [1, 2, 3, 0]].contiguous(),
            torch.tensor(kernel_size, device=device, dtype=torch.int32),
            torch.tensor(stride, device=device, dtype=torch.int32),
            kernel_volume
        )[:out_coords.size(0)].T.contiguous()  # kernel_volume x number_of_output_points
        del hashmap

        in_out_maps = []
        valid_mask = out_in_map.bool()
        out_map = torch.nonzero(valid_mask, as_tuple=True)
        valid_num_cum = valid_mask.sum(1).cumsum(0).tolist()
        del valid_mask
        in_map = out_in_map[out_map] - 1
        del out_in_map
        out_map = out_map[1].to(torch.int32)
        for idx, cur_pos in enumerate(valid_num_cum):
            if idx == idx_omit_map:
                sub_in_map = sub_out_map = None
            else:
                last_pos = valid_num_cum[idx - 1] if idx > 0 else 0
                sub_valid_num = cur_pos - last_pos
                if sub_valid_num == 0:
                    sub_in_map = sub_out_map = None
                else:
                    sub_in_map = in_map[last_pos: cur_pos]
                    sub_out_map = out_map[last_pos: cur_pos]
            in_out_maps.append((sub_in_map, sub_out_map))

    out_feats = torch.zeros((out_coords.shape[0], weight.size(1)), dtype=torch.int32, device=device)
    if idx_omit_map != -1:
        int_sparse_conv_ext.cutlass_gemm_int8(
            in_feats, weight[idx_omit_map],
            out_feats if zero_point_comp is None else out_feats + zero_point_comp[idx_omit_map],
            out_feats)
    for idx, (sub_in_map, sub_out_map) in enumerate(in_out_maps):
        if sub_in_map is not None:
            if zero_point_comp is not None:
                out_feats[sub_out_map] += zero_point_comp[idx]
            int_sparse_conv_ext.cutlass_gather_gemm_scatter_int8(
                in_feats, weight[idx],
                out_feats,
                out_feats, sub_in_map, sub_out_map)

    ret = out_feats, hashmap_kv, in_out_maps
    return ret


class LoadSaveUint32RequantMul(nn.Module):
    def __init__(self):
        super().__init__()

    # torch.save does not support uint32
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        key = prefix + 'requant_mul'
        destination[key] = destination[key].to(torch.int64)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'requant_mul'
        state_dict[key] = state_dict[key].to(torch.uint32)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class SparseConvIn8Out8(LoadSaveUint32RequantMul):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: Tuple[int, int, int], stride: Tuple[int, int, int],
                 with_prelu: bool, out_scaled_int: bool, eps=None, requant_mul_guard_bits=None):
        super().__init__()
        kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
        self.register_buffer('weight', torch.zeros((kernel_volume, out_ch, in_ch), dtype=torch.int8), persistent=True)
        self.register_buffer('bias', torch.zeros((out_ch,), dtype=torch.int32), persistent=True)
        if with_prelu:
            self.register_buffer('slope', torch.zeros((1,), dtype=torch.int32), persistent=True)
        self.register_buffer('requant_mul', torch.zeros((out_ch,), dtype=torch.uint32), persistent=True)
        self.register_buffer('requant_shift', torch.zeros((1,), dtype=torch.int32), persistent=True)
        self.register_buffer('int_zero_point_out', torch.zeros((1,), dtype=torch.int64), persistent=True)

        self.register_buffer('scale_in', torch.zeros((1,), dtype=torch.float32) - 1, persistent=True)
        self.register_buffer('zero_point_in', torch.zeros((1,), dtype=torch.float32), persistent=True)
        self.register_buffer('scale_weight', torch.zeros((out_ch,), dtype=torch.float32) - 1, persistent=True)
        self.register_buffer('scale_out', torch.zeros((1,), dtype=torch.float32) - 1, persistent=True)
        self.register_buffer('zero_point_out', torch.zeros((1,), dtype=torch.float32), persistent=True)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_volume = kernel_volume
        self.kernel_size = kernel_size
        self.stride = stride
        self.with_prelu = with_prelu
        self.use_zero_point_in = False

        # True: Input/output int with per-tensor scales.
        # False: Input int with per-tensor scales, output fixed-point. scale_out is not needed.
        self.out_scaled_int = out_scaled_int
        self.eps = eps if eps is not None else torch.finfo(torch.float32).eps
        self.requant_mul_guard_bits = requant_mul_guard_bits \
            if requant_mul_guard_bits is not None else 10  # For processing bias, slope, zero point, kernel accumulation

    @torch.no_grad()
    def import_parameters(
            self, scale_in: torch.Tensor, zero_point_in: torch.Tensor,
            scale_out: Optional[torch.Tensor], zero_point_out: Optional[torch.Tensor],
            conv: torchsparse.nn.Conv3d, prelu: Optional[nn.PReLU]):
        assert scale_in.dtype == conv.kernel.dtype == torch.float32
        assert scale_in.numel() == 1
        assert zero_point_in.dtype in (torch.int32, torch.int64)
        assert zero_point_in.numel() == 1
        assert self.weight.size(0) == conv.kernel.size(0)
        assert conv.bias is not None
        assert all(_ == __ for _, __ in zip(self.kernel_size, conv.kernel_size))
        assert all(_ == __ for _, __ in zip(self.stride, conv.stride))
        if scale_in <= self.eps:
            print(f'Warning: {scale_in}')
            scale_in = scale_in.clip(min=self.eps)
        if self.out_scaled_int is True:
            assert scale_out.dtype == scale_in.dtype
            assert scale_out.numel() == 1
            if scale_out <= self.eps:
                print(f'Warning: {scale_out}')
                scale_out = scale_out.clip(min=self.eps)
            self.scale_out[:] = scale_out
        else:
            assert scale_out is None
        self.scale_in[:] = scale_in

        per_out_abs_max = conv.kernel.abs().amax(dim=(0, 1))  # kernel_volume * in_ch * out_ch
        scale_weight = per_out_abs_max / WeightRange
        if (scale_weight <= self.eps).any():
            print(f'Warning: {scale_weight}')
            scale_weight = scale_weight.clip(min=self.eps)
        self.scale_weight[:] = scale_weight
        permuted_conv_kernel = conv.kernel.permute((0, 2, 1))  # permute to kernel_volume * out_ch * in_ch
        self.weight[...] = (
            permuted_conv_kernel / scale_weight[None, :, None]
        ).round().clip(-WeightRange, WeightRange).to(self.weight.dtype)

        if zero_point_in != 0:
            self.use_zero_point_in = True
            self.zero_point_in[:] = zero_point_in
            self.register_buffer(
                'int_zero_point_in_comp',
                torch.zeros((self.kernel_volume, self.out_ch,), dtype=torch.int32), persistent=True)
            self.int_zero_point_in_comp[...] = -(
                zero_point_in.to(torch.float) * self.weight.to(torch.float)
            ).sum(2).round().to(torch.int32)

        scale_bias = scale_in * scale_weight
        self.bias[:] = (conv.bias / scale_bias).round().to(torch.int32)

        if self.with_prelu:
            assert prelu.weight.numel() == 1
            if prelu.weight.abs().item() > 63:
                print(f'Warning: abnormal prelu slope {prelu.weight.item()}.')
            self.slope[:] = (prelu.weight * (1 << 25)).round().to(torch.int32)  # Q6.25
        else:
            assert prelu is None

        if self.out_scaled_int is True:
            assert scale_out is not None
            requant_mul = scale_in * scale_weight / scale_out
            self.requant_shift[:] = torch.log2((1 << (32 - self.requant_mul_guard_bits)) / requant_mul).min().floor()
            assert self.requant_shift >= 0, self.requant_shift
            self.requant_mul[:] = (requant_mul * 2 ** self.requant_shift.to(torch.float)).round().to(torch.uint32)
            assert zero_point_out.dtype in (torch.int32, torch.int64)
            assert zero_point_out.numel() == 1
            self.zero_point_out[:] = zero_point_out
            self.int_zero_point_out[:] = zero_point_out.to(torch.int64) << self.requant_shift

        else:
            assert scale_out is None
            requant_mul = scale_in * scale_weight
            self.requant_shift[:] = torch.log2((1 << (32 - self.requant_mul_guard_bits)) / requant_mul).min().floor()
            assert self.requant_shift >= 0, self.requant_shift
            self.requant_mul[:] = (requant_mul * 2 ** self.requant_shift.to(torch.float)).round().to(torch.uint32)
            assert zero_point_out is None
            self.zero_point_out[:] = 0
            self.int_zero_point_out[:] = 0

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'int_zero_point_in_comp'
        if key in state_dict:
            self.use_zero_point_in = True
            self.register_buffer(
                'int_zero_point_in_comp',
                torch.zeros((self.kernel_volume, self.out_ch,), dtype=torch.int32), persistent=True)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def unique(self, *args, **kwargs):
        return torch.unique(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if isinstance(args[0], SparseTensor) or \
                ('input' in kwargs and isinstance(kwargs['input'], SparseTensor)):
            return self.forward_with_sparse_tensor(*args, **kwargs)
        else:
            return self.forward_with_coords(*args, **kwargs)

    def forward_with_sparse_tensor(self, input: SparseTensor) -> SparseTensor:
        caches = input._caches
        cur_kmap_tag = (input.stride, self.kernel_size, self.stride)
        cur_kmap: Dict[str, Any] = caches.kmaps.get(cur_kmap_tag)
        in_out_maps = cur_kmap.get('in_out_maps') if cur_kmap is not None else None
        hashmap_kv = caches.hashmaps.get(input.stride)

        if self.stride == (1, 1, 1):
            output_stride = input.stride
            output_coords = input.C
            if_in_coords_equals_out_coords = True

        else:
            if_in_coords_equals_out_coords = False
            output_stride = tuple(_a * _b for _a, _b in zip(input.stride, self.stride))
            if output_stride in caches.cmaps:
                output_coords = caches.cmaps[output_stride][0]

            elif (self.stride[0] & (self.stride[0] - 1)) == 0 and all(self.stride[0] == __ for __ in self.stride[1:]):
                output_coords = input.C.clone()
                output_coords[:, 1:] >>= (self.stride[0].bit_length() - 1)
                output_coords = self.unique(output_coords, dim=0)

            else:
                raise NotImplementedError((input.stride, self.stride))

        conv_out_feats, hashmap_kv, in_out_maps = self.forward_with_coords(
            input.F, input.C, output_coords, in_out_maps, hashmap_kv, if_in_coords_equals_out_coords)

        if cur_kmap_tag not in caches.kmaps:
            caches.kmaps[cur_kmap_tag] = {}
        if 'in_out_maps' not in caches.kmaps[cur_kmap_tag]:
            caches.kmaps[cur_kmap_tag]['in_out_maps'] = in_out_maps
        if input.stride not in caches.hashmaps:
            caches.hashmaps[input.stride] = hashmap_kv
        if input.stride not in caches.cmaps:
            caches.cmaps[input.stride] = input.C, input.spatial_range
        if output_stride not in caches.cmaps:
            caches.cmaps[output_stride] = output_coords, None

        ret = SparseTensor(conv_out_feats, output_coords, output_stride, None)
        ret._caches = caches
        return ret

    def forward_with_coords(
        self,
        in_feats: torch.Tensor,    # N1 x C1, scaled int
        in_coords: torch.Tensor,   # N1 x 4(batch_idx, x, y, z), int32
        out_coords: torch.Tensor,  # N2 x 4, int32
        in_out_maps: Optional[List[torch.Tensor]] = None,
        hashmap_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        if_in_coords_equals_out_coords: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
        # output: N2 x C2 scaled int, hashmap_kv, in_out_maps

        conv_out_feats, hashmap_kv, in_out_maps = sparse_conv_in8w8out32(
            in_feats, self.weight, in_coords, out_coords,
            self.kernel_size, self.stride, in_out_maps, hashmap_kv,
            self.int_zero_point_in_comp if self.use_zero_point_in else None, if_in_coords_equals_out_coords)

        if self.with_prelu is False:
            if self.out_scaled_int is True:
                out_feats = int_sparse_conv_ext.bias_requant_to_int8(  # scaled int
                    conv_out_feats, self.bias, self.requant_mul, self.int_zero_point_out,
                    self.requant_shift.item())

            else:
                out_feats = int_sparse_conv_ext.bias_requant_to_int32(  # shared fixed-point
                    conv_out_feats, self.bias, self.requant_mul, self.int_zero_point_out,
                    self.requant_shift.item() - SharedFxpShift)

        else:
            if self.out_scaled_int is True:
                out_feats = int_sparse_conv_ext.bias_prelu_requant_to_int8(  # scaled int
                    conv_out_feats, self.bias, self.slope, self.requant_mul, self.int_zero_point_out,
                    self.requant_shift.item())

            else:
                out_feats = int_sparse_conv_ext.bias_prelu_requant_to_int32(  # shared fixed-point
                    conv_out_feats, self.bias, self.slope, self.requant_mul, self.int_zero_point_out,
                    self.requant_shift.item() - SharedFxpShift)

        return out_feats, hashmap_kv, in_out_maps


class SparseConvIn8W8Out8(SparseConvIn8Out8):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1), *args, **kwargs):
        super().__init__(in_ch, out_ch, kernel_size, stride, False, True, *args, **kwargs)

    @torch.no_grad()
    def import_parameters(
            self, scale_in: torch.Tensor, zero_point_in: torch.Tensor,
            scale_out: torch.Tensor, zero_point_out: torch.Tensor,
            conv: torchsparse.nn.Conv3d):
        super().import_parameters(scale_in, zero_point_in, scale_out, zero_point_out, conv, None)


class SparseConvIn8W8Out32(SparseConvIn8Out8):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1), *args, **kwargs):
        super().__init__(in_ch, out_ch, kernel_size, stride, False, False, *args, **kwargs)

    @torch.no_grad()
    def import_parameters(
            self, scale_in: torch.Tensor, zero_point_in: torch.Tensor,
            conv: torchsparse.nn.Conv3d):
        super().import_parameters(scale_in, zero_point_in, None, None, conv, None)


class SparseConvPReLUIn8W8Out8(SparseConvIn8Out8):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1), *args, **kwargs):
        super().__init__(in_ch, out_ch, kernel_size, stride, True, True, *args, **kwargs)

    @torch.no_grad()
    def import_parameters(
            self, scale_in: torch.Tensor, zero_point_in: torch.Tensor,
            scale_out: torch.Tensor, zero_point_out: torch.Tensor,
            conv: torchsparse.nn.Conv3d, prelu: nn.PReLU):
        super().import_parameters(scale_in, zero_point_in, scale_out, zero_point_out, conv, prelu)


class SparseConvPReLUIn8W8Out32(SparseConvIn8Out8):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1), *args, **kwargs):
        super().__init__(in_ch, out_ch, kernel_size, stride, True, False, *args, **kwargs)

    @torch.no_grad()
    def import_parameters(
            self, scale_in: torch.Tensor, zero_point_in: torch.Tensor,
            conv: torchsparse.nn.Conv3d, prelu: nn.PReLU):
        super().import_parameters(scale_in, zero_point_in, None, None, conv, prelu)


class PReLUIn32Out32(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('slope', torch.zeros((1,), dtype=torch.int32), persistent=True)

    @torch.no_grad()
    def import_parameters(self, prelu: nn.PReLU):
        assert prelu.weight.dtype == torch.float32
        assert prelu.weight.numel() == 1
        self.slope[:] = (prelu.weight * (1 << 25)).round().to(torch.int32)  # Q6.25

    def forward(self, input: torch.Tensor):
        return int_sparse_conv_ext.prelu(input, self.slope)  # (fixed-point x Qx.xx) shift xx -> fixed-point


class RequantFxpToScaledInt8(LoadSaveUint32RequantMul):
    def __init__(self, eps=None, requant_mul_guard_bits=None):
        super().__init__()
        self.register_buffer('requant_mul', torch.zeros((1,), dtype=torch.uint32), persistent=True)
        self.register_buffer('requant_shift', torch.zeros((1,), dtype=torch.int32), persistent=True)
        self.register_buffer('int_zero_point_out', torch.zeros((1,), dtype=torch.int64), persistent=True)

        self.register_buffer('scale_out', torch.zeros((1,), dtype=torch.float32) - 1, persistent=True)
        self.register_buffer('zero_point_out', torch.zeros((1,), dtype=torch.float32), persistent=True)

        self.eps = eps if eps is not None else torch.finfo(torch.float32).eps
        self.requant_mul_guard_bits = requant_mul_guard_bits \
            if requant_mul_guard_bits is not None else 2  # For adding zero point

    @torch.no_grad()
    def import_parameters(self, scale_out: torch.Tensor, zero_point_out: torch.Tensor):
        assert scale_out.dtype == torch.float32
        assert scale_out.numel() == 1
        if scale_out <= self.eps:
            print(f'Warning: {scale_out}')
            scale_out = scale_out.clip(min=self.eps)
        self.scale_out[:] = scale_out
        assert zero_point_out.dtype in (torch.int32, torch.int64)
        assert zero_point_out.numel() == 1
        self.requant_shift[:] = torch.log2((1 << (32 - self.requant_mul_guard_bits)) * scale_out).floor()
        assert self.requant_shift >= 0, self.requant_shift
        self.requant_mul[:] = ((2 ** self.requant_shift.to(torch.float)) / scale_out).round().to(torch.uint32)
        self.zero_point_out[:] = zero_point_out
        self.int_zero_point_out[:] = (
            zero_point_out * (2 ** (SharedFxpShift + self.requant_shift.to(torch.float)))
        ).round().to(torch.int64)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # (shared fixed-point x Qx.xx) shift (SharedFxpShift + xx) -> scaled int
        return int_sparse_conv_ext.requant_to_int8(
            input, self.requant_mul.repeat(input.size(1)), self.int_zero_point_out,
            SharedFxpShift + self.requant_shift.item())


class LinearIn8W8(LoadSaveUint32RequantMul):
    def __init__(self, in_ch: int, out_ch: int, with_prelu: bool, out_scaled_int: bool,
                 eps=None, requant_mul_guard_bits=None):
        super().__init__()
        self.register_buffer('weight', torch.zeros((out_ch, in_ch), dtype=torch.int8), persistent=True)
        self.register_buffer('bias', torch.zeros((out_ch,), dtype=torch.int32), persistent=True)
        if with_prelu:
            self.register_buffer('slope', torch.zeros((1,), dtype=torch.int32), persistent=True)
        self.register_buffer('requant_mul', torch.zeros((out_ch,), dtype=torch.uint32), persistent=True)
        self.register_buffer('requant_shift', torch.zeros((1,), dtype=torch.int32), persistent=True)
        self.register_buffer('int_zero_point_out', torch.zeros((1,), dtype=torch.int64), persistent=True)

        self.register_buffer('scale_in', torch.zeros((1,), dtype=torch.float32) - 1, persistent=True)
        self.register_buffer('zero_point_in', torch.zeros((1,), dtype=torch.float32), persistent=True)
        self.register_buffer('scale_weight', torch.zeros((out_ch,), dtype=torch.float32) - 1, persistent=True)
        self.register_buffer('scale_out', torch.zeros((1,), dtype=torch.float32) - 1, persistent=True)
        self.register_buffer('zero_point_out', torch.zeros((1,), dtype=torch.float32), persistent=True)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.with_prelu = with_prelu

        # True: Input/output int with per-tensor scales.
        # False: Input int with per-tensor scales, output fixed-point. scale_out is not needed.
        self.out_scaled_int = out_scaled_int
        self.eps = eps if eps is not None else torch.finfo(torch.float32).eps
        self.requant_mul_guard_bits = requant_mul_guard_bits \
            if requant_mul_guard_bits is not None else 7  # For processing bias, slope, zero point

    @torch.no_grad()
    def import_parameters(
            self, scale_in: torch.Tensor, zero_point_in: torch.Tensor,
            scale_out: Optional[torch.Tensor], zero_point_out: Optional[torch.Tensor],
            linear: nn.Linear, prelu: Optional[nn.PReLU]):
        assert scale_in.dtype == linear.weight.dtype == torch.float32
        assert scale_in.numel() == 1
        assert zero_point_in.dtype in (torch.int32, torch.int64)
        assert zero_point_in.numel() == 1
        assert linear.bias is not None
        if scale_in <= self.eps:
            print(f'Warning: {scale_in}')
            scale_in = scale_in.clip(min=self.eps)
        if self.out_scaled_int is True:
            assert scale_out.dtype == scale_in.dtype
            assert scale_out.numel() == 1
            if scale_out <= self.eps:
                print(f'Warning: {scale_out}')
                scale_out = scale_out.clip(min=self.eps)
            self.scale_out[:] = scale_out
        else:
            assert scale_out is None
        self.scale_in[:] = scale_in

        per_out_abs_max = linear.weight.abs().amax(dim=1)  # out_ch * in_ch
        scale_weight = per_out_abs_max / WeightRange
        if (scale_weight <= self.eps).any():
            print(f'Warning: {scale_weight}')
            scale_weight = scale_weight.clip(min=self.eps)
        self.scale_weight[:] = scale_weight
        self.weight[...] = (
            linear.weight / scale_weight[:, None]
        ).round().clip(-WeightRange, WeightRange).to(self.weight.dtype)

        self.zero_point_in[:] = zero_point_in
        scale_bias = scale_in * scale_weight
        self.bias[:] = (
            linear.bias / scale_bias - (zero_point_in.to(torch.float) * self.weight.to(torch.float)).sum(1)
        ).round().to(torch.int32)

        if self.with_prelu:
            assert prelu.weight.numel() == 1
            if prelu.weight.abs().item() > 63:
                print(f'Warning: abnormal prelu slope {prelu.weight.item()}.')
            self.slope[:] = (prelu.weight * (1 << 25)).round().to(torch.int32)  # Q6.25
        else:
            assert prelu is None

        if self.out_scaled_int is True:
            assert scale_out is not None
            requant_mul = scale_in * scale_weight / scale_out
            self.requant_shift[:] = torch.log2((1 << (32 - self.requant_mul_guard_bits)) / requant_mul).min().floor()
            assert self.requant_shift >= 0, self.requant_shift
            self.requant_mul[:] = (requant_mul * 2 ** self.requant_shift.to(torch.float)).round().to(torch.uint32)
            assert zero_point_out.dtype in (torch.int32, torch.int64)
            assert zero_point_out.numel() == 1
            self.zero_point_out[:] = zero_point_out
            self.int_zero_point_out[:] = zero_point_out.to(torch.int64) << self.requant_shift

        else:
            assert zero_point_out is None
            requant_mul = scale_in * scale_weight
            self.requant_shift[:] = torch.log2((1 << (32 - self.requant_mul_guard_bits)) / requant_mul).min().floor()
            assert self.requant_shift >= 0, self.requant_shift
            self.requant_mul[:] = (requant_mul * 2 ** self.requant_shift.to(torch.float)).round().to(torch.uint32)
            self.zero_point_out[:] = 0
            self.int_zero_point_out[:] = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # input scaled int
        mm_out = torch.empty((input.size(0), self.weight.size(0)), device=input.device, dtype=torch.int32)
        int_sparse_conv_ext.cutlass_gemm_int8(input, self.weight, self.bias, mm_out)

        if self.with_prelu is False:
            if self.out_scaled_int is True:
                output = int_sparse_conv_ext.requant_to_int8(  # scaled int
                    mm_out, self.requant_mul, self.int_zero_point_out,
                    self.requant_shift.item())

            else:
                output = int_sparse_conv_ext.requant_to_int32(  # shared fixed-point
                    mm_out, self.requant_mul, self.int_zero_point_out,
                    self.requant_shift.item() - SharedFxpShift)

        else:
            if self.out_scaled_int is True:
                output = int_sparse_conv_ext.prelu_requant_to_int8(  # scaled int
                    mm_out, self.slope, self.requant_mul, self.int_zero_point_out,
                    self.requant_shift.item())

            else:
                output = int_sparse_conv_ext.prelu_requant_to_int32(  # shared fixed-point
                    mm_out, self.slope, self.requant_mul, self.int_zero_point_out,
                    self.requant_shift.item() - SharedFxpShift)

        return output


class LinearIn8W8Out8(LinearIn8W8):
    def __init__(self, in_ch: int, out_ch: int, *args, **kwargs):
        super().__init__(in_ch, out_ch, False, True, *args, **kwargs)

    @torch.no_grad()
    def import_parameters(
            self, scale_in: torch.Tensor, zero_point_in: torch.Tensor,
            scale_out: torch.Tensor, zero_point_out: torch.Tensor,
            linear: nn.Linear):
        super().import_parameters(scale_in, zero_point_in, scale_out, zero_point_out, linear, None)


class LinearIn8W8Out32(LinearIn8W8):
    def __init__(self, in_ch: int, out_ch: int, *args, **kwargs):
        super().__init__(in_ch, out_ch, False, False, *args, **kwargs)

    @torch.no_grad()
    def import_parameters(
            self, scale_in: torch.Tensor, zero_point_in: torch.Tensor,
            linear: nn.Linear):
        super().import_parameters(scale_in, zero_point_in, None, None, linear, None)


class LinearPReLUIn8W8Out8(LinearIn8W8):
    def __init__(self, in_ch: int, out_ch: int, *args, **kwargs):
        super().__init__(in_ch, out_ch, True, True, *args, **kwargs)

    @torch.no_grad()
    def import_parameters(
            self, scale_in: torch.Tensor, zero_point_in: torch.Tensor,
            scale_out: torch.Tensor, zero_point_out: torch.Tensor,
            linear: nn.Linear, prelu: nn.PReLU):
        super().import_parameters(scale_in, zero_point_in, scale_out, zero_point_out, linear, prelu)


class LinearPReLUIn8W8Out32(LinearIn8W8):
    def __init__(self, in_ch: int, out_ch: int, *args, **kwargs):
        super().__init__(in_ch, out_ch, True, False, *args, **kwargs)

    @torch.no_grad()
    def import_parameters(
            self, scale_in: torch.Tensor, zero_point_in: torch.Tensor,
            linear: nn.Linear, prelu: nn.PReLU):
        super().import_parameters(scale_in, zero_point_in, None, None, linear, prelu)


def softmax_int32(input: torch.IntTensor) -> torch.IntTensor:
    return int_sparse_conv_ext.softmax_int32(input)
