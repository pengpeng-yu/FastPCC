import io
from typing import List, Union, Tuple, Optional, Literal
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse import SparseTensor

from lib.utils import Timer
from lib.torch_utils import TorchCudaMaxMemoryAllocated
from lib.data_utils import PCData
from lib.evaluators import PCCEvaluator
from lib.morton_code import morton_encode_magicbits
from lib.int_sparse_conv.cuda_ops import softmax_int32, SharedFxpShift, \
    sparse_conv_in8w8out32, SparseResBlockIn32W8Out32, \
    SparseConvIn8W8Out8, SparseConvIn8W8Out32, SparseConvPReLUIn8W8Out8, SparseConvPReLUIn8W8Out32, \
    PReLUIn32Out32, RequantFxpToScaledInt8,  \
    LinearIn8W8Out8, LinearIn8W8Out32, LinearPReLUIn8W8Out8, LinearPReLUIn8W8Out32
from .model_config import Config
from models.convolutional.lossy_coord_v3.rans_coder import RansEncoder, RansDecoder


log2_e = math.log2(math.e)


class OneScalePredictor(nn.Module):
    def __init__(self, channels, if_upsample=True, allow_single_ch=False):
        super(OneScalePredictor, self).__init__()
        if allow_single_ch is True:
            self.dec_init = SparseConvIn8W8Out32(1, channels)
        self.dec = SparseResBlockIn32W8Out32(channels)

        self.pred = SparseSequential(
            RequantFxpToScaledInt8(),
            SparseConvPReLUIn8W8Out8(channels, channels),
            LinearIn8W8Out32(channels, 255))

        self.if_upsample = if_upsample
        if self.if_upsample:
            self.upsample = SparseSequential(
                RequantFxpToScaledInt8(),
                LinearPReLUIn8W8Out32(channels + 8, channels),
                SparseResBlockIn32W8Out32(channels),
                RequantFxpToScaledInt8(),
                LinearIn8W8Out32(channels, channels * 8))
        else:
            self.upsample = None

        self.register_buffer('_shared_fxp_shift', torch.tensor(SharedFxpShift, dtype=torch.int32), persistent=False)

    def compress(self, cur_rec: SparseTensor, up_ref: SparseTensor, cur_bin: torch.IntTensor,
                 bin2oct_kernel, if_upsample):
        if cur_rec.F.shape[1] == 1:
            cur_rec = self.dec_init(cur_rec)
        cur_rec = self.dec(cur_rec)

        cur_pred = self.pred(cur_rec).F
        cur_oct = (cur_bin << bin2oct_kernel).sum(1, dtype=torch.int16).add_(-1)

        if if_upsample:
            cur_rec.F = torch.cat((cur_rec.F, cur_bin << self._shared_fxp_shift), 1)
            cur_rec = self.upsample(cur_rec)
            cur_rec = SparseTensor(
                cur_rec.F.reshape(cur_rec.F.shape[0], 8, cur_rec.F.shape[1] // 8)[cur_bin.bool()],
                up_ref.C,
                tuple(_ // 2 for _ in cur_rec.stride))
            cur_rec._caches = up_ref._caches
        return cur_rec, cur_pred, cur_oct

    def decompress(self, cur_rec: SparseTensor, device, bin2oct_kernel, unfold_kernel,
                   rans_decode_oct, if_upsample):
        if cur_rec.F.shape[1] == 1:
            cur_rec = self.dec_init(cur_rec)
        cur_rec = self.dec(cur_rec)

        cur_pred = self.pred(cur_rec).F
        cur_oct = rans_decode_oct(cur_pred, cur_rec.C.shape[0], device, torch.int16)
        del cur_pred
        cur_bin = ((cur_oct[:, None] + 1) >> bin2oct_kernel).bitwise_and_(1).bool()

        if if_upsample:
            cur_rec.F = torch.cat((cur_rec.F, cur_bin << self._shared_fxp_shift), 1)
            cur_rec = self.upsample(cur_rec)
            new_c = cur_rec.C[:, None]
            new_c[..., 1:] <<= 1
            cur_rec = SparseTensor(
                cur_rec.F.reshape(cur_rec.F.shape[0], 8, cur_rec.F.shape[1] // 8)[cur_bin],
                (new_c + unfold_kernel)[cur_bin],
                tuple(_ // 2 for _ in cur_rec.stride))
        return cur_rec, cur_bin


class OneScaleMultiStepPredictor(nn.Module):
    def __init__(self, channels, pred_steps=2, use_more_ch_for_multi_step_pred=True):
        super(OneScaleMultiStepPredictor, self).__init__()
        self.pred_steps = pred_steps
        if pred_steps == 2:
            self.embed = SparseSequential()
            out_ch = channels
            self.dec = SparseSequential(
                RequantFxpToScaledInt8(),
                LinearPReLUIn8W8Out32(channels + 8, out_ch),
                SparseResBlockIn32W8Out32(out_ch))
        elif use_more_ch_for_multi_step_pred:
            if pred_steps == 3:
                self.embed = SparseSequential(
                    RequantFxpToScaledInt8(),
                    SparseConvPReLUIn8W8Out32(8, 64, (2, 2, 2), (2, 2, 2)))
                out_ch = round(channels * 1.25)
                self.dec = SparseSequential(
                    RequantFxpToScaledInt8(),
                    LinearPReLUIn8W8Out32(channels + 64, out_ch),
                    SparseResBlockIn32W8Out32(out_ch)) if channels + 64 != out_ch else SparseResBlockIn32W8Out32(out_ch)
            elif pred_steps >= 4:
                self.embed = SparseSequential(
                    RequantFxpToScaledInt8(),
                    SparseConvPReLUIn8W8Out32(8, 512, (2 ** (pred_steps - 2),) * 3, (2 ** (pred_steps - 2),) * 3))
                out_ch = channels * 2
                self.dec = SparseSequential(
                    RequantFxpToScaledInt8(),
                    LinearPReLUIn8W8Out32(round(channels * 1.25) + 512, out_ch),
                    SparseResBlockIn32W8Out32(out_ch)) if round(channels * 1.25) + 512 != out_ch else SparseResBlockIn32W8Out32(out_ch)
            else: raise NotImplementedError
        else:
            assert pred_steps >= 3
            self.embed = SparseSequential(
                RequantFxpToScaledInt8(),
                (SparseConvPReLUIn8W8Out32 if channels >= 256 else SparseConvIn8W8Out32)(
                    8, channels, (2 ** (pred_steps - 2),) * 3, (2 ** (pred_steps - 2),) * 3))
            self.dec = SparseSequential(
                RequantFxpToScaledInt8(),
                LinearPReLUIn8W8Out32(channels + channels, channels),
                SparseResBlockIn32W8Out32(channels))
            out_ch = channels

        self.pred = nn.ModuleList()
        for idx in range(pred_steps):
            if idx == 0:
                self.pred.append(SparseSequential(
                    RequantFxpToScaledInt8(),
                    SparseConvPReLUIn8W8Out8(out_ch, out_ch),
                    LinearIn8W8Out32(out_ch, channels * 8)))
            elif idx != pred_steps - 1:
                self.pred.append(SparseSequential(
                    PReLUIn32Out32(), RequantFxpToScaledInt8(), LinearPReLUIn8W8Out8(channels + 8, channels),
                    SparseConvPReLUIn8W8Out8(channels, channels),
                    LinearIn8W8Out32(channels, channels * 8)))
            else:
                self.pred.append(SparseSequential(
                    RequantFxpToScaledInt8(),
                    SparseConvPReLUIn8W8Out8(channels, channels),
                    LinearIn8W8Out32(channels, 255)))

        self.register_buffer('_shared_fxp_shift', torch.tensor(SharedFxpShift, dtype=torch.int32), persistent=False)

    def compress(self, cur_rec: SparseTensor, cur_bins: List[SparseTensor], bin2oct_kernel):
        embed_in = SparseTensor(cur_bins[1].F << self._shared_fxp_shift, cur_bins[1].C, stride=cur_bins[1].stride)
        embed_in._caches = cur_rec._caches
        embed_f = self.embed(embed_in).F
        cur_rec.F = torch.cat([cur_rec.F, embed_f], 1)
        cur_rec = self.dec(cur_rec)

        cur_pred = self.pred[0](cur_rec)
        for idx, pred_block in enumerate(self.pred):
            if idx == 0: continue
            cur_bins[-idx].F = cur_bins[-idx].F.bool()
            cur_pred.F = cur_pred.F.reshape(
                cur_pred.F.shape[0], 8, cur_pred.F.shape[1] // 8)[cur_bins[-idx].F]
            if idx != len(self.pred) - 1:
                cur_pred.F = torch.cat([cur_pred.F, cur_bins[-idx - 1].F << self._shared_fxp_shift], 1)
            cur_pred.C = cur_bins[-idx - 1].C
            cur_pred.stride = cur_bins[-idx - 1].stride
            cur_pred = pred_block(cur_pred)

        cur_oct = (cur_bins[0].F << bin2oct_kernel).sum(1, dtype=torch.int16).add_(-1)
        return cur_rec, cur_pred.F, cur_oct

    def decompress(self, cur_rec: SparseTensor, cur_bins: List[torch.BoolTensor],
                   top_rec: torch.Tensor, top_stride: int,
                   device, bin2oct_kernel, unfold_kernel, rans_decode_oct):
        if len(cur_bins) == 1:
            top_rec = cur_rec.C
            top_stride = cur_rec.stride[0]
        top_rec = top_rec[:, None].clone()
        top_rec[..., 1:] <<= 1
        top_rec = (top_rec + unfold_kernel)[cur_bins[-1]]
        top_stride = top_stride // 2
        cur_rec._caches.cmaps[(top_stride,) * 3] = top_rec, None

        embed_in = SparseTensor(
            cur_bins[-1] << self._shared_fxp_shift,
            cur_rec._caches.cmaps[(top_stride * 2,) * 3][0],
            stride=(top_stride * 2,) * 3)
        embed_in._caches = cur_rec._caches
        embed_f = self.embed(embed_in).F
        cur_rec.F = torch.cat([cur_rec.F, embed_f], 1)
        cur_rec = self.dec(cur_rec)

        cur_pred = self.pred[0](cur_rec)
        for idx, pred_block in enumerate(self.pred):
            if idx == 0: continue
            cur_pred.F = cur_pred.F.reshape(
                cur_pred.F.shape[0], 8, cur_pred.F.shape[1] // 8)[cur_bins[idx - 1]]
            if idx != len(self.pred) - 1:
                cur_pred.F = torch.cat([cur_pred.F, cur_bins[idx] << self._shared_fxp_shift], 1)
            cur_pred.stride = tuple(_ // 2 for _ in cur_pred.stride)
            cur_pred.C = cur_rec._caches.cmaps[cur_pred.stride][0]
            cur_pred = pred_block(cur_pred)
        cur_oct = rans_decode_oct(cur_pred.F, cur_pred.F.shape[0], device, torch.int16)
        cur_bin = ((cur_oct[:, None] + 1) >> bin2oct_kernel).bitwise_and_(1).bool()
        return cur_rec, cur_bin, top_rec, top_stride


class Model(nn.Module):
    def __init__(self, cfg: Config, device):
        super(Model, self).__init__()
        self.cfg = cfg
        self.device = device
        self.evaluator = PCCEvaluator(
            cal_mpeg_pc_error=not cfg.cal_avs_pc_evalue, cal_avs_pc_evalue=cfg.cal_avs_pc_evalue)

        self.max_downsample_times_wo_recurrent = int(np.log2(cfg.max_stride_wo_recurrent))
        self.max_downsample_times = int(np.log2(cfg.max_stride))
        assert cfg.fea_stride >= 2

        self.blocks_dec = nn.ModuleList()
        for idx in range(self.max_downsample_times_wo_recurrent):
            pred_steps = int(np.log2(cfg.fea_stride)) - idx
            if pred_steps < 1:
                self.blocks_dec.append(OneScalePredictor(cfg.channels, True, False))
            elif pred_steps == 1:
                self.blocks_dec.append(OneScalePredictor(cfg.channels, False, False))
            else:
                self.blocks_dec.append(OneScaleMultiStepPredictor(
                    cfg.channels, pred_steps, cfg.use_more_ch_for_multi_step_pred))
        self.block_dec_recurrent = OneScalePredictor(cfg.channels, True, True)

        self.register_buffer('fold2bin_kernel', torch.empty(8, 1 * 8, 1, dtype=torch.int8), persistent=False)
        with torch.no_grad():
            self.fold2bin_kernel.reshape(8, 8)[...] = torch.eye(8)
        self.register_buffer('bin2oct_kernel', torch.arange(7, -1, -1, dtype=torch.uint8), persistent=False)
        self.register_buffer('unfold_kernel', torch.tensor(
            ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1),
             (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1)), dtype=torch.int32)[None], persistent=False)

        self.flag_init_coder = False

    def train(self, mode: bool = True):
        if mode is False and self.flag_init_coder is False:
            self.rans_encoder = RansEncoder(32 * 1024 * 1024)  # 32MB
            self.rans_decoder = RansDecoder()
            self.fea_side_info_cdf1 = np.arange(2, 65537, dtype=np.uint16)[None]
            self.fea_side_info_cdf2 = np.arange(1, 129, dtype=np.uint16)[None] * 512
            self.fea_side_info_cdf1[:, -1] = 65535
            self.fea_side_info_cdf2[:, -1] = 65535
            self.flag_init_coder = True
        return super(Model, self).train(mode=mode)

    @torch.no_grad()
    def get_bin(self, input, ones_feats):
        sp_f = input.F
        input.F = ones_feats[:input.C.shape[0]]
        cur_kmap_tag = (input.stride, (2, 2, 2), (2, 2, 2))
        output_coords = input.C.clone()
        output_coords[:, 1:] >>= 1
        output_coords = torch.unique_consecutive(output_coords, dim=0)
        output_stride = tuple(_a * 2 for _a in input.stride)

        conv_out_feats, hashmap_kv, in_out_maps = sparse_conv_in8w8out32(
            ones_feats[:input.C.shape[0]],
            self.fold2bin_kernel,
            input.C, output_coords,
            (2, 2, 2), (2, 2, 2),
            if_in_coords_equals_out_coords=True
        )

        input.F = sp_f
        caches = input._caches
        if input.stride != (1, 1, 1):  # Caches of (1, 1, 1) are not needed
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

    def forward(self, pc_data: PCData):
        if self.training:
            raise NotImplementedError
        else:
            assert pc_data.batch_size == 1, 'Only supports batch size == 1 during testing.'
            return self.test_forward(pc_data)

    @staticmethod
    def get_init_pc(xyz: torch.Tensor, stride: int = 1) -> SparseTensor:
        # Input coordinates are assumed to be Morton-sorted with unique points.
        sparse_pc_feature = torch.ones((xyz.shape[0], 1), dtype=torch.int8, device=xyz.device)
        sparse_pc = SparseTensor(sparse_pc_feature, xyz, (stride,) * 3)
        return sparse_pc

    def test_forward(self, pc_data: PCData):
        not_part = isinstance(pc_data.xyz, torch.Tensor)
        with Timer() as encoder_t, TorchCudaMaxMemoryAllocated() as encoder_m:
            compressed_bytes = self.compress(pc_data.xyz) if not_part else \
                self.compress_partitions(pc_data.xyz)
            torch.cuda.synchronize()

        torch.cuda.empty_cache()
        with Timer() as decoder_t, TorchCudaMaxMemoryAllocated() as decoder_m:
            coord_recon = self.decompress(compressed_bytes) if not_part else \
                self.decompress_partitions(compressed_bytes)
            torch.cuda.synchronize()

        if pc_data.inv_transform is not None:
            inv_trans = pc_data.inv_transform[0].to(coord_recon.device)
            coord_recon = coord_recon * inv_trans[3]
            coord_recon += inv_trans[None, :3]
            compressed_bytes = pc_data.inv_transform[0].numpy().astype('<f4').tobytes() + compressed_bytes
        ret = self.evaluator.log(
            pred=coord_recon,
            org_points_num=pc_data.org_points_num[0],
            compressed_bytes=compressed_bytes,
            file_path=pc_data.file_path[0],
            resolution=pc_data.resolution[0],
            results_dir=pc_data.results_dir,
            extra_info_dict={
                'encode time': encoder_t.elapsed_time,
                'encode memory': encoder_m.max_memory_allocated_kb,
                'decode time': decoder_t.elapsed_time,
                'decode memory': decoder_m.max_memory_allocated_kb}
        )
        return ret

    @staticmethod
    def batch_quantize_pmf_torch(pmfs: torch.Tensor) -> torch.Tensor:
        assert pmfs.dtype == torch.int32
        pmfs = softmax_int32(pmfs >> torch.tensor((SharedFxpShift - 16), dtype=torch.int32, device=pmfs.device))
        assert pmfs.dtype == torch.uint32  # Q32
        pmfs = ((pmfs.to(torch.int64) * (65536 - pmfs.shape[1])) >> 32).add_(1)
        pmfs.cumsum_(-1)
        pmfs[:, -1] = 65535
        pmfs = pmfs.to('cpu', torch.uint16, memory_format=torch.contiguous_format, non_blocking=True)
        return pmfs

    def rans_encode_oct(self, quantized_cdfs: torch.Tensor, values: torch.Tensor) -> int:
        assert values.dtype == torch.uint16
        encoded_size = self.rans_encoder.encode(quantized_cdfs.numpy(), values.numpy())
        return encoded_size

    def rans_decode_oct(self, logits: torch.Tensor, length: int, device, dtype) -> torch.Tensor:
        quantized_cdfs = self.batch_quantize_pmf_torch(logits).numpy()
        ret_values = np.empty((length,), dtype=np.uint16)
        torch.cuda.synchronize()
        self.rans_decoder.decode(quantized_cdfs, ret_values)
        return torch.from_numpy(ret_values).reshape(length).to(device, dtype, non_blocking=True)

    def rans_encode_fea(self, quantized_cdf: torch.Tensor, rounded: torch.Tensor, rounded_min: torch.Tensor = None):
        quantized_cdf = quantized_cdf.numpy()
        self.rans_encoder.encode(quantized_cdf[None], rounded.numpy())
        self.rans_encoder.encode(self.fea_side_info_cdf1, quantized_cdf[:-1] - 1)
        assert len(quantized_cdf) - 2 <= self.fea_side_info_cdf2.shape[1], len(quantized_cdf)
        self.rans_encoder.encode(self.fea_side_info_cdf2, np.array((len(quantized_cdf) - 2,), dtype=np.uint16))
        if rounded_min is not None:
            self.rans_encoder.encode(self.fea_side_info_cdf2, rounded_min[None].numpy())
        return True

    def rans_decode_fea(self, length: int, device, dtype, decode_rounded_min: bool = True) -> torch.Tensor:
        if decode_rounded_min:
            rounded_min = np.empty((1,), dtype=np.uint16)
            self.rans_decoder.decode(self.fea_side_info_cdf2, rounded_min)
        cdf_len = np.empty((1,), dtype=np.uint16)
        self.rans_decoder.decode(self.fea_side_info_cdf2, cdf_len)
        cdf = np.empty((cdf_len.item() + 1,), dtype=np.uint16)
        self.rans_decoder.decode(self.fea_side_info_cdf1, cdf)
        cdf = np.pad(cdf + 1, (0, 1))
        cdf[-1] = 65535

        decoded = np.empty((length,), dtype=np.uint16)
        self.rans_decoder.decode(cdf[None], decoded)
        decoded = torch.from_numpy(decoded).to(device, dtype, non_blocking=True)
        if decode_rounded_min:
            decoded -= torch.from_numpy(rounded_min).to(device, dtype, non_blocking=True)
        return decoded

    def compress(self, xyz: torch.Tensor) -> bytes:
        coord_offset = xyz[:, 1:].amin(0)
        xyz = xyz - F.pad(coord_offset, (1, 0))
        xyz = xyz[torch.argsort(morton_encode_magicbits(xyz[:, 1:], inverse=True))]
        org = self.get_init_pc(xyz, 1)

        blocks_dec = self.blocks_dec[self.cfg.skip_top_scales_num:]

        strided_list = [org]
        for _ in range(0, self.max_downsample_times - self.cfg.skip_top_scales_num):
            strided_list.append(self.get_bin(strided_list[-1], org.F))

        cached_c = strided_list[-1].C[:, 1:].reshape(-1)
        cached_c_cpu = cached_c.to('cpu', torch.uint16, memory_format=torch.contiguous_format, non_blocking=True)
        cached_c_counts = torch.bincount(cached_c, minlength=2).to(torch.int64)
        cached_c_pmfs = (cached_c_counts * torch.div(
            (65536 - cached_c_counts.shape[0]) << 8, cached_c.numel(), rounding_mode='floor')) >> 8
        cached_c_pmfs.add_(1)
        cached_c_cdf = cached_c_pmfs.cumsum(-1)
        cached_c_cdf[-1] = 65535
        cached_c_cdf = cached_c_cdf.to('cpu', torch.uint16, memory_format=torch.contiguous_format, non_blocking=True)

        cur_rec = SparseTensor(
            org.F[:strided_list[-1].C.shape[0]], strided_list[-1].C,
            (2 ** (self.max_downsample_times - self.cfg.skip_top_scales_num),) * 3)
        cur_rec._caches = org._caches

        cached_list = []
        for idx in range(self.max_downsample_times - self.cfg.skip_top_scales_num, 0, -1):
            if idx > len(blocks_dec):
                block_dec = self.block_dec_recurrent
            else:
                block_dec = blocks_dec[idx - 1]
            if isinstance(block_dec, OneScalePredictor):
                cur_rec, cur_pred, cur_oct = block_dec.compress(
                    cur_rec, strided_list[idx - 1], strided_list[idx].F, self.bin2oct_kernel,
                    if_upsample=idx != 1 and block_dec.if_upsample)
            else:
                assert isinstance(block_dec, OneScaleMultiStepPredictor)
                cur_rec, cur_pred, cur_oct = block_dec.compress(
                    cur_rec, strided_list[idx: idx + block_dec.pred_steps], self.bin2oct_kernel)
            cached_list.append((
                self.batch_quantize_pmf_torch(cur_pred),
                cur_oct.to('cpu', torch.uint16, non_blocking=True),
            ))

        torch.cuda.synchronize()
        while cached_list:
            cur_pred, cur_oct = cached_list.pop()
            self.rans_encode_oct(cur_pred, cur_oct)
        self.rans_encode_fea(cached_c_cdf, cached_c_cpu)

        with io.BytesIO() as bs:
            for _ in coord_offset.tolist():
                bs.write(int_to_bytes(_, 2))
            bs.write(int_to_bytes(cached_c.shape[0] // 3, 2))
            bs.write(self.rans_encoder.flush())
            compressed_bytes = bs.getvalue()
        return compressed_bytes

    def compress_partitions(self, batched_coord: List[torch.Tensor]) -> bytes:
        compressed_bytes_list = []
        for idx in range(1, len(batched_coord)):
            # The first one is supposed to be the original coordinates.
            compressed_bytes = self.compress(batched_coord[idx])
            compressed_bytes_list.append(compressed_bytes)

        concat_bytes = b''.join((int_to_bytes(len(s), 3) + s for s in compressed_bytes_list))
        return concat_bytes

    def decompress(self, compressed_bytes: bytes) -> torch.Tensor:
        device = self.device
        coord_offset = []
        with io.BytesIO(compressed_bytes) as bs:
            for _ in range(3):
                coord_offset.append(bytes_to_int(bs.read(2)))
            bottom_points_num = bytes_to_int(bs.read(2))
            rans_comp_bytes = bs.read()  # Keep this reference
            self.rans_decoder.flush(rans_comp_bytes)
        coord_offset = torch.tensor(coord_offset, device=device, dtype=torch.int32)[None]

        cur_rec = self.get_init_pc(
            F.pad(self.rans_decode_fea(
                bottom_points_num * 3, device, torch.int32, decode_rounded_min=False).reshape(-1, 3), (1, 0, 0, 0)),
            2 ** (self.max_downsample_times - self.cfg.skip_top_scales_num))

        blocks_dec = self.blocks_dec[self.cfg.skip_top_scales_num:]
        cur_bins, top_rec, top_stride = [], None, None  # for OneScaleMultiStepPredictor
        for idx in range(self.max_downsample_times - self.cfg.skip_top_scales_num, 0, -1):
            if idx > len(blocks_dec):
                block_dec = self.block_dec_recurrent
            else:
                block_dec = blocks_dec[idx - 1]
            if isinstance(block_dec, OneScalePredictor):
                cur_rec, cur_bin = block_dec.decompress(
                    cur_rec, device, self.bin2oct_kernel, self.unfold_kernel,
                    self.rans_decode_oct, if_upsample=idx != 1 and block_dec.if_upsample)
            else:
                assert isinstance(block_dec, OneScaleMultiStepPredictor)
                cur_bins.append(cur_bin)
                cur_rec, cur_bin, top_rec, top_stride = block_dec.decompress(
                    cur_rec, cur_bins, top_rec, top_stride,
                    device, self.bin2oct_kernel, self.unfold_kernel, self.rans_decode_oct)

        if top_rec is None:  # recon via OneScalePredictor
            assert cur_rec.stride[0] == 2
            coord_recon = cur_rec.C[:, None, 1:]
        else:  # recon via OneScaleMultiStepPredictor
            assert top_stride == 2
            coord_recon = top_rec[:, None, 1:]
        coord_recon <<= 1
        coord_recon = (coord_recon + self.unfold_kernel[:, :, 1:])[cur_bin]
        coord_recon += coord_offset
        return coord_recon

    def decompress_partitions(self, concat_bytes: bytes) -> torch.Tensor:
        coord_recon_list = []
        concat_bytes_len = len(concat_bytes)

        with io.BytesIO(concat_bytes) as bs:
            while bs.tell() != concat_bytes_len:
                length = bytes_to_int(bs.read(3))
                coord_recon = self.decompress(bs.read(length))
                coord_recon_list.append(coord_recon)

        coord_recon_concat = torch.cat(coord_recon_list, 0)
        return coord_recon_concat


class SparseSequential(nn.Sequential):
    def forward(self, input: SparseTensor) -> SparseTensor:
        x = SparseTensor(input.F, input.C, input.stride, input.spatial_range)
        x._caches = input._caches
        for module in self:
            if isinstance(module, (PReLUIn32Out32, RequantFxpToScaledInt8,
                                   LinearIn8W8Out8, LinearIn8W8Out32, LinearPReLUIn8W8Out8, LinearPReLUIn8W8Out32)):
                x.F = module(x.F)
            else:
                x = module(x)
        return x


def int_to_bytes(x, length, byteorder: Literal['little', 'big'] = 'little', signed=False):
    assert isinstance(x, int)
    return x.to_bytes(length, byteorder=byteorder, signed=signed)


def bytes_to_int(s, byteorder: Literal['little', 'big'] = 'little', signed=False):
    assert isinstance(s, bytes)
    return int.from_bytes(s, byteorder=byteorder, signed=signed)
