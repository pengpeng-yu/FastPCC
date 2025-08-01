import io
from typing import List, Union, Tuple, Optional, Literal
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor

from lib.utils import Timer
from lib.torch_utils import TorchCudaMaxMemoryAllocated
from lib.data_utils import PCData
from lib.evaluators import PCCEvaluator
from lib.morton_code import morton_encode_magicbits
from .model_config import Config
from models.convolutional.lossy_coord_v3.rans_coder import RansEncoder, RansDecoder


log2_e = math.log2(math.e)


def me_conv_flops_counter_hook(module, input, out_sp):
    in_sp = input[0]
    kernel_map = in_sp.coordinate_manager.kernel_map(
        in_sp.coordinate_map_key,
        out_sp.coordinate_map_key,
        stride=module.kernel_generator.kernel_size,
        kernel_size=module.kernel_generator.kernel_stride)
    map_num = sum([_.size(1) for _ in kernel_map.values()])
    flops = in_sp.F.size(1) * out_sp.F.size(1) * map_num * 2 + \
            map_num * out_sp.F.size(1) + \
            out_sp.F.size(0) * out_sp.F.size(1)
    module.__flops__ += flops


ME.MinkowskiConvolution.__flops__ = me_conv_flops_counter_hook


class OneScalePredictor(nn.Module):
    def __init__(self, channels, if_upsample=True, allow_single_ch=False):
        super(OneScalePredictor, self).__init__()
        if allow_single_ch is True:
            self.dec_init = ME.MinkowskiConvolution(1, channels, 3, 1, 1, bias=True, dimension=3)
        self.dec = Block(channels)

        self.pred = SparseSequential(
            ME.MinkowskiConvolution(channels, channels, 3, 1, 1, bias=True, dimension=3), ME.MinkowskiPReLU(),
            ME.MinkowskiLinear(channels, 255, bias=True))

        self.if_upsample = if_upsample
        if self.if_upsample:
            self.upsample = SparseSequential(
                ME.MinkowskiLinear(channels + 8, channels, bias=True), ME.MinkowskiPReLU(),
                Block(channels),
                ME.MinkowskiLinear(channels, channels * 8, bias=True))
        else:
            self.upsample = None

    def forward(self, cur_rec: SparseTensor, up_ref: SparseTensor, cur_bin: torch.FloatTensor,
                device, points_num: List[int], bin2oct_kernel):
        batch_size = len(points_num)

        if cur_rec.F.shape[1] == 1:
            cur_rec = self.dec_init(cur_rec)
        cur_rec = self.dec(cur_rec)

        cur_pred = self.pred(cur_rec).F
        cur_oct = (cur_bin.to(torch.uint8) << bin2oct_kernel).sum(1, dtype=torch.int64).add_(-1)
        if batch_size > 1:
            divider = torch.searchsorted(
                cur_rec.C[:, 0].contiguous(), torch.arange(batch_size + 1, device=device, dtype=torch.int32))
            scattered_points_num = torch.empty((cur_rec.C.shape[0],), device=device, dtype=torch.float)
            for b in range(batch_size):
                scattered_points_num[divider[b]: divider[b + 1]] = points_num[b]
            cur_geo_loss = (F.cross_entropy(cur_pred, cur_oct, reduction='none')
                            / scattered_points_num).sum() * (log2_e / batch_size)
        else:
            cur_geo_loss = F.cross_entropy(cur_pred, cur_oct, reduction='sum') \
                            * (log2_e / batch_size / points_num[0])

        if self.if_upsample:
            cur_rec = self.upsample(ME.SparseTensor(
                torch.cat((cur_rec.F, cur_bin), 1),
                coordinate_map_key=cur_rec.coordinate_map_key,
                coordinate_manager=cur_rec.coordinate_manager))
            cur_rec = SparseTensor(
                cur_rec.F.reshape(cur_rec.F.shape[0], 8, cur_rec.F.shape[1] // 8)[cur_bin.bool()],
                coordinate_map_key=ME.CoordinateMapKey([_ // 2 for _ in cur_rec.tensor_stride], ''),
                coordinate_manager=cur_rec.coordinate_manager)
        return cur_rec, cur_geo_loss

    def compress(self, cur_rec: SparseTensor, up_ref: SparseTensor, cur_bin: torch.FloatTensor,
                 bin2oct_kernel, if_upsample):
        if cur_rec.F.shape[1] == 1:
            cur_rec = self.dec_init(cur_rec)
        cur_rec = self.dec(cur_rec)

        cur_pred = self.pred(cur_rec).F
        cur_oct = (cur_bin.to(torch.uint8) << bin2oct_kernel).sum(1, dtype=torch.int16).add_(-1)

        if if_upsample:
            cur_rec = self.upsample(ME.SparseTensor(
                torch.cat((cur_rec.F, cur_bin), 1),
                coordinate_map_key=cur_rec.coordinate_map_key,
                coordinate_manager=cur_rec.coordinate_manager))
            cur_rec = SparseTensor(
                cur_rec.F.reshape(cur_rec.F.shape[0], 8, cur_rec.F.shape[1] // 8)[cur_bin.bool()],
                coordinate_map_key=ME.CoordinateMapKey([_ // 2 for _ in cur_rec.tensor_stride], ''),
                coordinate_manager=cur_rec.coordinate_manager)
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
            cur_rec = self.upsample(ME.SparseTensor(
                torch.cat((cur_rec.F, cur_bin), 1),
                coordinate_map_key=cur_rec.coordinate_map_key,
                coordinate_manager=cur_rec.coordinate_manager))
            new_c = cur_rec.C[:, None]
            cur_rec = SparseTensor(
                cur_rec.F.reshape(cur_rec.F.shape[0], 8, cur_rec.F.shape[1] // 8)[cur_bin],
                (new_c + (unfold_kernel * cur_rec.tensor_stride[0] // 2))[cur_bin],
                tuple(_ // 2 for _ in cur_rec.tensor_stride))
        return cur_rec, cur_bin


class OneScaleMultiStepPredictor(nn.Module):
    def __init__(self, channels, pred_steps=2, use_more_ch_for_multi_step_pred=True):
        super(OneScaleMultiStepPredictor, self).__init__()
        self.pred_steps = pred_steps
        if pred_steps == 2:
            self.embed = SparseSequential()
            out_ch = channels
            self.dec = SparseSequential(
                ME.MinkowskiLinear(channels + 8, out_ch), ME.MinkowskiPReLU(),
                Block(out_ch))
        elif use_more_ch_for_multi_step_pred:
            if pred_steps == 3:
                self.embed = SparseSequential(
                    ME.MinkowskiConvolution(8, 64, 2, 2, bias=True, dimension=3), ME.MinkowskiPReLU())
                out_ch = round(channels * 1.25)
                self.dec = SparseSequential(
                    ME.MinkowskiLinear(channels + 64, out_ch), ME.MinkowskiPReLU(),
                    Block(out_ch)) if channels + 64 != out_ch else Block(out_ch)
            elif pred_steps == 4:
                self.embed = SparseSequential(
                    ME.MinkowskiConvolution(8, 512, kernel_size=4, stride=4, bias=True, dimension=3), ME.MinkowskiPReLU())
                out_ch = channels * 2
                self.dec = SparseSequential(
                    ME.MinkowskiLinear(round(channels * 1.25) + 512, out_ch), ME.MinkowskiPReLU(),
                    Block(out_ch)) if round(channels * 1.25) + 512 != out_ch else Block(out_ch)
            else: raise NotImplementedError
        else:
            assert pred_steps >= 3
            self.embed = SparseSequential(
                ME.MinkowskiConvolution(8, 64, 2 ** (pred_steps - 2), 2 ** (pred_steps - 2), bias=True, dimension=3), ME.MinkowskiPReLU())
            self.dec = SparseSequential(
                ME.MinkowskiLinear(channels + 64, channels), ME.MinkowskiPReLU(),
                Block(channels))
            out_ch = channels

        self.pred = nn.ModuleList()
        for idx in range(pred_steps):
            if idx == 0:
                self.pred.append(SparseSequential(
                    ME.MinkowskiConvolution(out_ch, out_ch, 3, 1, 1, bias=True, dimension=3), ME.MinkowskiPReLU(),
                    ME.MinkowskiLinear(out_ch, (channels * 8), bias=True)))
            elif idx != pred_steps - 1:
                self.pred.append(SparseSequential(
                    ME.MinkowskiPReLU(), ME.MinkowskiLinear(channels + 8, channels, bias=True), ME.MinkowskiPReLU(),
                    ME.MinkowskiConvolution(channels, channels, 3, 1, 1, bias=True, dimension=3), ME.MinkowskiPReLU(),
                    ME.MinkowskiLinear(channels, (channels * 8), bias=True)))
            else:
                self.pred.append(SparseSequential(
                    ME.MinkowskiConvolution(channels, channels, 3, 1, 1, bias=True, dimension=3), ME.MinkowskiPReLU(),
                    ME.MinkowskiLinear(channels, 255, bias=True)))

    def forward(self, cur_rec: SparseTensor, cur_ref: SparseTensor,
                cur_bins: List[Union[SparseTensor, torch.BoolTensor]],
                device, points_num: List[int], bin2oct_kernel):
        batch_size = len(points_num)

        embed_f = self.embed(cur_bins[1]).F
        cur_rec = self.dec(ME.SparseTensor(
            torch.cat([cur_rec.F, embed_f], 1),
            coordinate_map_key=cur_rec.coordinate_map_key,
            coordinate_manager=cur_rec.coordinate_manager
        ))

        cur_pred = self.pred[0](cur_rec)
        for idx, pred_block in enumerate(self.pred):
            if idx == 0: continue
            if isinstance(cur_bins[-idx], SparseTensor):
                cur_bins[-idx] = cur_bins[-idx].F.bool()
            if isinstance(cur_bins[-idx - 1], SparseTensor):
                cur_bins[-idx - 1] = cur_bins[-idx - 1].F.bool()
            tmp_f = cur_pred.F.reshape(
                cur_pred.F.shape[0], 8, cur_pred.F.shape[1] // 8)[cur_bins[-idx]]
            if idx != len(self.pred) - 1:
                tmp_f = torch.cat([tmp_f, cur_bins[-idx - 1]], 1)
            cur_pred = pred_block(ME.SparseTensor(
                tmp_f,
                coordinate_map_key=ME.CoordinateMapKey([_ // 2 for _ in cur_pred.tensor_stride], ''),
                coordinate_manager=cur_pred.coordinate_manager
            ))

        cur_oct = (cur_bins[0].to(torch.uint8) << bin2oct_kernel).sum(1, dtype=torch.int64).add_(-1)
        if batch_size > 1:
            divider = torch.searchsorted(
                cur_ref.C[:, 0].contiguous(), torch.arange(batch_size + 1, device=device, dtype=torch.int32))
            scattered_points_num = torch.empty((cur_ref.C.shape[0],), device=device, dtype=torch.float)
            for b in range(batch_size):
                scattered_points_num[divider[b]: divider[b + 1]] = points_num[b]
            cur_geo_loss = (F.cross_entropy(cur_pred.F, cur_oct, reduction='none')
                            / scattered_points_num).sum() * (log2_e / batch_size)
        else:
            cur_geo_loss = F.cross_entropy(cur_pred.F, cur_oct, reduction='sum') \
                           * (log2_e / batch_size / points_num[0])
        return cur_rec, cur_geo_loss

    def compress(self, cur_rec: SparseTensor, cur_bins: List[SparseTensor], bin2oct_kernel):
        embed_f = self.embed(cur_bins[1]).F
        cur_rec = self.dec(ME.SparseTensor(
            torch.cat([cur_rec.F, embed_f], 1),
            coordinate_map_key=cur_rec.coordinate_map_key,
            coordinate_manager=cur_rec.coordinate_manager
        ))

        cur_pred = self.pred[0](cur_rec)
        for idx, pred_block in enumerate(self.pred):
            if idx == 0: continue
            if isinstance(cur_bins[-idx], SparseTensor):
                cur_bins[-idx] = cur_bins[-idx].F.bool()
            if isinstance(cur_bins[-idx - 1], SparseTensor):
                cur_bins[-idx - 1] = cur_bins[-idx - 1].F.bool()
            tmp_f = cur_pred.F.reshape(
                cur_pred.F.shape[0], 8, cur_pred.F.shape[1] // 8)[cur_bins[-idx]]
            if idx != len(self.pred) - 1:
                tmp_f = torch.cat([tmp_f, cur_bins[-idx - 1]], 1)
            cur_pred = pred_block(ME.SparseTensor(
                tmp_f,
                coordinate_map_key=ME.CoordinateMapKey([_ // 2 for _ in cur_pred.tensor_stride], ''),
                coordinate_manager=cur_pred.coordinate_manager
            ))

        cur_oct = (cur_bins[0].to(torch.uint8) << bin2oct_kernel).sum(1, dtype=torch.int16).add_(-1)
        return cur_rec, cur_pred.F, cur_oct

    def decompress(self, cur_rec: SparseTensor, cur_bins: List[torch.BoolTensor],
                   top_rec: torch.Tensor, top_stride: int,
                   device, bin2oct_kernel, unfold_kernel, rans_decode_oct):
        if len(cur_bins) == 1:
            top_rec = cur_rec.C
            top_stride = cur_rec.tensor_stride[0]
        top_rec = top_rec[:, None].clone()
        top_stride = top_stride // 2
        top_rec = (top_rec + unfold_kernel * top_stride)[cur_bins[-1]]
        cur_rec.coordinate_manager.insert_and_map(top_rec, top_stride)

        embed_in = SparseTensor(
            cur_bins[-1].to(cur_rec.F.dtype),
            coordinate_map_key=ME.CoordinateMapKey([top_stride * 2] * 3, ''),
            coordinate_manager=cur_rec.coordinate_manager)
        embed_f = self.embed(embed_in).F
        cur_rec = self.dec(ME.SparseTensor(
            torch.cat([cur_rec.F, embed_f], 1),
            coordinate_map_key=cur_rec.coordinate_map_key,
            coordinate_manager=cur_rec.coordinate_manager
        ))

        cur_pred = self.pred[0](cur_rec)
        for idx, pred_block in enumerate(self.pred):
            if idx == 0: continue
            tmp_f = cur_pred.F.reshape(
                cur_pred.F.shape[0], 8, cur_pred.F.shape[1] // 8)[cur_bins[idx - 1]]
            if idx != len(self.pred) - 1:
                tmp_f = torch.cat([tmp_f, cur_bins[idx]], 1)
            cur_pred = pred_block(ME.SparseTensor(
                tmp_f,
                coordinate_map_key=ME.CoordinateMapKey([_ // 2 for _ in cur_pred.tensor_stride], ''),
                coordinate_manager=cur_pred.coordinate_manager
            ))
        cur_oct = rans_decode_oct(cur_pred.F, cur_pred.F.shape[0], device, torch.int16)
        cur_bin = ((cur_oct[:, None] + 1) >> bin2oct_kernel).bitwise_and_(1).bool()
        return cur_rec, cur_bin, top_rec, top_stride


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super(Model, self).__init__()
        ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
        self.minkowski_algorithm = getattr(ME.MinkowskiAlgorithm, cfg.minkowski_algorithm)

        self.cfg = cfg
        self.evaluator = PCCEvaluator()

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

        fold2bin_kernel = torch.empty(8, 1, 1 * 8, dtype=torch.float)
        fold2bin_kernel.reshape(8, 8)[...] = torch.eye(8)
        self.fold2bin_conv = ME.MinkowskiConvolution(1, 8, (2, 2, 2), (2, 2, 2), bias=False, dimension=3)
        with torch.no_grad():
            self.fold2bin_conv.kernel[...] = fold2bin_kernel
        self.register_buffer('bin2oct_kernel', torch.arange(7, -1, -1, dtype=torch.uint8), persistent=False)
        self.register_buffer('unfold_kernel', torch.tensor(
            ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0),
             (0, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 1, 1)), dtype=torch.int32)[None], persistent=False)

        self.flag_init_coder = False

    def set_global_cm(self):
        ME.clear_global_coordinate_manager()
        global_coord_mg = ME.CoordinateManager(
            D=3,
            coordinate_map_type=ME.CoordinateMapType.CUDA if
            next(self.parameters()).device.type == 'cuda'
            else ME.CoordinateMapType.CPU,
            minkowski_algorithm=self.minkowski_algorithm
        )
        ME.set_global_coordinate_manager(global_coord_mg)
        return global_coord_mg

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
    def get_bin(self, sp, ones_feats):
        ret = self.fold2bin_conv(ME.SparseTensor(
            ones_feats[:sp.C.shape[0]],
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager))
        # assert (ret.F == ret.F.round()).all()
        return ret

    def forward(self, pc_data: PCData):
        if self.training:
            return self.train_forward(pc_data.xyz, pc_data.points_num, pc_data.training_step)
        else:
            assert pc_data.batch_size == 1, 'Only supports batch size == 1 during testing.'
            return self.test_forward(pc_data)

    def get_init_pc(self, xyz: torch.Tensor, stride: int = 1) -> SparseTensor:
        # Input coordinates are assumed to be Morton-sorted with unique points.
        global_coord_mg = self.set_global_cm()
        sparse_pc_feature = torch.ones((xyz.shape[0], 1), dtype=torch.float, device=xyz.device)
        sparse_pc = SparseTensor(sparse_pc_feature, xyz, (stride,) * 3, coordinate_manager=global_coord_mg)
        return sparse_pc

    def train_forward(self, xyz: torch.Tensor, points_num: List[int], training_step: int):
        org = self.get_init_pc(xyz)
        device = org.F.device

        strided_list = [org]
        for _ in range(0, self.max_downsample_times):
            strided_list.append(self.get_bin(strided_list[-1], org.F))

        cur_rec = SparseTensor(
            org.F[:strided_list[-1].C.shape[0]],
            coordinate_map_key=strided_list[-1].coordinate_map_key,
            coordinate_manager=strided_list[-1].coordinate_manager)

        loss_dict = {}
        for idx in range(self.max_downsample_times, 0, -1):
            if idx > len(self.blocks_dec):
                block_dec = self.block_dec_recurrent
            else:
                block_dec = self.blocks_dec[idx - 1]
            if isinstance(block_dec, OneScalePredictor):
                cur_rec, geo_bpp_loss = block_dec(
                    cur_rec, strided_list[idx - 1], strided_list[idx].F,
                    device, points_num, self.bin2oct_kernel)
            else:
                assert isinstance(block_dec, OneScaleMultiStepPredictor)
                cur_rec, geo_bpp_loss = block_dec(
                    cur_rec, strided_list[idx], strided_list[idx: idx + block_dec.pred_steps],
                    device, points_num, self.bin2oct_kernel)
            loss_dict[f'stride{2 ** idx}_geo_loss'] = geo_bpp_loss

        loss_dict['loss'] = sum(loss_dict.values())
        for k, v in loss_dict.items():
            if k != 'loss':
                loss_dict[k] = v.item()
        return loss_dict

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
    def batch_quantize_pmf_torch(pmfs: torch.Tensor, softmax=True) -> torch.Tensor:
        if softmax:
            pmfs = F.softmax(pmfs, dim=-1)
        pmfs.mul_(65536 - pmfs.shape[1]).floor_().add_(1)
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
        xyz = xyz[torch.argsort(morton_encode_magicbits(xyz[:, 1:], inverse=False))]
        org = self.get_init_pc(xyz, 1)

        blocks_dec = self.blocks_dec[self.cfg.skip_top_scales_num:]

        strided_list = [org]
        for _ in range(0, self.max_downsample_times - self.cfg.skip_top_scales_num):
            strided_list.append(self.get_bin(strided_list[-1], org.F))

        cached_c = strided_list[-1].C[:, 1:].reshape(-1) // strided_list[-1].tensor_stride[0]
        cached_c_cpu = cached_c.to('cpu', torch.uint16, memory_format=torch.contiguous_format, non_blocking=True)
        cached_c_cdf = self.batch_quantize_pmf_torch(
            (torch.bincount(cached_c.to(torch.int32), minlength=2) / cached_c.numel())[None], False)[0]

        cur_rec = SparseTensor(
            org.F[:strided_list[-1].C.shape[0]],
            coordinate_map_key=strided_list[-1].coordinate_map_key,
            coordinate_manager=strided_list[-1].coordinate_manager)

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
        device = next(self.parameters()).device
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
                    bottom_points_num * 3, device, torch.int32, decode_rounded_min=False
                ).reshape(-1, 3) * 2 ** (self.max_downsample_times - self.cfg.skip_top_scales_num),
                  (1, 0, 0, 0)),
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
            assert cur_rec.tensor_stride[0] == 2
            coord_recon = cur_rec.C[:, None, 1:]
        else:  # recon via OneScaleMultiStepPredictor
            assert top_stride == 2
            coord_recon = top_rec[:, None, 1:]
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


class Block(nn.Module):
    def __init__(self, ch):
        super(Block, self).__init__()
        self.ch = ch
        self.conv = ME.MinkowskiConvolution(ch, ch, 3, 1, 1, bias=True, dimension=3)
        self.act = ME.MinkowskiPReLU()
        self.conv2 = ME.MinkowskiConvolution(ch, ch, 3, 1, 1, bias=True, dimension=3)
        self.act2 = ME.MinkowskiPReLU()

    def forward(self, org: SparseTensor, coordinate_map_key=None):
        x = self.conv(org, coordinate_map_key)
        x = self.act(x)
        x = self.conv2(x)
        x.F.add_(org.F)
        x = self.act2(x)
        return x


class SparseSequential(nn.Sequential):
    def forward(self, x: SparseTensor, coordinate_map_key=None) -> SparseTensor:
        for module in self:
            if coordinate_map_key is not None and isinstance(x, (ME.MinkowskiConvolution, Block)):
                x = module(x, coordinate_map_key)
            else:
                x = module(x)
        return x


def int_to_bytes(x, length, byteorder: Literal['little', 'big'] = 'little', signed=False):
    assert isinstance(x, int)
    return x.to_bytes(length, byteorder=byteorder, signed=signed)


def bytes_to_int(s, byteorder: Literal['little', 'big'] = 'little', signed=False):
    assert isinstance(s, bytes)
    return int.from_bytes(s, byteorder=byteorder, signed=signed)
