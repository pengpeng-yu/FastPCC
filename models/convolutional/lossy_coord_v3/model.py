import io
from typing import List, Union, Tuple, Optional, Literal
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse import SparseTensor
import torchsparse.nn as spnn
import torchsparse.nn.functional as SF
from torchsparse.utils.tensor_cache import TensorCache

from lib.utils import Timer
from lib.torch_utils import TorchCudaMaxMemoryAllocated
from lib.data_utils import PCData
from lib.evaluators import PCCEvaluator
from lib.morton_code import morton_encode_magicbits
from lib.entropy_models.distributions.deep_factorized import DeepFactorized
from lib.entropy_models.distributions.uniform_noise import NoisyDeepFactorized
from .model_config import Config
from .rans_coder import RansEncoder, RansDecoder


log2_e = math.log2(math.e)


class OneScalePredictor(nn.Module):
    def __init__(self, channels, num_latents=0, if_pred_oct_lossl=True, if_upsample=True, allow_single_ch=False,
                 coord_recon_loss_factor=None, max_lossy_stride=None):
        super(OneScalePredictor, self).__init__()
        self.compressed_channels = 1
        if allow_single_ch is True:
            self.dec_init = spnn.Conv3d(1, channels, 3, 1, 1, bias=True)
        self.dec = Block(channels)

        self.transforms = nn.ModuleList()
        self.num_latents = num_latents
        for idx in range(num_latents):
            transforms = nn.ModuleList((
                SparseSequential(
                    nn.Linear(channels, channels, bias=True), nn.PReLU()),
                SparseSequential(
                    nn.Linear(channels * 2, channels, bias=True), nn.PReLU(),
                    spnn.Conv3d(channels, channels, 3, 1, 1, bias=True), nn.PReLU(),
                    spnn.Conv3d(channels, self.compressed_channels, 3, 1, 1, bias=True)),
                SparseSequential(
                    nn.Linear(self.compressed_channels, channels, bias=True), nn.PReLU()),
                SparseSequential(
                    nn.Linear(channels * 2, channels, bias=True), nn.PReLU(),
                    Block(channels))
            ))
            self.transforms.append(transforms)

        self.if_pred_oct_lossl = if_pred_oct_lossl
        self.if_upsample = if_upsample  # Only False for stride2->stride1.
        self.coord_recon_loss_factor = coord_recon_loss_factor
        self.max_lossy_stride = max_lossy_stride
        if self.if_pred_oct_lossl:
            self.pred = SparseSequential(
                Block(channels),
                nn.Linear(channels, 255, bias=True))
        else:
            assert max_lossy_stride is not None
            self.pred = SparseSequential(
                Block(channels),
                spnn.Conv3d(channels, 8, 3, 1, 1, bias=True))
        if self.if_upsample:
            self.upsample = SparseSequential(
                nn.Linear(channels + 8, channels, bias=True), nn.PReLU(),
                Block(channels),
                nn.Linear(channels, channels * 8, bias=True))
        else:
            self.upsample = None

    def forward(self, cur_rec: SparseTensor, cur_ref: SparseTensor, up_ref: SparseTensor, cur_bin: torch.FloatTensor,
                device, points_num: List[int], em, bin2oct_kernel, unfold_kernel,
                warmup: bool):
        if not self.if_upsample: assert up_ref.stride[0] == 1
        batch_size = len(points_num)
        divider = torch.searchsorted(
            cur_rec.C[:, 0].contiguous(), torch.arange(batch_size + 1, device=device, dtype=torch.int32))
        scattered_points_num = torch.empty((cur_rec.C.shape[0],), device=device, dtype=torch.float)
        for b in range(batch_size):
            scattered_points_num[divider[b]: divider[b + 1]] = points_num[b]

        if cur_rec.F.shape[1] == 1:
            cur_rec = self.dec_init(cur_rec)
        cur_rec = self.dec(cur_rec)

        fea_loss_list = []
        for transform0, transform1, transform2, dec in self.transforms:
            cur_ref_ = transform0(cur_ref)
            cur_ref_.F = torch.cat((cur_ref_.F, cur_rec.F), 1)
            cur_ref_ = transform1(cur_ref_)
            noisy_f, fea_loss = em(cur_ref_.F)
            fea_loss = (fea_loss / scattered_points_num[:, None]).sum() * ((0.01 if warmup else 1) / batch_size)
            fea_loss_list.append(fea_loss)
            cur_ref_.F = noisy_f
            cur_rec.F = torch.cat((cur_rec.F, transform2(cur_ref_).F), 1)
            cur_rec = dec(cur_rec)

        cur_pred = self.pred(cur_rec).F
        if self.if_pred_oct_lossl:
            cur_oct = (cur_bin.to(torch.uint8) << bin2oct_kernel).sum(1, dtype=torch.int64).add_(-1)
            cur_geo_loss = (F.cross_entropy(cur_pred, cur_oct, reduction='none')
                            / scattered_points_num).sum() * (log2_e / batch_size)
            if self.if_upsample:
                pred_bin = cur_bin
                new_c = up_ref.C

        else:
            if up_ref.stride[0] == 1:
                up_scattered_points_num = scattered_points_num
                up_points_num = points_num
            else:
                up_scattered_points_num = torch.empty((cur_rec.C.shape[0],), device=device, dtype=torch.float)
                up_points_num = torch.diff(torch.searchsorted(
                    up_ref.C[:, 0].contiguous(), torch.arange(batch_size + 1, device=device, dtype=torch.int32)))
                for b in range(batch_size):
                    up_scattered_points_num[divider[b]: divider[b + 1]] = up_points_num[b]
            cur_geo_loss = (F.binary_cross_entropy_with_logits(cur_pred, cur_bin, reduction='none').sum(1)
                            / up_scattered_points_num).sum() * (self.coord_recon_loss_factor * log2_e / batch_size)

            if self.if_upsample:
                mask_list = []
                for b in range(batch_size):
                    sample = cur_pred[divider[b]: divider[b + 1]]
                    sample_ = sample.reshape(-1, 8)
                    mask = sample_ == sample_.amax(1, keepdim=True)
                    mask = mask.reshape(-1, 8)
                    kth_v = torch.kthvalue(
                        sample.reshape(-1), sample.shape[0] * 8 - up_points_num[b]
                    ).values
                    mask |= sample > kth_v
                    mask_list.append(mask)
                pred_bin = torch.cat(mask_list, 0)
                pred_bin |= cur_bin.bool()
                new_c = cur_rec.C.clone()[:, None]
                new_c[..., 1:] <<= 1
                new_c = (new_c + unfold_kernel)[pred_bin]

        if self.if_upsample:
            cur_rec.F = torch.cat((cur_rec.F, pred_bin), 1)
            cur_rec = self.upsample(cur_rec)
            cur_rec = SparseTensor(
                cur_rec.F.reshape(cur_rec.F.shape[0], 8, cur_rec.F.shape[1] // 8)[pred_bin.bool()],
                new_c,
                tuple(_ // 2 for _ in cur_rec.stride))
            if self.if_pred_oct_lossl:
                cur_rec._caches = cur_ref._caches
            else:
                pass  # Can't reuse caches in lossy cases.
        else:
            cur_rec = None
        return cur_rec, fea_loss_list, cur_geo_loss

    def compress(self, cur_rec: SparseTensor, cur_ref: SparseTensor, up_ref: SparseTensor, cur_bin: torch.FloatTensor,
                 bin2oct_kernel, if_upsample):
        if cur_rec.F.shape[1] == 1:
            cur_rec = self.dec_init(cur_rec)
        cur_rec = self.dec(cur_rec)

        rounded_f_list = []
        for transform0, transform1, transform2, dec in self.transforms:
            cur_ref_ = transform0(cur_ref)
            cur_ref_.F = torch.cat((cur_ref_.F, cur_rec.F), 1)
            cur_ref_ = transform1(cur_ref_)
            cur_ref_.F.round_()
            rounded_f = cur_ref_.F
            rounded_f_list.append(rounded_f)
            cur_rec.F = torch.cat((cur_rec.F, transform2(cur_ref_).F), 1)
            cur_rec = dec(cur_rec)

        if self.if_pred_oct_lossl:
            cur_pred = self.pred(cur_rec).F
            cur_oct = (cur_bin.to(torch.uint8) << bin2oct_kernel).sum(1, dtype=torch.int16).add_(-1)
            if if_upsample:
                cur_rec.F = torch.cat((cur_rec.F, cur_bin), 1)
                cur_rec = self.upsample(cur_rec)
                cur_rec = SparseTensor(
                    cur_rec.F.reshape(cur_rec.F.shape[0], 8, cur_rec.F.shape[1] // 8)[cur_bin.bool()],
                    up_ref.C,
                    tuple(_ // 2 for _ in cur_rec.stride))
                cur_rec._caches = cur_ref._caches

        else:
            cur_rec = cur_pred = cur_oct = None

        return cur_rec, rounded_f_list, cur_pred, cur_oct

    def decompress(self, cur_rec: SparseTensor, cached_points_num, device, bin2oct_kernel, unfold_kernel,
                   rans_decode_fea, rans_decode_oct, if_upsample):
        if cur_rec.F.shape[1] == 1:
            cur_rec = self.dec_init(cur_rec)
        cur_rec = self.dec(cur_rec)

        latent = SparseTensor(None, cur_rec.C, cur_rec.stride, cur_rec.spatial_range)
        latent._caches = cur_rec._caches
        for transform0, transform1, transform2, dec in self.transforms:
            rounded_f = rans_decode_fea(
                cur_rec.C.shape[0] * self.compressed_channels, device, torch.float
            ).reshape(cur_rec.C.shape[0], -1)
            latent.F = rounded_f
            cur_rec.F = torch.cat((cur_rec.F, transform2(latent).F), 1)
            cur_rec = dec(cur_rec)

        cur_pred = self.pred(cur_rec)
        if self.if_pred_oct_lossl:
            cur_oct = rans_decode_oct(cur_pred.F, cur_rec.C.shape[0], device, torch.int16)
            cur_bin = ((cur_oct[:, None] + 1) >> bin2oct_kernel).bitwise_and_(1).bool()
        else:
            cur_pred_f = cur_pred.F
            if cur_rec.stride[0] == 2:
                if self.max_lossy_stride != 2:
                    kth_v = torch.kthvalue(
                        cur_pred_f.reshape(-1), cur_pred_f.shape[0] * 8 - cached_points_num.pop()
                    ).values
                    mask = cur_pred_f > kth_v
                else:
                    cur_pred_f_ = cur_pred_f.reshape(-1, 8)
                    mask = cur_pred_f_ == cur_pred_f_.amax(1, keepdim=True)
                    mask = mask.reshape(-1, 8)
                    non_local_max = cur_pred_f[~mask]
                    kth_v = torch.kthvalue(
                        non_local_max, non_local_max.shape[0] - cached_points_num.pop() + cur_pred_f.shape[0]
                    ).values
                    mask |= cur_pred_f > kth_v
            else:
                cur_pred_f_ = cur_pred_f.reshape(-1, 8)
                mask = cur_pred_f_ == cur_pred_f_.amax(1, keepdim=True)
                mask = mask.reshape(-1, 8)
                kth_v = torch.kthvalue(
                    cur_pred_f.reshape(-1), cur_pred_f.shape[0] * 8 - cached_points_num.pop()
                ).values
                mask |= cur_pred_f > kth_v
            cur_bin = mask

        if if_upsample:
            cur_rec.F = torch.cat((cur_rec.F, cur_bin), 1)
            cur_rec = self.upsample(cur_rec)
            new_c = cur_rec.C[:, None]
            new_c[..., 1:] <<= 1
            cur_rec = SparseTensor(
                cur_rec.F.reshape(cur_rec.F.shape[0], 8, cur_rec.F.shape[1] // 8)[cur_bin],
                (new_c + unfold_kernel)[cur_bin],
                tuple(_ // 2 for _ in cur_rec.stride))
            return cur_rec
        else:
            new_c = cur_rec.C[:, None, 1:]
            new_c <<= 1
            coord_recon = (new_c + unfold_kernel[:, :, 1:])[cur_bin]
            return coord_recon


class Fold(nn.Module):
    def __init__(self):
        super(Fold, self).__init__()
        self.register_buffer('kernel', torch.empty(8, 1, 1 * 8, dtype=torch.float), persistent=False)
        with torch.no_grad():
            self.kernel.reshape(8, 8)[...] = torch.eye(8)

    def forward(self, sp: SparseTensor):
        return SF.conv3d(
            sp,
            weight=self.kernel,
            kernel_size=(2, 2, 2),
            bias=None,
            stride=(2, 2, 2),
            padding=(0, 0, 0),
            dilation=1,
            transposed=False,
            generative=False,
            config=None,
            training=False,
        )


tmp_custom_coord: Optional[torch.Tensor] = None


# >>>> Monkey Patching
def custom_spdownsample(
        _coords: torch.Tensor,
        stride: Union[int, Tuple[int, ...]] = 2,
        kernel_size: Union[int, Tuple[int, ...]] = 2,
        padding: torch.Tensor = 0,
        spatial_range: Optional[Tuple[int]] = None,
        downsample_mode: str = "spconv",
) -> torch.Tensor:
    assert downsample_mode in ["spconv", "minkowski"], downsample_mode
    assert (padding == 0).all(), padding
    global tmp_custom_coord
    if tmp_custom_coord is not None:
        return tmp_custom_coord
    else:
        assert all((_ == 2 for _ in kernel_size)), kernel_size
        assert all((_ == 2 for _ in stride)), stride
        coords = _coords.clone()
        coords[:, 1:] >>= 1
        coords = torch.unique_consecutive(coords, dim=0)
        return coords


SF.spdownsample = custom_spdownsample
# Monkey Patching <<<<


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super(Model, self).__init__()
        conv_config = SF.conv_config.get_default_conv_config(conv_mode=SF.get_conv_mode())
        conv_config['dataflow'] = getattr(SF.conv_config.Dataflow, cfg.torchsparse_dataflow)  # tune?
        conv_config.kmap_mode = 'hashmap'
        SF.conv_config.set_global_conv_config(conv_config)

        self.cfg = cfg
        self.evaluator = PCCEvaluator()

        self.max_downsample_times = int(np.log2(cfg.max_stride))
        assert self.max_downsample_times > len(cfg.num_latents)
        assert len(cfg.num_latents) == len(cfg.lossl_geo_upsample)
        for idx, i in enumerate(cfg.lossl_geo_upsample):
            if i == 1:
                break
        else:
            idx += 1
        assert all(_ == 1 for _ in cfg.lossl_geo_upsample[idx:])
        assert all(_ == 0 for _ in cfg.num_latents[:max(idx-1, 0)])
        self.max_lossy_stride = 2 ** idx

        self.blocks_enc = nn.ModuleList()
        for idx, _ in enumerate(cfg.num_latents):
            if all(_ == 0 for _ in cfg.num_latents[idx:]): break
            if idx == 0:
                block_enc = Fold()
            elif idx == 1:
                block_enc = SparseSequential(
                    spnn.Conv3d(8, cfg.channels, 3, 1, bias=True), nn.PReLU(),
                    spnn.Conv3d(cfg.channels, cfg.channels, 2, 2, bias=True),
                    Block(cfg.channels))
            else:
                block_enc = SparseSequential(
                    spnn.Conv3d(cfg.channels, cfg.channels, 2, 2, bias=True), Block(cfg.channels))
            self.blocks_enc.append(block_enc)

        self.block_dec_recurrent = OneScalePredictor(cfg.channels, 0, True, True, True)

        self.blocks_dec = nn.ModuleList()
        for idx, (a, b) in enumerate(zip(cfg.num_latents, cfg.lossl_geo_upsample)):
            self.blocks_dec.append(
                OneScalePredictor(
                    cfg.channels,
                    a, bool(b), idx != 0,
                    coord_recon_loss_factor=cfg.coord_recon_loss_factor,
                    max_lossy_stride=self.max_lossy_stride))

        if len(self.blocks_enc) != 0:
            self.em = EntropyModel(batch_shape=torch.Size([1]), init_scale=10,)
        else:
            self.em = None

        self.register_buffer('fold2bin_kernel', torch.empty(8, 1, 1 * 8, dtype=torch.float), persistent=False)
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
    def get_bin(self, sp, ones_feats):
        sp_f = sp.F
        sp.F = ones_feats[:sp.C.shape[0]]
        ret = SF.conv3d(
            sp,
            weight=self.fold2bin_kernel,
            kernel_size=(2, 2, 2),
            bias=None,
            stride=(2, 2, 2),
            padding=(0, 0, 0),
            dilation=1,
            transposed=False,
            generative=False,
            config=None,
            training=False,
        )
        sp.F = sp_f
        # assert (ret.F == ret.F.round()).all()
        return ret

    def forward(self, pc_data: PCData):
        if self.training:
            return self.train_forward(pc_data.xyz, pc_data.points_num, pc_data.training_step)
        else:
            assert pc_data.batch_size == 1, 'Only supports batch size == 1 during testing.'
            return self.test_forward(pc_data)

    @staticmethod
    def get_init_pc(xyz: torch.Tensor, stride: int = 1) -> SparseTensor:
        # Input coordinates are assumed to be Morton-sorted with unique points.
        sparse_pc_feature = torch.ones((xyz.shape[0], 1), dtype=torch.float, device=xyz.device)
        sparse_pc = SparseTensor(sparse_pc_feature, xyz, (stride,) * 3)
        return sparse_pc

    def train_forward(self, xyz: torch.Tensor, points_num: List[int], training_step: int):
        self.warmup_forward = training_step < self.cfg.warmup_steps
        org = self.get_init_pc(xyz)
        device = org.F.device

        strided_list = [org]
        for block_enc in self.blocks_enc:
            strided_list.append(block_enc(strided_list[-1]))

        for _ in range(len(self.blocks_enc), self.max_downsample_times):
            strided_list.append(self.get_bin(strided_list[-1], org.F))

        cur_rec = SparseTensor(
            org.F[:strided_list[-1].C.shape[0]], strided_list[-1].C,
            (2 ** self.max_downsample_times,) * 3)
        cur_rec._caches = org._caches

        global tmp_custom_coord
        loss_dict = {}
        for idx in range(self.max_downsample_times, 0, -1):
            if idx > len(self.blocks_dec):
                block_dec = self.block_dec_recurrent
            else:
                block_dec = self.blocks_dec[idx - 1]
            if (idx == 1 and self.cfg.lossl_geo_upsample[idx]) or idx > len(self.blocks_enc):
                cur_bin = strided_list[idx].F
            else:
                if cur_rec._caches is not org._caches:
                    tmp_custom_coord = cur_rec.C
                    strided_list[idx - 1]._caches = TensorCache()
                cur_bin = self.get_bin(strided_list[idx - 1], org.F).F
                tmp_custom_coord = None
            cur_rec, fea_loss_list, geo_bpp_loss = block_dec(
                cur_rec, strided_list[idx], strided_list[idx - 1],
                cur_bin, device, points_num, self.em, self.bin2oct_kernel, self.unfold_kernel,
                warmup=training_step < self.cfg.warmup_steps)
            loss_dict[f'stride{2 ** idx}_geo_loss'] = geo_bpp_loss
            for i, fea_loss in enumerate(fea_loss_list):
                loss_dict[f'stride{2 ** idx}_fea{i}_loss'] = fea_loss

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
        xyz = xyz[torch.argsort(morton_encode_magicbits(xyz[:, 1:], inverse=True))]
        org = self.get_init_pc(xyz, 1)

        blocks_enc = self.blocks_enc[self.cfg.skip_top_scales_num:]
        blocks_dec = self.blocks_dec[self.cfg.skip_top_scales_num:]

        strided_list = [org]
        for block_enc in blocks_enc:
            strided_list.append(block_enc(strided_list[-1]))

        for _ in range(len(blocks_enc), self.max_downsample_times - self.cfg.skip_top_scales_num):
            strided_list.append(self.get_bin(strided_list[-1], org.F))

        cached_points_num = []

        cached_c = strided_list[-1].C[:, 1:].reshape(-1)
        cached_c_cpu = cached_c.to('cpu', torch.uint16, memory_format=torch.contiguous_format, non_blocking=True)
        cached_c_cdf = self.batch_quantize_pmf_torch(
            (torch.bincount(cached_c.to(torch.int32), minlength=2) / cached_c.numel())[None], False)[0]

        for idx, _ in enumerate(self.cfg.lossl_geo_upsample):
            if _ == 1: break
            cached_points_num.append(strided_list[idx].C.shape[0])

        cur_rec = SparseTensor(
            org.F[:strided_list[-1].C.shape[0]], strided_list[-1].C,
            (2 ** (self.max_downsample_times - self.cfg.skip_top_scales_num),) * 3)
        cur_rec._caches = org._caches

        global tmp_custom_coord
        cached_list = []
        for idx in range(self.max_downsample_times - self.cfg.skip_top_scales_num, 0, -1):
            if idx > len(blocks_dec):
                block_dec = self.block_dec_recurrent
            else:
                block_dec = blocks_dec[idx - 1]
            if (idx == 1 and self.cfg.lossl_geo_upsample[idx]) or idx > len(blocks_enc):
                cur_bin = strided_list[idx].F
            else:
                if cur_rec._caches is not org._caches:
                    tmp_custom_coord = cur_rec.C
                    strided_list[idx - 1]._caches = TensorCache()
                cur_bin = self.get_bin(strided_list[idx - 1], org.F).F
                tmp_custom_coord = None
            cur_rec, rounded_f_list, cur_pred, cur_oct = block_dec.compress(
                cur_rec, strided_list[idx], strided_list[idx - 1], cur_bin, self.bin2oct_kernel, if_upsample=idx != 1)
            strided_list.pop()
            if cur_rec is None:
                break
            else:
                for i, rounded_f in enumerate(rounded_f_list):
                    rounded_f_min = -rounded_f.min()
                    rounded_f += rounded_f_min
                    rounded_f = rounded_f.reshape(-1)
                    quantized_cdf = self.batch_quantize_pmf_torch(
                        (torch.bincount(rounded_f.to(torch.int32), minlength=2) / rounded_f.numel())[None], False)[0]
                    rounded_f = rounded_f.to('cpu', torch.uint16, non_blocking=True)
                    rounded_f_min = rounded_f_min.to('cpu', torch.uint16, non_blocking=True)
                    rounded_f_list[i] = (quantized_cdf, rounded_f, rounded_f_min)
                cached_list.append((
                    rounded_f_list,
                    self.batch_quantize_pmf_torch(cur_pred),
                    cur_oct.to('cpu', torch.uint16, non_blocking=True),
                ))

        torch.cuda.synchronize()
        while cached_list:
            rounded_f_list, cur_pred, cur_oct = cached_list.pop()
            self.rans_encode_oct(cur_pred, cur_oct)
            while rounded_f_list:
                self.rans_encode_fea(*rounded_f_list.pop())
        self.rans_encode_fea(cached_c_cdf, cached_c_cpu)

        with io.BytesIO() as bs:
            for _ in coord_offset.tolist():
                bs.write(int_to_bytes(_, 2))
            bs.write(int_to_bytes(cached_c.shape[0] // 3, 2))
            for _ in cached_points_num:
                bs.write(int_to_bytes(_, 3))
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
        cached_points_num = []
        with io.BytesIO(compressed_bytes) as bs:
            for _ in range(3):
                coord_offset.append(bytes_to_int(bs.read(2)))
            bottom_points_num = bytes_to_int(bs.read(2))
            for idx, _ in enumerate(self.cfg.lossl_geo_upsample):
                if _ == 1: break
                cached_points_num.append(bytes_to_int(bs.read(3)))
            rans_comp_bytes = bs.read()  # Keep this reference
            self.rans_decoder.flush(rans_comp_bytes)
        coord_offset = torch.tensor(coord_offset, device=device, dtype=torch.int32)[None]

        cur_rec = self.get_init_pc(
            F.pad(self.rans_decode_fea(
                bottom_points_num * 3, device, torch.int32, decode_rounded_min=False).reshape(-1, 3), (1, 0, 0, 0)),
            2 ** (self.max_downsample_times - self.cfg.skip_top_scales_num))

        blocks_dec = self.blocks_dec[self.cfg.skip_top_scales_num:]
        for idx in range(self.max_downsample_times - self.cfg.skip_top_scales_num, 0, -1):
            if idx > len(blocks_dec):
                block_dec = self.block_dec_recurrent
            else:
                block_dec = blocks_dec[idx - 1]
            cur_rec = block_dec.decompress(
                cur_rec, cached_points_num, device, self.bin2oct_kernel, self.unfold_kernel,
                self.rans_decode_fea, self.rans_decode_oct, if_upsample=idx != 1)

        cur_rec += coord_offset
        return cur_rec

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
        self.conv = spnn.Conv3d(ch, ch, 3, 1, 1, bias=True)
        self.act = nn.PReLU()
        self.conv2 = spnn.Conv3d(ch, ch, 3, 1, 1, bias=True)
        self.act2 = nn.PReLU()

    def forward(self, org: SparseTensor):
        x = self.conv(org)
        x.F = self.act(x.F)
        x = self.conv2(x)
        x.F.add_(org.F)
        x.F = self.act2(x.F)
        return x


class SparseSequential(nn.Sequential):
    def forward(self, input: SparseTensor) -> SparseTensor:
        x = SparseTensor(input.F, input.C, input.stride, input.spatial_range)
        x._caches = input._caches
        for module in self:
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.ReLU, nn.LeakyReLU, nn.PReLU)):
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


class EntropyModel(torch.nn.Module):
    def __init__(self,
                 batch_shape: torch.Size = torch.Size([1]),
                 init_scale: float = 10,
                 num_filters: Tuple[int, ...] = (1, 3, 3, 3, 3, 1),):
        super(EntropyModel, self).__init__()
        prior_weights, prior_biases, prior_factors = \
            DeepFactorized.make_parameters(
                batch_numel=batch_shape.numel(),
                init_scale=init_scale,
                num_filters=num_filters)
        self.prior = NoisyDeepFactorized(
            batch_shape=batch_shape,
            weights=prior_weights,
            biases=prior_biases,
            factors=prior_factors)
        self.prior_weights, self.prior_biases, self.prior_factors = prior_weights, prior_biases, prior_factors

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + torch.empty_like(x).uniform_(-0.5, 0.5)
        log_probs = self.prior.log_prob(x)
        return x, log_probs * -log2_e
