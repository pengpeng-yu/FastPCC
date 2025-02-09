import io
from typing import List, Union, Tuple, Optional
import math
import os

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensorQuantizationMode

from lib.utils import Timer
from lib.mpeg_gpcc_utils import gpcc_octree_lossless_geom_encode, gpcc_decode
from lib.torch_utils import TorchCudaMaxMemoryAllocated
from lib.data_utils import PCData, write_ply_file
from lib.evaluators import PCCEvaluator
from lib.entropy_models.continuous_batched import \
    NoisyDeepFactorizedEntropyModel as PriorEM
from lib.entropy_models.hyperprior.noisy_deep_factorized.basic import \
    ScaleNoisyNormalEntropyModel as HyperPriorScaleNoisyNormalEM, \
    NoisyDeepFactorizedEntropyModel as HyperPriorNoisyDeepFactorizedEM

from .geo_lossl_em import GeoLosslessNoisyDeepFactorizedEntropyModel
from .generative_upsample import GenerativeUpsampleMessage
from .layers import MLPBlock, \
    Encoder, Decoder, \
    HyperEncoder, HyperDecoder, \
    HyperDecoderGenUpsample, HyperDecoderUpsample, EncoderRecurrent
from .model_config import ModelConfig


class PCC(nn.Module):

    def params_divider(self, s: str) -> int:
        if self.cfg.recurrent_part_enabled:
            if 'em_lossless_based' in s:
                if 'non_shared_blocks_out_first' in s:
                    return 0
                elif '.non_shared' in s:
                    return 1
                else:
                    return 2
            else:
                return 0
        else:
            return 0

    def __init__(self, cfg: ModelConfig):
        super(PCC, self).__init__()
        self.cfg = cfg
        ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
        self.minkowski_algorithm = getattr(ME.MinkowskiAlgorithm, cfg.minkowski_algorithm)
        self.evaluator = PCCEvaluator()
        self.basic_block_args = (
            cfg.basic_block_type,
            cfg.conv_region_type,
            cfg.basic_block_num,
            cfg.use_batch_norm,
            cfg.activation
        )
        self.hyper_dec_fea_chnls = cfg.compressed_channels * (
            len(cfg.prior_indexes_range)
            if not cfg.hybrid_hyper_decoder_fea
            else len(cfg.prior_indexes_range) + 1
        )
        assert len(cfg.encoder_channels) == len(cfg.decoder_channels) + 1
        self.normal_part_coder_num = len(cfg.decoder_channels)

        def parameter_fns_factory(in_channels, out_channels):
            ret = [
                *(MLPBlock(
                    in_channels, in_channels,
                    bn=cfg.use_batch_norm,
                    act=cfg.activation
                ) for _ in range(cfg.parameter_fns_mlp_num - 2)),
                MLPBlock(
                    in_channels, out_channels,
                    bn=cfg.use_batch_norm,
                    act=cfg.activation
                ),
                nn.Linear(out_channels, out_channels, bias=True)
            ]
            return nn.Sequential(*ret)

        self.encoder = Encoder(
            1,
            (cfg.compressed_channels if not cfg.recurrent_part_enabled
                else cfg.recurrent_part_channels),
            cfg.encoder_channels,
            cfg.first_conv_kernel_size,
            cfg.adaptive_pruning,
            cfg.adaptive_pruning_num_scaler,
            *self.basic_block_args,
            None if not cfg.recurrent_part_enabled else cfg.activation
        )
        self.decoder = Decoder(
            cfg.compressed_channels,
            cfg.decoder_channels,
            *self.basic_block_args,
            loss_type=cfg.coord_recon_loss_type,
            dist_upper_bound=cfg.dist_upper_bound
        )
        if not cfg.recurrent_part_enabled:
            enc_lossl = None
            hyper_dec_coord = None
            hyper_dec_fea = None
        else:
            enc_lossl = self.init_enc_rec()
            hyper_dec_coord = self.init_hyper_dec_gen_up()
            hyper_dec_fea = self.init_hyper_dec_up()

        em = self.init_em(parameter_fns_factory)
        if cfg.recurrent_part_enabled:
            self.em = None
            self.em_lossless_based = self.init_em_lossless_based(
                em, enc_lossl,
                hyper_dec_coord, hyper_dec_fea,
                parameter_fns_factory
            )
        else:
            self.em = em
            self.em_lossless_based = None

    def init_em(self, parameter_fns_factory) -> nn.Module:
        if self.cfg.recurrent_part_enabled:
            assert self.cfg.hyperprior == 'None'

        if self.cfg.hyperprior == 'None':
            em = PriorEM(
                batch_shape=torch.Size([self.cfg.compressed_channels]),
                coding_ndim=2,
                bottleneck_process=self.cfg.bottleneck_process,
                bottleneck_scaler=self.cfg.bottleneck_scaler,
                init_scale=2 / self.cfg.bottleneck_scaler,
                broadcast_shape_bytes=(3,) if not self.cfg.recurrent_part_enabled else (0,),
            )
        else:
            hyper_encoder = HyperEncoder(
                self.cfg.compressed_channels,
                self.cfg.hyper_compressed_channels,
                self.cfg.hyper_encoder_channels,
                *self.basic_block_args
            )
            hyper_decoder = HyperDecoder(
                self.cfg.hyper_compressed_channels,
                self.cfg.compressed_channels * len(self.cfg.prior_indexes_range),
                self.cfg.hyper_decoder_channels,
                *self.basic_block_args
            )

            if self.cfg.hyperprior == 'ScaleNoisyNormal':
                assert len(self.cfg.prior_indexes_range) == 1
                if self.cfg.bottleneck_scaler != 1:
                    raise NotImplementedError
                em = HyperPriorScaleNoisyNormalEM(
                    hyper_encoder=hyper_encoder,
                    hyper_decoder=hyper_decoder,
                    hyperprior_batch_shape=torch.Size([self.cfg.hyper_compressed_channels]),
                    hyperprior_broadcast_shape_bytes=(3,),
                    hyperprior_init_scale=10 / self.cfg.bottleneck_scaler,
                    prior_bytes_num_bytes=3,
                    coding_ndim=2,
                    num_scales=self.cfg.prior_indexes_range[0],
                    scale_min=0.11,
                    scale_max=64,
                    bottleneck_process=self.cfg.bottleneck_process,
                    quantize_indexes=self.cfg.quantize_indexes,
                    indexes_scaler=self.cfg.prior_indexes_scaler,
                )
            elif self.cfg.hyperprior == 'NoisyDeepFactorized':
                em = HyperPriorNoisyDeepFactorizedEM(
                    hyper_encoder=hyper_encoder,
                    hyper_decoder=hyper_decoder,
                    hyperprior_batch_shape=torch.Size([self.cfg.hyper_compressed_channels]),
                    hyperprior_broadcast_shape_bytes=(3,),
                    hyperprior_init_scale=10 / self.cfg.bottleneck_scaler,
                    prior_bytes_num_bytes=3,
                    coding_ndim=2,
                    index_ranges=self.cfg.prior_indexes_range,
                    parameter_fns_type='transform',
                    parameter_fns_factory=parameter_fns_factory,
                    num_filters=self.cfg.fea_num_filters,
                    bottleneck_process=self.cfg.bottleneck_process,
                    bottleneck_scaler=self.cfg.bottleneck_scaler,
                    quantize_indexes=self.cfg.quantize_indexes,
                    indexes_scaler=self.cfg.prior_indexes_scaler
                )
            else:
                raise NotImplementedError
        return em

    def init_em_lossless_based(
            self, bottom_fea_entropy_model, encoder_geo_lossless,
            hyper_decoder_coord_geo_lossless, hyper_decoder_fea_geo_lossless,
            parameter_fns_factory
    ):
        em_lossless_based = GeoLosslessNoisyDeepFactorizedEntropyModel(
            bottom_fea_entropy_model=bottom_fea_entropy_model,
            encoder=encoder_geo_lossless,
            hyper_decoder_coord=hyper_decoder_coord_geo_lossless,
            hyper_decoder_fea=hyper_decoder_fea_geo_lossless,
            hybrid_hyper_decoder_fea=self.cfg.hybrid_hyper_decoder_fea,
            coord_index_ranges=self.cfg.lossless_coord_indexes_range,
            coord_parameter_fns_type='transform',
            coord_parameter_fns_factory=parameter_fns_factory,
            coord_num_filters=(1, 3, 3, 3, 1),
            fea_index_ranges=self.cfg.prior_indexes_range,
            fea_parameter_fns_type='transform',
            fea_parameter_fns_factory=parameter_fns_factory,
            fea_num_filters=self.cfg.lossless_fea_num_filters,
            upper_fea_grad_scaler_for_bits_loss=self.cfg.upper_fea_grad_scaler,
            bottleneck_fea_process=self.cfg.bottleneck_process,
            bottleneck_scaler=self.cfg.bottleneck_scaler,
            quantize_indexes=self.cfg.quantize_indexes,
            indexes_scaler=self.cfg.prior_indexes_scaler
        )
        return em_lossless_based
    
    def init_enc_rec(self):
        enc_rec = EncoderRecurrent(
            self.cfg.recurrent_part_channels,
            self.cfg.compressed_channels,
            *self.basic_block_args
        )
        return enc_rec
    
    def init_hyper_dec_gen_up(self):
        hyper_dec_gen_up = HyperDecoderGenUpsample(
            self.cfg.compressed_channels,
            len(self.cfg.lossless_coord_indexes_range),
            self.cfg.recurrent_part_channels,
            *self.basic_block_args
        )
        return hyper_dec_gen_up

    def init_hyper_dec_up(self):
        hyper_dec_up = HyperDecoderUpsample(
            self.cfg.compressed_channels,
            self.hyper_dec_fea_chnls,
            self.cfg.recurrent_part_channels,
            *self.basic_block_args
        )
        return hyper_dec_up

    def forward(self, pc_data: PCData):
        if self.training:
            return self.train_forward(pc_data.xyz, pc_data.training_step, pc_data.batch_size)
        else:
            assert pc_data.batch_size == 1, 'Only supports batch size == 1 during testing.'
            return self.test_forward(pc_data)

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

    def get_sparse_pc(self, xyz: torch.Tensor,
                      tensor_stride: int = 1,
                      only_return_coords: bool = False)\
            -> Union[ME.SparseTensor, Tuple[ME.CoordinateMapKey, ME.CoordinateManager]]:
        global_coord_mg = self.set_global_cm()
        if only_return_coords:
            pc_coord_key = global_coord_mg.insert_and_map(xyz, [tensor_stride] * 3)[0]
            return pc_coord_key, global_coord_mg
        else:
            sparse_pc_feature = torch.full(
                (xyz.shape[0], 1), fill_value=1,
                dtype=torch.float, device=xyz.device
            )
            sparse_pc = ME.SparseTensor(
                features=sparse_pc_feature,
                coordinates=xyz,
                tensor_stride=[tensor_stride] * 3,
                coordinate_manager=global_coord_mg,
                quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            )
            return sparse_pc

    def train_forward(self, batched_coord: torch.Tensor, training_step: int, batch_size: int):
        sparse_pc = self.get_sparse_pc(batched_coord)
        warmup_forward = training_step < self.cfg.warmup_steps

        strided_fea_list, points_num_list = self.encoder(sparse_pc)
        feature = strided_fea_list[-1]

        if self.cfg.recurrent_part_enabled:
            bottleneck_feature, loss_dict = self.em_lossless_based(feature, batch_size)
        else:
            bottleneck_feature, loss_dict = self.em(feature)

        decoder_message: GenerativeUpsampleMessage = self.decoder(
            GenerativeUpsampleMessage(
                fea=bottleneck_feature,
                target_key=sparse_pc.coordinate_map_key,
                max_stride_lossy_recon=[2 ** len(self.cfg.decoder_channels)] * 3,
                points_num_list=points_num_list
            )
        )
        loss_dict['coord_recon_loss'] = self.get_coord_recon_loss(
            decoder_message.cached_pred_list,
            decoder_message.cached_target_list
        )

        if warmup_forward and self.cfg.linear_warmup:
            warmup_bpp_loss_factor = self.cfg.warmup_bpp_loss_factor - \
                (self.cfg.warmup_bpp_loss_factor - self.cfg.bpp_loss_factor) \
                / self.cfg.warmup_steps * training_step
        else:
            warmup_bpp_loss_factor = self.cfg.warmup_bpp_loss_factor
        for key in loss_dict:
            if key.endswith('bits_loss'):
                if warmup_forward and 'coord' not in key:
                    loss_dict[key] = loss_dict[key] * (
                        warmup_bpp_loss_factor / sparse_pc.shape[0])
                else:
                    loss_dict[key] = loss_dict[key] * (
                        self.cfg.bpp_loss_factor / sparse_pc.shape[0])

        loss_dict['loss'] = sum(loss_dict.values())
        for key in loss_dict:
            if key != 'loss':
                loss_dict[key] = loss_dict[key].item()
        return loss_dict

    def test_forward(self, pc_data: PCData):
        not_part = isinstance(pc_data.xyz, torch.Tensor)
        with Timer() as encoder_t, TorchCudaMaxMemoryAllocated() as encoder_m:
            compressed_bytes, sparse_tensor_coords = self.compress(pc_data.xyz) if not_part else \
                self.compress_partitions(pc_data.xyz)
        ME.clear_global_coordinate_manager()
        torch.cuda.empty_cache()
        with Timer() as decoder_t, TorchCudaMaxMemoryAllocated() as decoder_m:
            coord_recon = self.decompress(compressed_bytes, sparse_tensor_coords) if not_part else \
                self.decompress_partitions(compressed_bytes, sparse_tensor_coords)
        ME.clear_global_coordinate_manager()
        if pc_data.inv_transform is not None:
            inv_trans = pc_data.inv_transform[0].to(coord_recon.device)
            pred_xyz = coord_recon * inv_trans[3]
            pred_xyz += inv_trans[None, :3]
            compressed_bytes = pc_data.inv_transform[0].numpy().astype('<f4').tobytes() + compressed_bytes
        else:
            pred_xyz = coord_recon
        ret = self.evaluator.log(
            pred=pred_xyz,
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

    def compress(self, batched_coord: torch.Tensor) -> Tuple[bytes, Optional[torch.Tensor]]:
        coord_offset = batched_coord[:, 1:].amin(0)
        sparse_pc = self.get_sparse_pc(batched_coord - F.pad(coord_offset, (1, 0)))
        strided_fea_list, points_num_list = self.encoder(sparse_pc)
        feature = strided_fea_list[-1]

        if self.cfg.recurrent_part_enabled:
            em_bytes, bottom_fea_recon, fea_recon = self.em_lossless_based.compress(feature, 1)
            assert bottom_fea_recon.C.shape[0] == 1
            sparse_tensor_coords_stride = bottom_fea_recon.tensor_stride[0]
            sparse_tensor_coords = bottom_fea_recon.C
            assert torch.all(sparse_tensor_coords == 0)
        else:
            em_bytes_list, coding_batch_shape, fea_recon = self.em.compress(feature)
            assert coding_batch_shape == torch.Size([1])
            em_bytes = em_bytes_list[0]
            sparse_tensor_coords = feature.C
            sparse_tensor_coords_stride = feature.tensor_stride[0]

        if not self.cfg.recurrent_part_enabled:
            tmp_file_path = f'tmp-{torch.rand(1).item()}'
            write_ply_file(sparse_tensor_coords[:, 1:] // sparse_tensor_coords_stride, f'{tmp_file_path}.ply')
            gpcc_octree_lossless_geom_encode(
                f'{tmp_file_path}.ply', f'{tmp_file_path}.bin'
            )
            with open(f'{tmp_file_path}.bin', 'rb') as f:
                sparse_tensor_coords_bytes = f.read()
            os.remove(f'{tmp_file_path}.ply')
            os.remove(f'{tmp_file_path}.bin')
            em_bytes = len(sparse_tensor_coords_bytes).to_bytes(3, 'little', signed=False) + \
                sparse_tensor_coords_bytes + em_bytes
        
        with io.BytesIO() as bs:
            for _ in coord_offset.tolist():
                bs.write(_.to_bytes(2, 'little', signed=False))
            if self.cfg.adaptive_pruning:
                bs.write(b''.join(
                    (_[0].to_bytes(3, 'little', signed=False) for _ in points_num_list)
                ))
            bs.write(int(math.log2(sparse_tensor_coords_stride)).to_bytes(
                     1, 'little', signed=False))
            bs.write(em_bytes)
            compressed_bytes = bs.getvalue()
        return compressed_bytes, sparse_tensor_coords

    def compress_partitions(self, batched_coord: List[torch.Tensor]) \
            -> Tuple[bytes, List[torch.Tensor]]:
        compressed_bytes_list = []
        sparse_tensor_coords_list = []
        for idx in range(1, len(batched_coord)):
            compressed_bytes, sparse_tensor_coords = self.compress(batched_coord[idx])
            compressed_bytes_list.append(compressed_bytes)
            sparse_tensor_coords_list.append(sparse_tensor_coords)

        concat_bytes = b''.join((len(s).to_bytes(3, 'little', signed=False) + s
                                 for s in compressed_bytes_list))
        return concat_bytes, sparse_tensor_coords_list

    def decompress(self, compressed_bytes: bytes, sparse_tensor_coords: Optional[torch.Tensor]
                   ) -> torch.Tensor:
        device = next(self.parameters()).device
        with io.BytesIO(compressed_bytes) as bs:
            coord_offset = []
            for _ in range(3):
                coord_offset.append(int.from_bytes(bs.read(2), 'little', signed=False))
            if self.cfg.adaptive_pruning:
                points_num_list = []
                for idx in range(self.normal_part_coder_num):
                    points_num_list.append([int.from_bytes(bs.read(3), 'little', signed=False)])
            else:
                points_num_list = None
            tensor_stride = 2 ** int.from_bytes(bs.read(1), 'little', signed=False)
            if not self.cfg.recurrent_part_enabled:
                sparse_tensor_coords_bytes_len = int.from_bytes(bs.read(3), 'little', signed=False)
                sparse_tensor_coords_bytes = bs.read(sparse_tensor_coords_bytes_len)
                tmp_file_path = f'tmp-{torch.rand(1).item()}'
                with open(f'{tmp_file_path}.bin', 'wb') as f:
                    f.write(sparse_tensor_coords_bytes)
                gpcc_decode(f'{tmp_file_path}.bin', f'{tmp_file_path}.ply')
                sparse_tensor_coords_ = torch.from_numpy(np.asarray(
                    o3d.io.read_point_cloud(f'{tmp_file_path}.ply').points
                ) * tensor_stride).cuda()
                os.remove(f'{tmp_file_path}.ply')
                os.remove(f'{tmp_file_path}.bin')
            em_bytes = bs.read()

        if self.cfg.recurrent_part_enabled:
            fea_recon = self.em_lossless_based.decompress(
                em_bytes,
                self.get_sparse_pc(
                    sparse_tensor_coords,
                    tensor_stride=tensor_stride,
                    only_return_coords=True
                ))
        else:
            fea_recon = self.em.decompress(
                [em_bytes], torch.Size([1]), device,
                sparse_tensor_coords_tuple=self.get_sparse_pc(
                    sparse_tensor_coords,
                    tensor_stride=2 ** self.normal_part_coder_num,
                    only_return_coords=True
                ))

        decoder_message = self.decoder(
            GenerativeUpsampleMessage(
                fea=fea_recon,
                max_stride_lossy_recon=[2 ** len(self.cfg.decoder_channels)] * 3,
                points_num_list=points_num_list
            ))
        coord_recon = decoder_message.fea.C[:, 1:]
        coord_recon += torch.tensor(coord_offset, dtype=torch.int32, device=coord_recon.device)
        return coord_recon

    def decompress_partitions(self, concat_bytes: bytes,
                              sparse_tensor_coords_list: List[Optional[torch.Tensor]]
                              ) -> torch.Tensor:
        coord_recon_list = []
        concat_bytes_len = len(concat_bytes)

        with io.BytesIO(concat_bytes) as bs:
            while bs.tell() != concat_bytes_len:
                length = int.from_bytes(bs.read(3), 'little', signed=False)
                coord_recon = self.decompress(
                    bs.read(length), sparse_tensor_coords_list.pop(0)
                )
                coord_recon_list.append(coord_recon)

        coord_recon_concat = torch.cat(coord_recon_list, 0)
        return coord_recon_concat

    def get_coord_recon_loss(
            self, cached_pred_list: List[ME.SparseTensor],
            cached_target_list: List[torch.Tensor]
    ):
        if self.cfg.coord_recon_loss_type == 'BCE':
            preds_num = len(cached_pred_list)
            recon_loss_list = []
            for idx in range(preds_num - 1, -1, -1):
                pred = cached_pred_list[idx]
                target = cached_target_list[idx]
                recon_loss = F.binary_cross_entropy_with_logits(
                    pred.F.squeeze(dim=1),
                    target.type(pred.F.dtype)
                )
                recon_loss_list.append(recon_loss)

        elif self.cfg.coord_recon_loss_type == 'Dist':
            recon_loss_list = [F.smooth_l1_loss(
                pred.F.squeeze(dim=1),
                target.type(pred.F.dtype)
            ) for pred, target in zip(cached_pred_list, cached_target_list)]

        else:
            raise NotImplementedError

        recon_loss = sum(recon_loss_list) / len(recon_loss_list)
        factor = self.cfg.coord_recon_loss_factor
        if factor != 1:
            recon_loss *= factor
        return recon_loss
