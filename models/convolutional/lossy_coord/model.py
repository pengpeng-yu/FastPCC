import io
from typing import List, Union, Tuple, Generator, Optional
import math
import os
import time
import hashlib

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensorQuantizationMode

from lib.utils import Timer
from lib.mpeg_gpcc_utils import gpcc_octree_lossless_geom_encode, gpcc_decode
from lib.torch_utils import MLPBlock, TorchCudaMaxMemoryAllocated
from lib.data_utils import PCData, write_ply_file
from lib.evaluators import PCGCEvaluator
from lib.sparse_conv_layers import GenerativeUpsampleMessage
from lib.entropy_models.continuous_batched import \
    NoisyDeepFactorizedEntropyModel as PriorEM
from lib.entropy_models.hyperprior.noisy_deep_factorized.basic import \
    ScaleNoisyNormalEntropyModel as HyperPriorScaleNoisyNormalEM, \
    NoisyDeepFactorizedEntropyModel as HyperPriorNoisyDeepFactorizedEM
from lib.entropy_models.hyperprior.noisy_deep_factorized.sparse_tensor_specialized import \
    GeoLosslessNoisyDeepFactorizedEntropyModel

from .layers import \
    Encoder, Decoder, \
    HyperEncoder, HyperDecoder, \
    HyperDecoderGenUpsample, HyperDecoderUpsample, EncoderRecurrent
from .model_config import ModelConfig


class PCC(nn.Module):

    def params_divider(self, s: str) -> int:
        if self.cfg.recurrent_part_enabled:
            if s.endswith("aux_param"): return 3
            else:
                if '.em_lossless_based' in s:
                    if 'non_shared_blocks_out_first' in s:
                        return 0
                    elif '.non_shared' in s:
                        return 1
                    else:
                        return 2
                else:
                    return 0
        else:
            if s.endswith("aux_param"): return 1
            else: return 0

    def __init__(self, cfg: ModelConfig):
        super(PCC, self).__init__()
        self.cfg = cfg
        ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
        self.minkowski_algorithm = getattr(ME.MinkowskiAlgorithm, cfg.minkowski_algorithm)
        self.evaluator = PCGCEvaluator(
            cfg.mpeg_pcc_error_command,
            cfg.mpeg_pcc_error_threads
        )
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
                    bn='nn.bn1d' if cfg.use_batch_norm else None,
                    act=cfg.activation
                ) for _ in range(cfg.parameter_fns_mlp_num - 2)),
                MLPBlock(
                    in_channels, out_channels,
                    bn='nn.bn1d' if cfg.use_batch_norm else None,
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
            self.cfg.coord_recon_p2points_weighted_bce,
            *self.basic_block_args,
            loss_type=cfg.coord_recon_loss_type,
            dist_upper_bound=cfg.dist_upper_bound
        )
        if not cfg.recurrent_part_enabled:
            enc_lossl = None
            hyper_dec_coord = None
            hyper_dec_fea = None
            assert len(cfg.skip_encoding_fea) == 0
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
                parameter_fns_factory, cfg.skip_encoding_fea
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
                    prior_bytes_num_bytes=3,
                    coding_ndim=2,
                    num_scales=self.cfg.prior_indexes_range[0],
                    scale_min=0.11,
                    scale_max=64,
                    bottleneck_process=self.cfg.bottleneck_process,
                    quantize_indexes=self.cfg.quantize_indexes,
                    indexes_scaler=self.cfg.prior_indexes_scaler,
                    init_scale=10 / self.cfg.bottleneck_scaler,
                )
            elif self.cfg.hyperprior == 'NoisyDeepFactorized':
                em = HyperPriorNoisyDeepFactorizedEM(
                    hyper_encoder=hyper_encoder,
                    hyper_decoder=hyper_decoder,
                    hyperprior_batch_shape=torch.Size([self.cfg.hyper_compressed_channels]),
                    hyperprior_broadcast_shape_bytes=(3,),
                    prior_bytes_num_bytes=3,
                    coding_ndim=2,
                    index_ranges=self.cfg.prior_indexes_range,
                    parameter_fns_type='transform',
                    parameter_fns_factory=parameter_fns_factory,
                    num_filters=self.cfg.fea_num_filters,
                    bottleneck_process=self.cfg.bottleneck_process,
                    bottleneck_scaler=self.cfg.bottleneck_scaler,
                    quantize_indexes=self.cfg.quantize_indexes,
                    indexes_scaler=self.cfg.prior_indexes_scaler,
                    init_scale=10 / self.cfg.bottleneck_scaler
                )
            else:
                raise NotImplementedError
        return em

    def init_em_lossless_based(
            self, bottom_fea_entropy_model, encoder_geo_lossless,
            hyper_decoder_coord_geo_lossless, hyper_decoder_fea_geo_lossless,
            parameter_fns_factory, skip_encoding_fea
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
            skip_encoding_fea=skip_encoding_fea,
            upper_fea_grad_scaler_for_bits_loss=self.cfg.upper_fea_grad_scaler,
            bottleneck_fea_process=self.cfg.bottleneck_process,
            bottleneck_scaler=self.cfg.bottleneck_scaler,
            quantize_indexes=self.cfg.quantize_indexes,
            indexes_scaler=self.cfg.prior_indexes_scaler,
            init_scale=10 / self.cfg.bottleneck_scaler
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
            sparse_pc = self.get_sparse_pc(pc_data.xyz)
            return self.train_forward(sparse_pc, pc_data.training_step, pc_data.batch_size)
        else:
            assert pc_data.batch_size == 1, 'Only supports batch size == 1 during testing.'
            if isinstance(pc_data.xyz, torch.Tensor):
                sparse_pc = self.get_sparse_pc(pc_data.xyz)
                return self.test_forward(sparse_pc, pc_data)
            else:
                sparse_pc_partitions = self.get_sparse_pc_partitions(pc_data.xyz)
                return self.test_partitions_forward(sparse_pc_partitions, pc_data)

    def get_sparse_pc(self, xyz: torch.Tensor,
                      tensor_stride: int = 1,
                      only_return_coords: bool = False)\
            -> Union[ME.SparseTensor, Tuple[ME.CoordinateMapKey, ME.CoordinateManager]]:
        ME.clear_global_coordinate_manager()
        global_coord_mg = ME.CoordinateManager(
            D=3,
            coordinate_map_type=ME.CoordinateMapType.CUDA if
            xyz.is_cuda
            else ME.CoordinateMapType.CPU,
            minkowski_algorithm=self.minkowski_algorithm
        )
        ME.set_global_coordinate_manager(global_coord_mg)
        if only_return_coords:
            pc_coord_key = global_coord_mg.insert_and_map(xyz, [tensor_stride] * 3)[0]
            return pc_coord_key, global_coord_mg
        else:
            sparse_pc_feature = torch.ones(
                xyz.shape[0], 1,
                dtype=torch.float,
                device=xyz.device
            )
            sparse_pc = ME.SparseTensor(
                features=sparse_pc_feature,
                coordinates=xyz,
                tensor_stride=[tensor_stride] * 3,
                coordinate_manager=global_coord_mg,
                quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            )
            return sparse_pc

    def get_sparse_pc_partitions(self, xyz: List[torch.Tensor]) -> Generator:
        # The first one is supposed to be the original coordinates.
        for idx in range(1, len(xyz)):
            yield self.get_sparse_pc(xyz[idx])

    def train_forward(self, sparse_pc: ME.SparseTensor, training_step: int, batch_size: int):
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
                points_num_list=points_num_list
            )
        )
        loss_dict['coord_recon_loss'] = self.get_coord_recon_loss(
            decoder_message.cached_pred_list,
            decoder_message.cached_target_list,
            decoder_message
        )

        if warmup_forward and self.cfg.linear_warmup:
            warmup_bpp_loss_factor = self.cfg.warmup_bpp_loss_factor - \
                (self.cfg.warmup_bpp_loss_factor - self.cfg.bpp_loss_factor) \
                / self.cfg.warmup_steps * training_step
        else:
            warmup_bpp_loss_factor = self.cfg.warmup_bpp_loss_factor
        for key in loss_dict:
            if key.endswith('bits_loss'):
                if warmup_forward:
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

    def test_forward(self, sparse_pc: ME.SparseTensor, pc_data: PCData):
        with Timer() as encoder_t, TorchCudaMaxMemoryAllocated() as encoder_m:
            compressed_bytes, sparse_tensor_coords = self.compress(sparse_pc)
        del sparse_pc
        ME.clear_global_coordinate_manager()
        torch.cuda.empty_cache()
        with Timer() as decoder_t, TorchCudaMaxMemoryAllocated() as decoder_m:
            coord_recon = self.decompress(compressed_bytes, sparse_tensor_coords)
        ret = self.evaluator.log_batch(
            preds=[coord_recon],
            targets=[pc_data.xyz[:, 1:]],
            compressed_bytes_list=[compressed_bytes],
            pc_data=pc_data,
            extra_info_dicts=[
                {'encoder_elapsed_time': encoder_t.elapsed_time,
                 'encoder_max_cuda_memory_allocated': encoder_m.max_memory_allocated,
                 'decoder_elapsed_time': decoder_t.elapsed_time,
                 'decoder_max_cuda_memory_allocated': decoder_m.max_memory_allocated}
            ]
        )
        return ret

    def test_partitions_forward(self, sparse_pc_partitions: Generator, pc_data: PCData):
        with Timer() as encoder_t, TorchCudaMaxMemoryAllocated() as encoder_m:
            compressed_bytes, sparse_tensor_coords_list = self.compress_partitions(sparse_pc_partitions)
        del sparse_pc_partitions
        ME.clear_global_coordinate_manager()
        torch.cuda.empty_cache()
        with Timer() as decoder_t, TorchCudaMaxMemoryAllocated() as decoder_m:
            coord_recon = self.decompress_partitions(compressed_bytes, sparse_tensor_coords_list)
        ret = self.evaluator.log_batch(
            preds=[coord_recon],
            targets=[pc_data.xyz[0]],
            compressed_bytes_list=[compressed_bytes],
            pc_data=pc_data,
            extra_info_dicts=[
                {'encoder_elapsed_time': encoder_t.elapsed_time,
                 'encoder_max_cuda_memory_allocated': encoder_m.max_memory_allocated,
                 'decoder_elapsed_time': decoder_t.elapsed_time,
                 'decoder_max_cuda_memory_allocated': decoder_m.max_memory_allocated}
            ]
        )
        return ret

    def compress(self, sparse_pc: ME.SparseTensor) -> Tuple[bytes, Optional[torch.Tensor]]:
        strided_fea_list, points_num_list = self.encoder(sparse_pc)
        feature = strided_fea_list[-1]

        if self.cfg.recurrent_part_enabled:
            em_bytes, bottom_fea_recon, fea_recon = self.em_lossless_based.compress(feature, 1)
            assert bottom_fea_recon.C.shape[0] == 1
            sparse_tensor_coords_stride = bottom_fea_recon.tensor_stride[0]
            sparse_tensor_coords = bottom_fea_recon.C
        else:
            em_bytes_list, coding_batch_shape, fea_recon = self.em.compress(feature)
            assert coding_batch_shape == torch.Size([1])
            em_bytes = em_bytes_list[0]
            sparse_tensor_coords = feature.C
            sparse_tensor_coords_stride = feature.tensor_stride[0]

        if sparse_tensor_coords.shape[0] != 1:
            h = hashlib.md5()
            h.update(str(time.time()).encode())
            tmp_file_path = 'tmp-' + h.hexdigest()
            write_ply_file(sparse_tensor_coords[:, 1:] // sparse_tensor_coords_stride, f'{tmp_file_path}.ply')
            gpcc_octree_lossless_geom_encode(
                f'{tmp_file_path}.ply', f'{tmp_file_path}.bin',
                command=self.cfg.mpeg_gpcc_command
            )
            with open(f'{tmp_file_path}.bin', 'rb') as f:
                sparse_tensor_coords_bytes = f.read()
            os.remove(f'{tmp_file_path}.ply')
            os.remove(f'{tmp_file_path}.bin')
            em_bytes = len(sparse_tensor_coords_bytes).to_bytes(3, 'little', signed=False) + \
                sparse_tensor_coords_bytes + em_bytes
        else:
            em_bytes = b''.join(
                [_.to_bytes(1, 'little', signed=False)
                 for _ in (sparse_tensor_coords[0, 1:] // sparse_tensor_coords_stride).tolist()]
            ) + em_bytes
        
        with io.BytesIO() as bs:
            if self.cfg.adaptive_pruning:
                bs.write(b''.join(
                    (_[0].to_bytes(3, 'little', signed=False) for _ in points_num_list)
                ))
            bs.write(int(math.log2(sparse_tensor_coords_stride)).to_bytes(
                     1, 'little', signed=False))
            bs.write(em_bytes)
            compressed_bytes = bs.getvalue()
        return compressed_bytes, sparse_tensor_coords

    def compress_partitions(self, sparse_pc_partitions: Generator) \
            -> Tuple[bytes, List[torch.Tensor]]:
        compressed_bytes_list = []
        sparse_tensor_coords_list = []
        for sparse_pc in sparse_pc_partitions:
            compressed_bytes, sparse_tensor_coords = self.compress(sparse_pc)
            ME.clear_global_coordinate_manager()
            compressed_bytes_list.append(compressed_bytes)
            sparse_tensor_coords_list.append(sparse_tensor_coords)

        # Log bytes of each partition.
        concat_bytes = b''.join((len(s).to_bytes(3, 'little', signed=False) + s
                                 for s in compressed_bytes_list))
        return concat_bytes, sparse_tensor_coords_list

    def decompress(self, compressed_bytes: bytes, sparse_tensor_coords: Optional[torch.Tensor]
                   ) -> torch.Tensor:
        device = next(self.parameters()).device
        with io.BytesIO(compressed_bytes) as bs:
            if self.cfg.adaptive_pruning:
                points_num_list = []
                for idx in range(self.normal_part_coder_num):
                    points_num_list.append([int.from_bytes(bs.read(3), 'little', signed=False)])
            else:
                points_num_list = None
            tensor_stride = 2 ** int.from_bytes(bs.read(1), 'little', signed=False)
            if self.cfg.recurrent_part_enabled:
                sparse_tensor_coords_bytes = bs.read(3)
            else:
                sparse_tensor_coords_bytes_len = int.from_bytes(bs.read(3), 'little', signed=False)
                sparse_tensor_coords_bytes = bs.read(sparse_tensor_coords_bytes_len)
                h = hashlib.md5()
                h.update(str(time.time()).encode())
                tmp_file_path = 'tmp-' + h.hexdigest()
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
                points_num_list=points_num_list
            ))
        coord_recon = decoder_message.fea.C[:, 1:]
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
                ME.clear_global_coordinate_manager()

        coord_recon_concat = torch.cat(coord_recon_list, 0)
        return coord_recon_concat

    def get_coord_recon_loss(
            self, cached_pred_list: List[ME.SparseTensor],
            cached_target_list: List[torch.Tensor], message: GenerativeUpsampleMessage
    ):
        if self.cfg.coord_recon_loss_type == 'BCE':
            preds_num = len(cached_pred_list)
            recon_loss_list = []
            mg = cached_pred_list[-1].coordinate_manager
            device = cached_pred_list[-1].device
            bce_weights = message.bce_weights
            for idx in range(preds_num - 1, -1, -1):
                pred = cached_pred_list[idx]
                target = cached_target_list[idx]
                if idx != preds_num - 1 and bce_weights is not None:
                    kernel_map = mg.kernel_map(
                        pred.coordinate_key, cached_pred_list[idx + 1].coordinate_key,
                        2, kernel_size=2, is_transpose=True, is_pool=True
                    )
                    kernel_map = kernel_map[0][0, :]
                    bce_weights = torch.zeros(
                        pred.shape[0], device=device
                    ).index_add_(0, kernel_map, bce_weights - 1)
                    bce_weights.sigmoid_()
                    bce_weights *= 2

                recon_loss = F.binary_cross_entropy_with_logits(
                    pred.F.squeeze(dim=1),
                    target.type(pred.F.dtype),
                    weight=bce_weights
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

    def train(self, mode: bool = True):
        """
        Use model.train() to reset evaluator.
        """
        if mode is True:
            self.evaluator.reset()
        return super(PCC, self).train(mode=mode)
