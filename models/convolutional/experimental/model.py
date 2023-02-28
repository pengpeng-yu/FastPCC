import io
from functools import partial
from typing import List, Union, Tuple, Generator, Optional
import math
import os
import time
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensorQuantizationMode
try:
    from pytorch3d.ops.knn import knn_points, knn_gather
except ImportError: pass

from lib.utils import Timer
from lib.metrics.misc import gen_rgb_to_yuvbt709_param
from lib.mpeg_gpcc_utils import gpcc_octree_lossless_geom_encode, gpcc_decode
from lib.torch_utils import MLPBlock, TorchCudaMaxMemoryAllocated, concat_loss_dicts
from lib.data_utils import PCData, write_ply_file
from lib.evaluators import PCGCEvaluator
from lib.entropy_models.continuous_batched import \
    NoisyDeepFactorizedEntropyModel as PriorEM
from lib.entropy_models.continuous_indexed import \
    ContinuousNoisyDeepFactorizedIndexedEntropyModel as IndexedNoisyDeepFactorizedEM
from lib.entropy_models.hyperprior.noisy_deep_factorized.basic import \
    ScaleNoisyNormalEntropyModel as HyperPriorScaleNoisyNormalEM, \
    NoisyDeepFactorizedEntropyModel as HyperPriorNoisyDeepFactorizedEM
from .geo_lossl_em import GeoLosslessNoisyDeepFactorizedEntropyModel

from .generative_upsample import GenerativeUpsampleMessage
from .layers import \
    Encoder, Decoder, \
    HyperEncoder, HyperDecoder, \
    HyperDecoderGenUpsample, HyperDecoderUpsample, \
    EncoderRecurrent, EncoderPartiallyRecurrent, \
    HyperDecoderGenUpsamplePartiallyRecurrent, \
    HyperDecoderUpsamplePartiallyRecurrent
from .model_config import ModelConfig


class PCC(nn.Module):

    @staticmethod
    def params_divider(s: str) -> int:
        if s.endswith("aux_param"): return 2
        else:
            if 'em_lossless_based' in s:
                if 'encoder' in s or 'decoder' in s:
                    return 0
                else:
                    return 1
            elif 'em' in s:
                return 1
            else:
                return 0

    def __init__(self, cfg: ModelConfig):
        super(PCC, self).__init__()
        self.cfg = cfg
        ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
        self.minkowski_algorithm = getattr(ME.MinkowskiAlgorithm, cfg.minkowski_algorithm)
        self.evaluator = PCGCEvaluator(
            cfg.mpeg_pcc_error_command,
            cfg.mpeg_pcc_error_processes
        )
        self.basic_block_args = (
            cfg.basic_block_type,
            cfg.conv_region_type,
            cfg.basic_block_num,
            cfg.use_batch_norm,
            cfg.activation
        )
        if cfg.input_feature_type == 'Occupation':
            self.input_feature_channels = 1
        elif cfg.input_feature_type == 'Color':
            self.input_feature_channels = 3
            if not cfg.lossless_color_enabled:
                rgb_to_yuvbt709_param = gen_rgb_to_yuvbt709_param()
                self.register_buffer('rgb_to_yuvbt709_weight', rgb_to_yuvbt709_param[0], False)
                self.register_buffer('rgb_to_yuvbt709_bias', rgb_to_yuvbt709_param[1], False)
        else:
            raise NotImplementedError
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

        encoder = Encoder(
            self.input_feature_channels,
            (cfg.compressed_channels if not cfg.recurrent_part_enabled
                else cfg.recurrent_part_channels),
            cfg.encoder_channels,
            cfg.first_conv_kernel_size,
            cfg.adaptive_pruning,
            cfg.adaptive_pruning_num_scaler,
            cfg.compressed_channels if cfg.lossless_coord_enabled or (
                cfg.input_feature_type == 'Color' and cfg.coord_lossy_residuals
            ) else 0,
            cfg.input_feature_type == 'Color' and cfg.lossless_color_enabled,
            *self.basic_block_args,
            None if not cfg.recurrent_part_enabled else cfg.activation
        )

        if not cfg.lossless_coord_enabled:
            assert not cfg.lossless_color_enabled
            self.encoder = encoder
            if cfg.input_feature_type == 'Occupation':
                self.decoder = Decoder(
                    cfg.compressed_channels,
                    0,
                    cfg.decoder_channels,
                    None, None, None, False, False, 0.0,
                    self.cfg.coord_recon_p2points_weighted_bce,
                    *self.basic_block_args,
                    loss_type=cfg.coord_recon_loss_type,
                    dist_upper_bound=cfg.dist_upper_bound
                )
            elif cfg.input_feature_type == 'Color':
                if not cfg.coord_lossy_residuals:
                    self.decoder = Decoder(
                        cfg.compressed_channels,
                        self.input_feature_channels,
                        cfg.decoder_channels,
                        None, None, None, False, False, 0.0,
                        self.cfg.coord_recon_p2points_weighted_bce,
                        *self.basic_block_args,
                        loss_type=cfg.coord_recon_loss_type,
                        dist_upper_bound=cfg.dist_upper_bound
                    )
                else:
                    indexed_em_fea = IndexedNoisyDeepFactorizedEM(
                        index_ranges=cfg.prior_indexes_range,
                        coding_ndim=2,
                        parameter_fns_factory=parameter_fns_factory,
                        num_filters=cfg.fea_num_filters,
                        bottleneck_process=cfg.bottleneck_process,
                        bottleneck_scaler=255,
                        quantize_indexes=cfg.quantize_indexes,
                        indexes_scaler=cfg.prior_indexes_scaler,
                        init_scale=1
                    )
                    lossy_decoder_residual_out_channels = \
                        (cfg.compressed_channels,) * (self.normal_part_coder_num - 1)\
                        + (self.input_feature_channels,)
                    lossy_decoder_residual_in_channels = \
                        lossy_decoder_residual_out_channels if not cfg.decoder_aware_residuals \
                        else tuple(_ * 2 for _ in lossy_decoder_residual_out_channels)
                    decoder_blocks_out_channels = (self.hyper_dec_fea_chnls,) * (self.normal_part_coder_num - 1) \
                        + (self.hyper_dec_fea_chnls // cfg.compressed_channels
                           * self.input_feature_channels,)
                    self.decoder = Decoder(
                        (cfg.compressed_channels,) * self.normal_part_coder_num,
                        decoder_blocks_out_channels,
                        cfg.decoder_channels,
                        lossy_decoder_residual_in_channels,
                        lossy_decoder_residual_out_channels,
                        indexed_em_fea,
                        cfg.hybrid_hyper_decoder_fea,
                        cfg.decoder_aware_residuals,
                        cfg.upper_fea_grad_scaler,
                        self.cfg.coord_recon_p2points_weighted_bce,
                        *self.basic_block_args,
                        loss_type=cfg.coord_recon_loss_type,
                        dist_upper_bound=cfg.dist_upper_bound
                    )
            else:
                raise NotImplementedError
            if not cfg.recurrent_part_enabled:
                enc_lossl = None
                hyper_dec_coord = None
                hyper_dec_fea = None
                assert len(cfg.skip_encoding_fea) == 0
            else:
                enc_lossl = self.init_enc_rec()
                hyper_dec_coord = self.init_hyper_dec_gen_up()
                hyper_dec_fea = self.init_hyper_dec_up()

        else:  # cfg.lossless_coord_enabled
            self.encoder = self.decoder = None
            hyper_dec_coord_in_chnls = [cfg.compressed_channels] * self.normal_part_coder_num
            hyper_dec_coord_out_chnls = \
                (len(cfg.lossless_coord_indexes_range),) * self.normal_part_coder_num
            hyper_dec_coord_intra_chnls = cfg.decoder_channels[::-1]

            hyper_dec_fea_in_chnls = [0] + [cfg.compressed_channels] * (self.normal_part_coder_num - 1)
            if cfg.input_feature_type == 'Occupation':
                assert 0 in cfg.skip_encoding_fea
                hyper_dec_fea_out_chnls = [0] + [self.hyper_dec_fea_chnls] * (self.normal_part_coder_num - 1)
                hyper_dec_fea_intra_chnls = [0, *cfg.decoder_channels[-2::-1]]
                for idx in cfg.skip_encoding_fea:
                    if idx == 0: continue
                    else:
                        hyper_dec_fea_out_chnls[idx] = hyper_dec_fea_in_chnls[idx - 1] = \
                            hyper_dec_coord_in_chnls[idx - 1] = hyper_dec_fea_intra_chnls[idx]
            elif cfg.input_feature_type == 'Color':
                if len(cfg.skip_encoding_fea) != 0:
                    raise NotImplementedError
                hyper_dec_fea_out_chnls = \
                    (self.hyper_dec_fea_chnls // cfg.compressed_channels
                     * self.input_feature_channels,) + \
                    (self.hyper_dec_fea_chnls,) * (self.normal_part_coder_num - 1)
                assert 0 not in cfg.skip_encoding_fea
                hyper_dec_fea_intra_chnls = cfg.decoder_channels[::-1]
            else:
                raise NotImplementedError

            if not cfg.recurrent_part_enabled:
                enc_lossl = EncoderPartiallyRecurrent(encoder)
            else:
                enc_lossl = EncoderPartiallyRecurrent(encoder, self.init_enc_rec())
                hyper_dec_coord_in_chnls += (cfg.compressed_channels,)
                hyper_dec_coord_out_chnls += (len(cfg.lossless_coord_indexes_range),)
                hyper_dec_coord_intra_chnls += (cfg.recurrent_part_channels,)
                hyper_dec_fea_in_chnls += (cfg.compressed_channels,)
                hyper_dec_fea_out_chnls += (self.hyper_dec_fea_chnls,)
                hyper_dec_fea_intra_chnls += (cfg.recurrent_part_channels,)

            hyper_dec_coord = self.init_hyper_dec_gen_up_rec(
                tuple(hyper_dec_coord_in_chnls),
                hyper_dec_coord_out_chnls,
                hyper_dec_coord_intra_chnls,
            )
            hyper_dec_fea = self.init_hyper_dec_up_rec(
                tuple(hyper_dec_fea_in_chnls),
                tuple(hyper_dec_fea_out_chnls),
                hyper_dec_fea_intra_chnls
            )

        em = self.init_em(parameter_fns_factory)
        if cfg.lossless_coord_enabled or cfg.recurrent_part_enabled:
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
            (self.cfg.recurrent_part_channels,),
            *self.basic_block_args
        )
        return hyper_dec_gen_up

    def init_hyper_dec_up(self):
        hyper_dec_up = HyperDecoderUpsample(
            self.cfg.compressed_channels,
            self.hyper_dec_fea_chnls,
            (self.cfg.recurrent_part_channels,),
            *self.basic_block_args
        )
        return hyper_dec_up

    def init_hyper_dec_gen_up_rec(
            self, in_channels: Tuple[int, ...],
            out_channels: Tuple[int, ...],
            intra_channels: Tuple[int, ...]):
        hyper_dec_gen_up_rec = HyperDecoderGenUpsamplePartiallyRecurrent(
            in_channels,
            out_channels,
            intra_channels,
            *self.basic_block_args
        )
        return hyper_dec_gen_up_rec

    def init_hyper_dec_up_rec(
            self, in_channels: Tuple[int, ...],
            out_channels: Tuple[int, ...],
            intra_channels: Tuple[int, ...]):
        hyper_dec_up_rec = HyperDecoderUpsamplePartiallyRecurrent(
            in_channels,
            out_channels,
            intra_channels,
            *self.basic_block_args
        )
        return hyper_dec_up_rec

    def forward(self, pc_data: PCData):
        lossless_coder_num = self.get_lossless_coder_num(pc_data.resolution)
        if self.training:
            sparse_pc = self.get_sparse_pc(pc_data.xyz, pc_data.color)
            return self.train_forward(sparse_pc, pc_data.training_step, lossless_coder_num)
        else:
            assert pc_data.batch_size == 1, 'Only supports batch size == 1 during testing.'
            if isinstance(pc_data.xyz, torch.Tensor):
                sparse_pc = self.get_sparse_pc(pc_data.xyz, pc_data.color)
                return self.test_forward(sparse_pc, pc_data, lossless_coder_num)
            else:
                sparse_pc_partitions = self.get_sparse_pc_partitions(pc_data.xyz, pc_data.color)
                return self.test_partitions_forward(sparse_pc_partitions, pc_data, lossless_coder_num)

    def get_sparse_pc(self, xyz: torch.Tensor,
                      color: Optional[torch.Tensor] = None,
                      tensor_stride: int = 1,
                      only_return_coords: bool = False)\
            -> Union[ME.SparseTensor, Tuple[ME.CoordinateMapKey, ME.CoordinateManager]]:
        assert xyz.min() >= 0
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
            if self.cfg.input_feature_type == 'Occupation':
                sparse_pc_feature = torch.ones(
                    xyz.shape[0], 1,
                    dtype=torch.float,
                    device=xyz.device
                )
            elif self.cfg.input_feature_type == 'Color':
                if self.cfg.lossless_color_enabled:
                    # Input point clouds are expected to be unnormalized discrete (-127~128 RGB) float32 tensor.
                    sparse_pc_feature = color - 127
                else:
                    # otherwise normalized (-0.5~0.5) float32 tensor.
                    sparse_pc_feature = (color / 255) - 0.5
            else:
                raise NotImplementedError
            sparse_pc = ME.SparseTensor(
                features=sparse_pc_feature,
                coordinates=xyz,
                tensor_stride=[tensor_stride] * 3,
                coordinate_manager=global_coord_mg,
                quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            )
            return sparse_pc

    def get_sparse_pc_partitions(
            self, xyz: List[torch.Tensor], color: Optional[List[torch.Tensor]]
    ) -> Generator:
        # The first one is supposed to be the original coordinates.
        for idx in range(1, len(xyz)):
            yield self.get_sparse_pc(
                xyz[idx], color[idx] if color is not None else None
            )

    def train_forward(self, sparse_pc: ME.SparseTensor, training_step: int,
                      lossless_coder_num: Optional[int]):
        warmup_forward = training_step < self.cfg.warmup_steps

        if not self.cfg.lossless_coord_enabled:
            strided_fea_list, points_num_list = self.encoder(sparse_pc)
            feature = strided_fea_list[-1]
        else:
            strided_fea_list = points_num_list = None
            feature = sparse_pc

        if self.em_lossless_based is not None:
            bottleneck_feature, loss_dict = self.em_lossless_based(
                feature, lossless_coder_num
            )
        else:
            bottleneck_feature, loss_dict = self.em(feature)

        if not self.cfg.lossless_coord_enabled:
            decoder_message: GenerativeUpsampleMessage = self.decoder(
                GenerativeUpsampleMessage(
                    fea=bottleneck_feature,
                    target_key=sparse_pc.coordinate_map_key,
                    points_num_list=points_num_list,
                    cached_fea_list=strided_fea_list[:-1]
                )
            )
            if self.cfg.input_feature_type == 'Color' and self.cfg.coord_lossy_residuals:
                assert len(decoder_message.em_loss_dict_list) == 1
                concat_loss_dicts(loss_dict, decoder_message.em_loss_dict_list[0])
            loss_dict['coord_recon_loss'] = self.get_coord_recon_loss(
                decoder_message.cached_pred_list,
                decoder_message.cached_target_list,
                decoder_message
            )

        if self.cfg.input_feature_type == 'Color' and not self.cfg.lossless_color_enabled:
            loss_dict['color_recon_loss'] = self.get_color_recon_loss(
                sparse_pc,
                bottleneck_feature if self.cfg.lossless_coord_enabled else decoder_message.fea
            )

        if warmup_forward and self.cfg.linear_warmup:
            warmup_bpp_loss_factor = self.cfg.warmup_bpp_loss_factor - \
                (self.cfg.warmup_bpp_loss_factor - self.cfg.bpp_loss_factor) \
                / self.cfg.warmup_steps * training_step
        else:
            warmup_bpp_loss_factor = self.cfg.warmup_bpp_loss_factor
        for key in loss_dict:
            if key.endswith('bits_loss'):
                if not self.cfg.lossless_coord_enabled:
                    # TODO: if self.em_lossless_based is None?
                    flag_warmup = warmup_forward
                else:
                    flag_warmup = warmup_forward and key.startswith('coord')
                if flag_warmup:
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

    def test_forward(self, sparse_pc: ME.SparseTensor, pc_data: PCData,
                     lossless_coder_num: Optional[int]):
        with Timer() as encoder_t, TorchCudaMaxMemoryAllocated() as encoder_m:
            compressed_bytes, sparse_tensor_coords = self.compress(sparse_pc, lossless_coder_num)
        del sparse_pc
        ME.clear_global_coordinate_manager()
        torch.cuda.empty_cache()
        with Timer() as decoder_t, TorchCudaMaxMemoryAllocated() as decoder_m:
            coord_recon, color_recon = self.decompress(compressed_bytes, sparse_tensor_coords)
        ret = self.evaluator.log_batch(
            preds=[coord_recon],
            targets=[pc_data.xyz[:, 1:]],
            compressed_bytes_list=[compressed_bytes],
            pc_data=pc_data,
            preds_color=[color_recon] if color_recon is not None else None,
            targets_color=[pc_data.color] if color_recon is not None else None,
            extra_info_dicts=[
                {'encoder_elapsed_time': encoder_t.elapsed_time,
                 'encoder_max_cuda_memory_allocated': encoder_m.max_memory_allocated,
                 'decoder_elapsed_time': decoder_t.elapsed_time,
                 'decoder_max_cuda_memory_allocated': decoder_m.max_memory_allocated}
            ]
        )
        return ret

    def test_partitions_forward(self, sparse_pc_partitions: Generator, pc_data: PCData,
                                lossless_coder_num: Optional[int]):
        with Timer() as encoder_t, TorchCudaMaxMemoryAllocated() as encoder_m:
            compressed_bytes, sparse_tensor_coords_list = self.compress_partitions(
                sparse_pc_partitions, lossless_coder_num
            )
        del sparse_pc_partitions
        ME.clear_global_coordinate_manager()
        torch.cuda.empty_cache()
        with Timer() as decoder_t, TorchCudaMaxMemoryAllocated() as decoder_m:
            coord_recon, color_recon = self.decompress_partitions(compressed_bytes, sparse_tensor_coords_list)
        ret = self.evaluator.log_batch(
            preds=[coord_recon],
            targets=[pc_data.xyz[0]],
            compressed_bytes_list=[compressed_bytes],
            pc_data=pc_data,
            preds_color=[color_recon] if color_recon is not None else None,
            targets_color=[pc_data.color[0]] if color_recon is not None else None,
            extra_info_dicts=[
                {'encoder_elapsed_time': encoder_t.elapsed_time,
                 'encoder_max_cuda_memory_allocated': encoder_m.max_memory_allocated,
                 'decoder_elapsed_time': decoder_t.elapsed_time,
                 'decoder_max_cuda_memory_allocated': decoder_m.max_memory_allocated}
            ]
        )
        return ret

    def compress(self, sparse_pc: ME.SparseTensor,
                 lossless_coder_num: Optional[int]) -> Tuple[bytes, Optional[torch.Tensor]]:
        if not self.cfg.lossless_coord_enabled:
            strided_fea_list, points_num_list = self.encoder(sparse_pc)
            feature = strided_fea_list[-1]
        else:
            strided_fea_list = points_num_list = None
            feature = sparse_pc

        if self.cfg.recurrent_part_enabled or self.cfg.lossless_coord_enabled:
            em_bytes, bottom_fea_recon, fea_recon = self.em_lossless_based.compress(
                feature, lossless_coder_num
            )
            if self.cfg.recurrent_part_enabled:
                assert bottom_fea_recon.C.shape[0] == 1
                sparse_tensor_coords = sparse_tensor_coords_stride = None
            else:
                sparse_tensor_coords = bottom_fea_recon.C
                sparse_tensor_coords_stride = bottom_fea_recon.tensor_stride[0]
        else:
            em_bytes_list, coding_batch_shape, fea_recon = self.em.compress(feature)
            assert coding_batch_shape == torch.Size([1])
            em_bytes = em_bytes_list[0]
            sparse_tensor_coords = feature.C
            sparse_tensor_coords_stride = feature.tensor_stride[0]

        if not self.cfg.lossless_coord_enabled and self.cfg.input_feature_type == 'Color' \
                and self.cfg.coord_lossy_residuals:
            decoder_message: GenerativeUpsampleMessage = self.decoder.compress(
                GenerativeUpsampleMessage(
                    fea=fea_recon,
                    points_num_list=points_num_list,
                    cached_fea_list=strided_fea_list[:-1]
                ))
            assert len(decoder_message.em_bytes_list) == 1
            lossy_em_bytes = decoder_message.em_bytes_list[0]
            em_bytes = len(lossy_em_bytes).to_bytes(3, 'little', signed=False) + \
                lossy_em_bytes + em_bytes

        if sparse_tensor_coords is not None:
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
            # TODO: avoid returning sparse_tensor_coords

        with io.BytesIO() as bs:
            if not self.cfg.lossless_coord_enabled and self.cfg.adaptive_pruning:
                bs.write(b''.join(
                    (_[0].to_bytes(3, 'little', signed=False) for _ in points_num_list)
                ))
            if self.cfg.recurrent_part_enabled:
                bs.write(lossless_coder_num.to_bytes(1, 'little', signed=False))
            bs.write(em_bytes)
            compressed_bytes = bs.getvalue()
        return compressed_bytes, sparse_tensor_coords

    def compress_partitions(self, sparse_pc_partitions: Generator,
                            lossless_coder_num: Optional[int]) \
            -> Tuple[bytes, List[torch.Tensor]]:
        compressed_bytes_list = []
        sparse_tensor_coords_list = []
        for sparse_pc in sparse_pc_partitions:
            compressed_bytes, sparse_tensor_coords = self.compress(sparse_pc, lossless_coder_num)
            ME.clear_global_coordinate_manager()
            compressed_bytes_list.append(compressed_bytes)
            sparse_tensor_coords_list.append(sparse_tensor_coords)

        # Log bytes of each partition.
        concat_bytes = b''.join((len(s).to_bytes(3, 'little', signed=False) + s
                                 for s in compressed_bytes_list))
        return concat_bytes, sparse_tensor_coords_list

    def decompress(self, compressed_bytes: bytes, sparse_tensor_coords: Optional[torch.Tensor]
                   ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = next(self.parameters()).device
        with io.BytesIO(compressed_bytes) as bs:
            if not self.cfg.lossless_coord_enabled and self.cfg.adaptive_pruning:
                points_num_list = []
                for idx in range(self.normal_part_coder_num):
                    points_num_list.append([int.from_bytes(bs.read(3), 'little', signed=False)])
            else:
                points_num_list = None
            if self.cfg.recurrent_part_enabled:
                lossless_coder_num = int.from_bytes(bs.read(1), 'little', signed=False)
                sparse_tensor_coords_bytes = None
            else:
                if not self.cfg.lossless_coord_enabled:
                    lossless_coder_num = None
                else:
                    lossless_coder_num = self.normal_part_coder_num
                sparse_tensor_coords_bytes_len = int.from_bytes(bs.read(3), 'little', signed=False)
                sparse_tensor_coords_bytes = bs.read(sparse_tensor_coords_bytes_len)
            if not self.cfg.lossless_coord_enabled and self.cfg.input_feature_type == 'Color' \
                    and self.cfg.coord_lossy_residuals:
                lossy_em_bytes_len = int.from_bytes(bs.read(3), 'little', signed=False)
                lossy_em_bytes = bs.read(lossy_em_bytes_len)
            else:
                lossy_em_bytes = None
            em_bytes = bs.read()

        if self.cfg.recurrent_part_enabled:
            tensor_stride = 2 ** lossless_coder_num
            if not self.cfg.lossless_coord_enabled:
                tensor_stride *= 2 ** self.normal_part_coder_num
            fea_recon = self.em_lossless_based.decompress(
                em_bytes,
                self.get_sparse_pc(
                    torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device=device),
                    tensor_stride=tensor_stride,
                    only_return_coords=True),
                lossless_coder_num
            )
        elif self.cfg.lossless_coord_enabled:
            fea_recon = self.em_lossless_based.decompress(
                em_bytes,
                self.get_sparse_pc(
                    sparse_tensor_coords,
                    tensor_stride=2 ** self.normal_part_coder_num,
                    only_return_coords=True
                ), lossless_coder_num
            )
        else:
            fea_recon = self.em.decompress(
                [em_bytes], torch.Size([1]), device,
                sparse_tensor_coords_tuple=self.get_sparse_pc(
                    sparse_tensor_coords,
                    tensor_stride=2 ** self.normal_part_coder_num,
                    only_return_coords=True
                )
            )

        if not self.cfg.lossless_coord_enabled:
            decoder_message = GenerativeUpsampleMessage(
                fea=fea_recon,
                points_num_list=points_num_list,
                em_bytes_list=[lossy_em_bytes]
            )
            if self.cfg.input_feature_type == 'Occupation':
                decoder_message = self.decoder(decoder_message)
            elif self.cfg.input_feature_type == 'Color' and not self.cfg.coord_lossy_residuals:
                decoder_message = self.decoder(decoder_message)
            elif self.cfg.input_feature_type == 'Color' and self.cfg.coord_lossy_residuals:
                decoder_message = self.decoder.decompress(
                    decoder_message, em_bytes_list_len=self.normal_part_coder_num
                )
            else:
                raise NotImplementedError
            coord_recon = decoder_message.fea.C[:, 1:]
            color_recon_raw = decoder_message.fea.F
        else:
            coord_recon = fea_recon.C[:, 1:]
            color_recon_raw = fea_recon.F
        if self.cfg.input_feature_type == 'Color':
            if self.cfg.lossless_color_enabled:
                color_recon = color_recon_raw.clip_(-127, 128).round_() + 127
            else:
                # Inverse transform is supposed to be done in the decoder
                color_recon = (color_recon_raw * 255).round_()
        else:
            color_recon = None
        return coord_recon, color_recon

    def decompress_partitions(self, concat_bytes: bytes,
                              sparse_tensor_coords_list: List[Optional[torch.Tensor]]
                              ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        coord_recon_list = []
        color_recon_list = []
        concat_bytes_len = len(concat_bytes)

        with io.BytesIO(concat_bytes) as bs:
            while bs.tell() != concat_bytes_len:
                length = int.from_bytes(bs.read(3), 'little', signed=False)
                coord_recon, color_recon = self.decompress(
                    bs.read(length), sparse_tensor_coords_list.pop(0)
                )
                coord_recon_list.append(coord_recon)
                color_recon_list.append(color_recon)

                ME.clear_global_coordinate_manager()

        coord_recon_concat = torch.cat(coord_recon_list, 0)
        if color_recon_list[0] is not None:
            color_recon_concat = torch.cat(color_recon_list, 0)
        else:
            color_recon_concat = None
        return coord_recon_concat, color_recon_concat

    def get_lossless_coder_num(self, resolution: Union[int, List[int]]) -> Optional[int]:
        if self.em_lossless_based is not None:
            if not isinstance(resolution, int):
                resolution = max(resolution)
            lossless_coder_num = math.ceil(math.log2(resolution))
            if not self.cfg.lossless_coord_enabled:
                lossless_coder_num -= self.normal_part_coder_num
            if not self.cfg.recurrent_part_enabled:
                lossless_coder_num = self.normal_part_coder_num
        else:
            lossless_coder_num = None
        return lossless_coder_num

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

    def rgb_to_yuvbt709(self, rgb: torch.Tensor):
        assert rgb.dtype == torch.float32
        assert rgb.ndim == 2 and rgb.shape[1] == 3
        return F.linear(rgb, self.rgb_to_yuvbt709_weight, self.rgb_to_yuvbt709_bias)

    def get_color_recon_loss(self, sparse_pc, color_pred):
        use_yuv = self.cfg.color_recon_loss_type.endswith('YUVBT709')
        if self.cfg.lossless_color_enabled:
            raise NotImplementedError

        target_fea_list = []
        ori_coord_list, ori_fea_list = sparse_pc.decomposed_coordinates_and_features
        pred_coord_list, pred_fea_list = color_pred.decomposed_coordinates_and_features
        batch_size = len(ori_coord_list)
        for idx in range(batch_size):
            nearest_idx = knn_points(
                pred_coord_list[idx][None].to(torch.float),
                ori_coord_list[idx][None].to(torch.float),
                K=1, return_sorted=False
            ).idx
            target_fea_list.append(((ori_fea_list[idx] + 0.5) * 255)[nearest_idx[0, :, 0]])
            # Inverse transform is supposed to be done in the decoder
            pred_fea_list[idx] = pred_fea_list[idx] * 255
            if use_yuv:
                target_fea_list[idx] = self.rgb_to_yuvbt709(target_fea_list[idx])
                pred_fea_list[idx] = self.rgb_to_yuvbt709(pred_fea_list[idx])

        if self.cfg.color_recon_loss_type.startswith('MSE'):
            loss_func = partial(F.mse_loss, reduction='sum')
        elif self.cfg.color_recon_loss_type.startswith('SmoothL1'):
            loss_func = partial(F.smooth_l1_loss, reduction='sum')
        elif self.cfg.color_recon_loss_type.startswith('L1'):
            loss_func = partial(F.l1_loss, reduction='sum')
        else:
            raise NotImplementedError

        recon_loss = sum(
            [loss_func(t, p) for t, p in zip(pred_fea_list, target_fea_list)]
        ) / color_pred.shape[0]
        factor = self.cfg.color_recon_loss_factor
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
