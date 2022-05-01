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
from torchvision.ops import sigmoid_focal_loss
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensorQuantizationMode

from lib.utils import Timer
from lib.mpeg_gpcc_utils import gpcc_octree_lossless_geom_encode, gpcc_decode
from lib.torch_utils import MLPBlock, TorchCudaMaxMemoryAllocated
from lib.data_utils import PCData, write_ply_file
from lib.evaluators import PCGCEvaluator
from lib.sparse_conv_layers import GenerativeUpsampleMessage
from lib.entropy_models.continuous_batched import \
    NoisyDeepFactorizedEntropyModel as PriorEntropyModel
from lib.entropy_models.hyperprior.noisy_deep_factorized.basic import \
    ScaleNoisyNormalEntropyModel as HyperPriorScaleNoisyNormalEntropyModel, \
    NoisyDeepFactorizedEntropyModel as HyperPriorNoisyDeepFactorizedEntropyModel
from lib.entropy_models.hyperprior.noisy_deep_factorized.sparse_tensor_specialized import \
    GeoLosslessNoisyDeepFactorizedEntropyModel

from models.convolutional.baseline.layers import Scaler, \
    Encoder, Decoder, \
    HyperEncoder, HyperDecoder, \
    HyperDecoderGenUpsample, HyperDecoderUpsample, \
    EncoderRecurrent, EncoderPartiallyRecurrent, \
    HyperDecoderGenUpsamplePartiallyRecurrent, \
    HyperDecoderUpsamplePartiallyRecurrent
from models.convolutional.baseline.model_config import ModelConfig


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
            cfg.mpeg_pcc_error_threads,
            cfg.chamfer_dist_test_phase
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
        else:
            raise NotImplementedError
        self.em_lossl_dec_fea_chnls = cfg.compressed_channels * (
            len(cfg.prior_indexes_range)
            if not cfg.lossless_hybrid_hyper_decoder_fea
            else len(cfg.prior_indexes_range) + 1
        )

        encoder = Encoder(
            self.input_feature_channels,
            cfg.compressed_channels if not cfg.recurrent_part_enabled
            else cfg.recurrent_part_channels,
            cfg.encoder_channels,
            cfg.first_conv_kernel_size,
            cfg.adaptive_pruning,
            cfg.adaptive_pruning_num_scaler,
            cfg.encoder_scaler if not cfg.recurrent_part_enabled else 1,
            0 if not cfg.lossless_coord_enabled else cfg.compressed_channels,
            cfg.encoder_scaler,
            cfg.input_feature_type == 'Color' and cfg.lossless_color_enabled,
            *self.basic_block_args,
            None if not cfg.recurrent_part_enabled else cfg.activation
        )

        if not cfg.lossless_coord_enabled:
            assert not self.cfg.lossless_color_enabled
            self.encoder = encoder
            self.decoder = Decoder(
                cfg.compressed_channels,
                0 if cfg.input_feature_type == 'Occupation' else self.input_feature_channels,
                cfg.decoder_channels,
                1 / cfg.encoder_scaler,
                *self.basic_block_args,
                loss_type=
                'BCE' if cfg.coord_recon_loss_type == 'Focal'
                else cfg.coord_recon_loss_type,
                dist_upper_bound=cfg.dist_upper_bound
            )
            if not cfg.recurrent_part_enabled:
                enc_lossl = None
                hyper_dec_coord_lossl = None
                hyper_dec_fea_lossl = None
                skip_encoding_top_fea = None
            else:
                enc_lossl = self.init_enc_rec()
                hyper_dec_coord_lossl = self.init_hyper_dec_gen_up()
                hyper_dec_fea_lossl = self.init_hyper_dec_up()
                skip_encoding_top_fea = 'no_skip'

        else:  # cfg.lossless_coord_enabled
            self.encoder = self.decoder = None
            hyper_dec_coord_lossl_out_chnls = \
                (len(cfg.lossless_coord_indexes_range),) * len(cfg.decoder_channels)
            hyper_dec_coord_lossl_intra_chnls = cfg.decoder_channels[::-1]

            if cfg.input_feature_type == 'Occupation':
                hyper_dec_fea_lossl_out_chnls = (self.em_lossl_dec_fea_chnls,) * (len(cfg.decoder_channels) - 1)
                hyper_dec_fea_lossl_intra_chnls = cfg.decoder_channels[-2::-1]
                skip_encoding_top_fea = 'skip'
            elif cfg.input_feature_type == 'Color':
                hyper_dec_fea_lossl_out_chnls = \
                    (self.em_lossl_dec_fea_chnls // self.cfg.compressed_channels
                     * self.input_feature_channels,) + \
                    (self.em_lossl_dec_fea_chnls,) * (len(cfg.decoder_channels) - 1)
                skip_encoding_top_fea = 'no_skip'
                hyper_dec_fea_lossl_intra_chnls = cfg.decoder_channels[::-1]
            else:
                raise NotImplementedError

            if not cfg.recurrent_part_enabled:
                enc_lossl = EncoderPartiallyRecurrent(encoder)
            else:
                enc_lossl = EncoderPartiallyRecurrent(encoder, self.init_enc_rec())
                hyper_dec_coord_lossl_out_chnls += (len(cfg.lossless_coord_indexes_range),)
                hyper_dec_coord_lossl_intra_chnls += (cfg.recurrent_part_channels,)
                hyper_dec_fea_lossl_out_chnls += (self.em_lossl_dec_fea_chnls,)
                hyper_dec_fea_lossl_intra_chnls += (cfg.recurrent_part_channels,)

            hyper_dec_coord_lossl = self.init_hyper_dec_gen_up_rec(
                hyper_dec_coord_lossl_out_chnls,
                hyper_dec_coord_lossl_intra_chnls,
            )
            hyper_dec_fea_lossl = self.init_hyper_dec_up_rec(
                hyper_dec_fea_lossl_out_chnls,
                hyper_dec_fea_lossl_intra_chnls
            )

        def parameter_fns_factory(in_channels, out_channels):
            ret = [
                MLPBlock(in_channels, out_channels,
                         bn=None, act=self.cfg.activation),
                nn.Linear(out_channels, out_channels,
                          bias=True)
            ]
            if self.cfg.prior_indexes_post_scaler != 1.0:
                ret.insert(0, Scaler(self.cfg.prior_indexes_post_scaler))
            return nn.Sequential(*ret)
        em = self.init_em(parameter_fns_factory)
        if cfg.lossless_coord_enabled or cfg.recurrent_part_enabled:
            self.em = None
            self.em_lossless_based = self.init_em_lossless_based(
                em, enc_lossl,
                hyper_dec_coord_lossl, hyper_dec_fea_lossl,
                parameter_fns_factory, skip_encoding_top_fea
            )
        else:
            self.em = em
            self.em_lossless_based = None

    def init_em(self, parameter_fns_factory) -> nn.Module:
        if self.cfg.recurrent_part_enabled:
            assert self.cfg.hyperprior == 'None'

        if self.cfg.hyperprior == 'None':
            em = PriorEntropyModel(
                batch_shape=torch.Size([self.cfg.compressed_channels]),
                broadcast_shape_bytes=(3,) if not self.cfg.recurrent_part_enabled else (0,),
                coding_ndim=2,
                init_scale=2
            )
        else:
            hyper_encoder = HyperEncoder(
                1 / self.cfg.encoder_scaler,
                self.cfg.hyper_encoder_scaler,
                self.cfg.compressed_channels,
                self.cfg.hyper_compressed_channels,
                self.cfg.hyper_encoder_channels,
                *self.basic_block_args
            )
            hyper_decoder = HyperDecoder(
                1 / self.cfg.hyper_encoder_scaler,
                self.cfg.prior_indexes_scaler,
                self.cfg.hyper_compressed_channels,
                self.cfg.compressed_channels * len(self.cfg.prior_indexes_range),
                self.cfg.hyper_decoder_channels,
                *self.basic_block_args
            )

            if self.cfg.hyperprior == 'ScaleNoisyNormal':
                assert len(self.cfg.prior_indexes_range) == 1
                em = HyperPriorScaleNoisyNormalEntropyModel(
                    hyper_encoder=hyper_encoder,
                    hyper_decoder=hyper_decoder,
                    hyperprior_batch_shape=torch.Size([self.cfg.hyper_compressed_channels]),
                    hyperprior_broadcast_shape_bytes=(3,),
                    prior_bytes_num_bytes=3,
                    coding_ndim=2,
                    num_scales=self.cfg.prior_indexes_range[0],
                    scale_min=0.11,
                    scale_max=64
                )
            elif self.cfg.hyperprior == 'NoisyDeepFactorized':
                em = HyperPriorNoisyDeepFactorizedEntropyModel(
                    hyper_encoder=hyper_encoder,
                    hyper_decoder=hyper_decoder,
                    hyperprior_batch_shape=torch.Size([self.cfg.hyper_compressed_channels]),
                    hyperprior_broadcast_shape_bytes=(3,),
                    prior_bytes_num_bytes=3,
                    coding_ndim=2,
                    index_ranges=self.cfg.prior_indexes_range,
                    parameter_fns_type='transform',
                    parameter_fns_factory=parameter_fns_factory,
                    num_filters=(1, 3, 3, 3, 1),
                    quantize_indexes=True
                )
            else:
                raise NotImplementedError
        return em

    def init_em_lossless_based(
            self, bottom_fea_entropy_model, encoder_geo_lossless,
            hyper_decoder_coord_geo_lossless, hyper_decoder_fea_geo_lossless,
            parameter_fns_factory, skip_encoding_top_fea
    ):
        em_lossless_based = GeoLosslessNoisyDeepFactorizedEntropyModel(
            bottom_fea_entropy_model=bottom_fea_entropy_model,
            encoder=encoder_geo_lossless,
            hyper_decoder_coord=hyper_decoder_coord_geo_lossless,
            hyper_decoder_fea=hyper_decoder_fea_geo_lossless,
            hybrid_hyper_decoder_fea=self.cfg.lossless_hybrid_hyper_decoder_fea,
            coord_index_ranges=self.cfg.lossless_coord_indexes_range,
            coord_parameter_fns_type='transform',
            coord_parameter_fns_factory=parameter_fns_factory,
            coord_num_filters=(1, 3, 3, 3, 1),
            fea_index_ranges=self.cfg.prior_indexes_range,
            fea_parameter_fns_type='transform',
            fea_parameter_fns_factory=parameter_fns_factory,
            fea_num_filters=(1, 3, 3, 3, 3, 1),
            skip_encoding_top_fea=skip_encoding_top_fea,
            quantize_indexes=True
        )
        return em_lossless_based
    
    def init_enc_rec(self):
        enc_rec = EncoderRecurrent(
            self.cfg.recurrent_part_channels,
            self.cfg.compressed_channels,
            self.cfg.encoder_scaler,
            *self.basic_block_args
        )
        return enc_rec
    
    def init_hyper_dec_gen_up(self):
        hyper_dec_gen_up = HyperDecoderGenUpsample(
            1 / self.cfg.encoder_scaler,
            self.cfg.prior_indexes_scaler,
            self.cfg.compressed_channels,
            len(self.cfg.lossless_coord_indexes_range),
            (self.cfg.recurrent_part_channels,),
            *self.basic_block_args
        )
        return hyper_dec_gen_up

    def init_hyper_dec_up(self):
        hyper_dec_up = HyperDecoderUpsample(
            1 / self.cfg.encoder_scaler,
            self.cfg.prior_indexes_scaler,
            self.cfg.compressed_channels,
            self.em_lossl_dec_fea_chnls,
            (self.cfg.recurrent_part_channels,),
            *self.basic_block_args
        )
        return hyper_dec_up

    def init_hyper_dec_gen_up_rec(
            self, out_channels: Tuple[int, ...],
            intra_channels: Tuple[int, ...]):
        hyper_dec_gen_up_rec = HyperDecoderGenUpsamplePartiallyRecurrent(
            1 / self.cfg.encoder_scaler,
            self.cfg.prior_indexes_scaler,
            self.cfg.compressed_channels,
            out_channels,
            intra_channels,
            *self.basic_block_args
        )
        return hyper_dec_gen_up_rec

    def init_hyper_dec_up_rec(
            self, out_channels: Tuple[int, ...],
            intra_channels: Tuple[int, ...]):
        hyper_dec_up_rec = HyperDecoderUpsamplePartiallyRecurrent(
            1 / self.cfg.encoder_scaler,
            self.cfg.prior_indexes_scaler,
            self.cfg.compressed_channels,
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
                # Input point clouds are expected to be unnormalized discrete (0~255) float32 tensor.
                # In lossy color compression case, we use normalized colors.
                # In lossless color compression case, we use unnormalized colors.
                sparse_pc_feature = color
                if not self.cfg.lossless_color_enabled:
                    sparse_pc_feature /= 255
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
            points_num_list = None
            feature = sparse_pc

        if self.em_lossless_based is not None:
            bottleneck_feature, loss_dict = self.em_lossless_based(
                feature, lossless_coder_num
            )
        else:
            bottleneck_feature, loss_dict = self.em(feature)

        for key in loss_dict:
            # TODO: how about
            #  (warmup_forward and key.startswith('fea'))
            #  if self.em_lossless_based is not None else warmup_forward)
            if key.endswith('bits_loss'):
                loss_dict[key] = loss_dict[key] * (
                    (self.cfg.warmup_bpp_loss_factor
                     if ((warmup_forward and key.startswith('fea'))
                         if self.cfg.lossless_coord_enabled else warmup_forward)
                     else self.cfg.bpp_loss_factor) / sparse_pc.shape[0]
                )

        if not self.cfg.lossless_coord_enabled:
            decoder_message = self.decoder(
                GenerativeUpsampleMessage(
                    fea=bottleneck_feature,
                    target_key=sparse_pc.coordinate_map_key,
                    points_num_list=points_num_list
                )
            )
            loss_dict['coord_recon_loss'] = self.get_coord_recon_loss(
                decoder_message.cached_pred_list,
                decoder_message.cached_target_list
            )

        if self.cfg.input_feature_type == 'Color' and not self.cfg.lossless_color_enabled:
            loss_dict['color_recon_loss'] = self.get_color_recon_loss(
                sparse_pc,
                bottleneck_feature if self.cfg.lossless_coord_enabled else decoder_message.fea
            )

        loss_dict['loss'] = sum(loss_dict.values())
        for key in loss_dict:
            if key != 'loss':
                loss_dict[key] = loss_dict[key].item()
        return loss_dict

    def test_forward(self, sparse_pc: ME.SparseTensor, pc_data: PCData,
                     lossless_coder_num: Optional[int]):
        with Timer() as encoder_t, TorchCudaMaxMemoryAllocated() as encoder_m:
            compressed_string, sparse_tensor_coords = self.compress(sparse_pc, lossless_coder_num)
        del sparse_pc
        ME.clear_global_coordinate_manager()
        torch.cuda.empty_cache()
        with Timer() as decoder_t, TorchCudaMaxMemoryAllocated() as decoder_m:
            coord_recon, color_recon = self.decompress(compressed_string, sparse_tensor_coords)
        ret = self.evaluator.log_batch(
            preds=[coord_recon],
            targets=[pc_data.xyz[:, 1:]],
            compressed_strings=[compressed_string],
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
            compressed_string, sparse_tensor_coords_list = self.compress_partitions(
                sparse_pc_partitions, lossless_coder_num
            )
        del sparse_pc_partitions
        ME.clear_global_coordinate_manager()
        torch.cuda.empty_cache()
        with Timer() as decoder_t, TorchCudaMaxMemoryAllocated() as decoder_m:
            coord_recon, color_recon = self.decompress_partitions(compressed_string, sparse_tensor_coords_list)
        ret = self.evaluator.log_batch(
            preds=[coord_recon],
            targets=[pc_data.xyz[0]],
            compressed_strings=[compressed_string],
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
            points_num_list = None
            feature = sparse_pc

        if self.cfg.recurrent_part_enabled or self.cfg.lossless_coord_enabled:
            em_string, bottom_fea_recon = self.em_lossless_based.compress(
                feature, lossless_coder_num
            )
            if self.cfg.recurrent_part_enabled:
                assert bottom_fea_recon.C.shape[0] == 1
                sparse_tensor_coords = None
            else:
                sparse_tensor_coords = bottom_fea_recon.C
        else:
            em_strings, coding_batch_shape, _ = self.em.compress(feature)
            assert coding_batch_shape == torch.Size([1])
            em_string = em_strings[0]
            sparse_tensor_coords = feature.C

        if sparse_tensor_coords is not None:
            h = hashlib.md5()
            h.update(str(time.time()).encode())
            tmp_file_path = 'tmp-' + h.hexdigest()
            write_ply_file(sparse_tensor_coords[:, 1:], f'{tmp_file_path}.ply')
            gpcc_octree_lossless_geom_encode(
                f'{tmp_file_path}.ply', f'{tmp_file_path}.bin',
                command=self.cfg.mpeg_gpcc_command
            )
            with open(f'{tmp_file_path}.bin', 'rb') as f:
                sparse_tensor_coords_bin = f.read()
            os.remove(f'{tmp_file_path}.ply')
            os.remove(f'{tmp_file_path}.bin')
            em_string = len(sparse_tensor_coords_bin).to_bytes(3, 'little', signed=False) + \
                sparse_tensor_coords_bin + em_string
            # TODO: avoid returning sparse_tensor_coords

        with io.BytesIO() as bs:
            if not self.cfg.lossless_coord_enabled and self.cfg.adaptive_pruning:
                bs.write(b''.join(
                    (_[0].to_bytes(3, 'little', signed=False) for _ in points_num_list)
                ))
            if self.cfg.recurrent_part_enabled:
                bs.write(lossless_coder_num.to_bytes(1, 'little', signed=False))
            bs.write(em_string)
            compressed_string = bs.getvalue()
        return compressed_string, sparse_tensor_coords

    def compress_partitions(self, sparse_pc_partitions: Generator,
                            lossless_coder_num: Optional[int]) \
            -> Tuple[bytes, List[torch.Tensor]]:
        compressed_string_list = []
        sparse_tensor_coords_list = []
        for sparse_pc in sparse_pc_partitions:
            compressed_string, sparse_tensor_coords = self.compress(sparse_pc, lossless_coder_num)
            ME.clear_global_coordinate_manager()
            compressed_string_list.append(compressed_string)
            sparse_tensor_coords_list.append(sparse_tensor_coords)

        # Log bytes of each partition.
        concat_string = b''.join((len(s).to_bytes(3, 'little', signed=False) + s
                                 for s in compressed_string_list))
        return concat_string, sparse_tensor_coords_list

    def decompress(self, compressed_string: bytes, sparse_tensor_coords: Optional[torch.Tensor]
                   ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = next(self.parameters()).device
        with io.BytesIO(compressed_string) as bs:
            if not self.cfg.lossless_coord_enabled and self.cfg.adaptive_pruning:
                points_num_list = []
                for idx in range(len(self.cfg.decoder_channels)):
                    points_num_list.append([int.from_bytes(bs.read(3), 'little', signed=False)])
            else:
                points_num_list = None
            if self.cfg.recurrent_part_enabled:
                lossless_coder_num = int.from_bytes(bs.read(1), 'little', signed=False)
            else:
                if not self.cfg.lossless_coord_enabled:
                    lossless_coder_num = None
                else:
                    lossless_coder_num = len(self.cfg.decoder_channels)
                sparse_tensor_coords_bytes = int.from_bytes(bs.read(3), 'little', signed=False)
                bs.read(sparse_tensor_coords_bytes)
            em_string = bs.read()

        if self.cfg.recurrent_part_enabled:
            tensor_stride = 2 ** lossless_coder_num
            if not self.cfg.lossless_coord_enabled:
                tensor_stride *= 2 ** len(self.cfg.decoder_channels)
            fea_recon = self.em_lossless_based.decompress(
                em_string, device,
                self.get_sparse_pc(
                    torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device=device),
                    tensor_stride=tensor_stride,
                    only_return_coords=True),
                lossless_coder_num
            )
        elif self.cfg.lossless_coord_enabled:
            fea_recon = self.em_lossless_based.decompress(
                em_string, device,
                self.get_sparse_pc(
                    sparse_tensor_coords,
                    tensor_stride=2 ** len(self.cfg.decoder_channels),
                    only_return_coords=True
                ), lossless_coder_num
            )
        else:
            fea_recon = self.em.decompress(
                [em_string], torch.Size([1]), device,
                sparse_tensor_coords_tuple=self.get_sparse_pc(
                    sparse_tensor_coords,
                    tensor_stride=2 ** len(self.cfg.decoder_channels),
                    only_return_coords=True
                )
            )

        if not self.cfg.lossless_coord_enabled:
            decoder_message = self.decoder(
                GenerativeUpsampleMessage(
                    fea=fea_recon,
                    points_num_list=points_num_list
                )
            )
            # The last one is supposed to be the final output of the decoder.
            coord_recon = decoder_message.cached_pred_list[-1].C[:, 1:]
            color_recon_raw = decoder_message.fea.F
        else:
            coord_recon = fea_recon.C[:, 1:]
            color_recon_raw = fea_recon.F
        if self.cfg.input_feature_type == 'Color':
            if not self.cfg.lossless_color_enabled:
                color_recon_raw *= 255
            color_recon = color_recon_raw.clip_(0, 255).round_().to('cpu', torch.uint8)
        else:
            color_recon = None
        return coord_recon, color_recon

    def decompress_partitions(self, concat_string: bytes,
                              sparse_tensor_coords_list: List[Optional[torch.Tensor]]
                              ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        coord_recon_list = []
        color_recon_list = []
        concat_string_len = len(concat_string)

        with io.BytesIO(concat_string) as bs:
            while bs.tell() != concat_string_len:
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
        assert len(self.cfg.encoder_channels) == len(self.cfg.decoder_channels) + 1
        if self.em_lossless_based is not None:
            if not isinstance(resolution, int):
                for r in resolution[1:]:
                    assert r == resolution[0]
                resolution = resolution[0]
            lossless_coder_num = math.ceil(math.log2(resolution))
            if not self.cfg.lossless_coord_enabled:
                lossless_coder_num -= len(self.cfg.decoder_channels)
            if not self.cfg.recurrent_part_enabled:
                lossless_coder_num = len(self.cfg.decoder_channels)
        else:
            lossless_coder_num = None
        return lossless_coder_num

    def get_coord_recon_loss(self, cached_pred_list, cached_target_list):
        if self.cfg.coord_recon_loss_type == 'BCE':
            loss_func = F.binary_cross_entropy_with_logits
        elif self.cfg.coord_recon_loss_type == 'Focal':
            loss_func = partial(
                sigmoid_focal_loss,
                alpha=0.25,
                gamma=2.0,
                reduction='mean'
            )
        elif self.cfg.coord_recon_loss_type == 'Dist':
            loss_func = F.smooth_l1_loss
        else:
            raise NotImplementedError

        recon_loss_list = [loss_func(
                pred.F.squeeze(dim=1),
                target.type(pred.F.dtype))
             for pred, target in zip(cached_pred_list, cached_target_list)]
        recon_loss = sum(recon_loss_list) / len(recon_loss_list)
        factor = self.cfg.coord_recon_loss_factor
        if factor != 1:
            recon_loss *= factor
        return recon_loss

    def get_color_recon_loss(self, sparse_pc, color_pred):
        kernel_map = sparse_pc.coordinate_manager.kernel_map(
            sparse_pc.coordinate_map_key,
            color_pred.coordinate_map_key,
            kernel_size=1
        )[0].to(torch.long)
        if self.cfg.color_recon_loss_type == 'SmoothL1':
            loss_func = F.smooth_l1_loss
        else:
            raise NotImplementedError
        recon_loss = loss_func(
            sparse_pc.F[kernel_map[0]],
            color_pred.F[kernel_map[1]]
        )
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


def model_debug():
    cfg = ModelConfig()
    cfg.prior_indexes_range = (8, 8, 8, 8)
    cfg.input_feature_type = 'Occupation'
    cfg.lossless_coord_enabled = False
    cfg.recurrent_part_enabled = False
    cfg.lossless_color_enabled = False
    cfg.encoder_scaler = 1000
    model = PCC(cfg).cuda()
    xyz_c = [ME.utils.sparse_quantize(torch.randint(0, 64, (1000, 3), dtype=torch.float32)) for _ in range(2)]
    xyz = ME.utils.batched_coordinates(xyz_c).cuda()
    color = torch.randint(
        0, 256, (xyz.shape[0], 3), device=xyz.device, dtype=torch.float32
    ) if cfg.input_feature_type == 'Color' else None
    pc_data = PCData(
        xyz=xyz,
        color=color,
        resolution=[64] * 2
    )
    pc_data.training_step = 0
    out = model(pc_data)
    out['loss'].backward()
    model.eval()
    with torch.no_grad():
        sample_0 = xyz[:, 0] == 0
        test_out = model(
            PCData(
                xyz=xyz[sample_0, :],
                color=color[sample_0] if color else None,
                file_path=[''], ori_resolution=[0], resolution=[64],
                batch_size=1
            ))
    print('Done')


if __name__ == '__main__':
    model_debug()
