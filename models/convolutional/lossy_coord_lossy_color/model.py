import io
from typing import List, Union, Tuple, Generator, Optional
import math

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensorQuantizationMode
try:
    from pytorch3d.ops.knn import knn_points, knn_gather
except ImportError: pass

from lib.utils import Timer
from lib.torch_utils import MLPBlock, TorchCudaMaxMemoryAllocated, concat_loss_dicts
from lib.data_utils import PCData
from lib.evaluators import PCGCEvaluator
from lib.entropy_models.continuous_batched import NoisyDeepFactorizedEntropyModel as PriorEM

from .geo_lossl_em import GeoLosslessNoisyDeepFactorizedEntropyModel
from .layers import Encoder, Decoder, \
    HyperDecoderGenUpsample, HyperDecoderUpsample, EncoderRecurrent, \
    ResidualRecurrent, DecoderRecurrent
from .model_config import ModelConfig


class PCC(nn.Module):

    @staticmethod
    def params_divider(s: str) -> int:
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

    def __init__(self, cfg: ModelConfig):
        super(PCC, self).__init__()
        self.cfg = cfg
        ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
        self.minkowski_algorithm = getattr(ME.MinkowskiAlgorithm, cfg.minkowski_algorithm)
        self.evaluator = PCGCEvaluator(
            cfg.mpeg_pcc_error_command, 16
        )
        self.basic_block_args = (
            cfg.basic_block_type,
            cfg.conv_region_type,
            cfg.basic_block_num,
            cfg.use_batch_norm,
            cfg.activation
        )
        assert len(cfg.encoder_channels) == 2

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
            4,
            cfg.recurrent_part_channels,
            cfg.encoder_channels,
            cfg.adaptive_pruning,
            cfg.adaptive_pruning_num_scaler,
            *self.basic_block_args
        )
        self.decoder = Decoder(
            cfg.recurrent_part_channels,
            3,
            cfg.decoder_channels,
            cfg.coord_recon_loss_factor,
            cfg.color_recon_loss_factor,
            *self.basic_block_args,
        )
        enc_lossl = EncoderRecurrent(
            cfg.recurrent_part_channels,
            cfg.recurrent_part_channels,
            cfg.basic_block_type,
            cfg.conv_region_type,
            cfg.basic_block_num,
            cfg.use_batch_norm,
            cfg.activation
        )
        hyper_dec_coord = HyperDecoderGenUpsample(
            cfg.recurrent_part_channels,
            len(cfg.lossless_coord_indexes_range),
            cfg.recurrent_part_channels,
            cfg.basic_block_type,
            cfg.conv_region_type,
            max(cfg.basic_block_num - 2, 1),
            cfg.use_batch_norm,
            cfg.activation
        )
        hyper_dec_fea = HyperDecoderUpsample(
            cfg.recurrent_part_channels,
            cfg.compressed_channels * len(cfg.prior_indexes_range) + cfg.recurrent_part_channels,
            self.cfg.recurrent_part_channels,
            *self.basic_block_args,
            skip_encoding_fea=cfg.skip_encoding_fea,
            out_channels2=cfg.recurrent_part_channels
        )
        self.em_lossless_based = self.init_em_lossless_based(
            self.init_em(), enc_lossl,
            ResidualRecurrent(
                cfg.recurrent_part_channels + cfg.recurrent_part_channels,
                cfg.compressed_channels, cfg.use_batch_norm, cfg.activation,
                cfg.skip_encoding_fea,
            ),
            DecoderRecurrent(
                cfg.compressed_channels + cfg.recurrent_part_channels,
                cfg.recurrent_part_channels,
                cfg.use_batch_norm, cfg.activation,
                cfg.skip_encoding_fea, cfg.recurrent_part_channels
            ),
            hyper_dec_coord, hyper_dec_fea,
            parameter_fns_factory
        )

    def init_em(self) -> nn.Module:
        em = PriorEM(
            batch_shape=torch.Size([self.cfg.recurrent_part_channels]),
            coding_ndim=2,
            bottleneck_process=self.cfg.bottleneck_process,
            bottleneck_scaler=1,
            init_scale=1,
            broadcast_shape_bytes=(0,),
        )
        return em

    def init_em_lossless_based(
            self, bottom_fea_entropy_model, encoder_geo_lossless, residual_block, decoder_block,
            hyper_decoder_coord_geo_lossless, hyper_decoder_fea_geo_lossless,
            parameter_fns_factory
    ):
        em_lossless_based = GeoLosslessNoisyDeepFactorizedEntropyModel(
            bottom_fea_entropy_model=bottom_fea_entropy_model,
            encoder=encoder_geo_lossless,
            residual_block=residual_block,
            decoder_block=decoder_block,
            hyper_decoder_coord=hyper_decoder_coord_geo_lossless,
            hyper_decoder_fea=hyper_decoder_fea_geo_lossless,
            coord_index_ranges=self.cfg.lossless_coord_indexes_range,
            coord_parameter_fns_type='transform',
            coord_parameter_fns_factory=parameter_fns_factory,
            coord_num_filters=(1, 3, 3, 1),
            fea_index_ranges=self.cfg.prior_indexes_range,
            fea_parameter_fns_type='transform',
            fea_parameter_fns_factory=parameter_fns_factory,
            skip_encoding_fea=self.cfg.skip_encoding_fea,
            fea_num_filters=self.cfg.lossless_fea_num_filters,
            bottleneck_fea_process=self.cfg.bottleneck_process,
            bottleneck_scaler=self.cfg.bottleneck_scaler,
            quantize_indexes=self.cfg.quantize_indexes,
            indexes_scaler=self.cfg.prior_indexes_scaler,
            init_scale=5 / self.cfg.bottleneck_scaler
        )
        return em_lossless_based

    def forward(self, pc_data: PCData):
        if self.training:
            sparse_pc = self.get_sparse_pc(pc_data.xyz, pc_data.color)
            return self.train_forward(sparse_pc, pc_data.training_step, pc_data.batch_size)
        else:
            assert pc_data.batch_size == 1, 'Only supports batch size == 1 during testing.'
            if isinstance(pc_data.xyz, torch.Tensor):
                sparse_pc = self.get_sparse_pc(pc_data.xyz, pc_data.color)
                return self.test_forward(sparse_pc, pc_data)
            else:
                sparse_pc_partitions = self.get_sparse_pc_partitions(pc_data.xyz, pc_data.color)
                return self.test_partitions_forward(sparse_pc_partitions, pc_data)

    def get_sparse_pc(self, xyz: torch.Tensor, color: Optional[torch.Tensor] = None,
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
            sparse_pc_feature = torch.cat((
                torch.div(color, 255),
                torch.full(
                    (color.shape[0], 1), fill_value=2,
                    dtype=torch.float,
                    device=color.device
                )), 1)
            sparse_pc = ME.SparseTensor(
                features=sparse_pc_feature,
                coordinates=xyz,
                tensor_stride=[tensor_stride] * 3,
                coordinate_manager=global_coord_mg,
                quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            )
            return sparse_pc

    def get_sparse_pc_partitions(self, xyz: List[torch.Tensor], color: List[torch.Tensor]) -> Generator:
        # The first one is supposed to be the original coordinates.
        for idx in range(1, len(xyz)):
            yield self.get_sparse_pc(xyz[idx], color[idx])

    def train_forward(self, sparse_pc: ME.SparseTensor,
                      training_step: int, batch_size: int):
        warmup_forward = training_step < self.cfg.warmup_steps

        strided_fea_list, points_num_list = self.encoder(sparse_pc)
        feature = strided_fea_list[-1]

        bottleneck_feature, loss_dict = self.em_lossless_based(feature, batch_size)

        decoder_loss_dict = self.decoder(
            bottleneck_feature, points_num_list,
            sparse_pc.coordinate_map_key, sparse_pc.F[:, :-1].mul(255).round_()
        )
        concat_loss_dicts(loss_dict, decoder_loss_dict)

        if warmup_forward and self.cfg.linear_warmup:
            warmup_bpp_loss_factor = self.cfg.warmup_bpp_loss_factor - \
                (self.cfg.warmup_bpp_loss_factor - self.cfg.bpp_loss_factor) \
                / self.cfg.warmup_steps * training_step
        else:
            warmup_bpp_loss_factor = self.cfg.warmup_bpp_loss_factor
        for key in loss_dict:
            if key.endswith('bits_loss'):
                if warmup_forward and 'coord' not in key:  # do not warm up for coord_idx_bits_loss
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
            coord_recon, color_recon = self.decompress(compressed_bytes, sparse_tensor_coords)
        ret = self.evaluator.log_batch(
            preds=[coord_recon],
            targets=[pc_data.xyz[:, 1:]],
            compressed_bytes_list=[compressed_bytes],
            pc_data=pc_data,
            preds_color=[color_recon],
            targets_color=[pc_data.color],
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
            coord_recon, color_recon = self.decompress_partitions(compressed_bytes, sparse_tensor_coords_list)
        ret = self.evaluator.log_batch(
            preds=[coord_recon],
            targets=[pc_data.xyz[0]],
            compressed_bytes_list=[compressed_bytes],
            pc_data=pc_data,
            preds_color=[color_recon],
            targets_color=[pc_data.color[0]],
            extra_info_dicts=[
                {'encoder_elapsed_time': encoder_t.elapsed_time,
                 'encoder_max_cuda_memory_allocated': encoder_m.max_memory_allocated,
                 'decoder_elapsed_time': decoder_t.elapsed_time,
                 'decoder_max_cuda_memory_allocated': decoder_m.max_memory_allocated}
            ]
        )
        return ret

    def compress(self, sparse_pc: ME.SparseTensor) -> Tuple[bytes, torch.Tensor]:
        strided_fea_list, points_num_list = self.encoder(sparse_pc)
        feature = strided_fea_list[-1]

        em_bytes, bottom_fea_recon, fea_recon = self.em_lossless_based.compress(feature, 1)
        assert bottom_fea_recon.C.shape[0] == 1
        sparse_tensor_coords_stride = bottom_fea_recon.tensor_stride[0]
        sparse_tensor_coords = bottom_fea_recon.C

        with io.BytesIO() as bs:
            if self.cfg.adaptive_pruning:
                bs.write(b''.join(
                    (_[0].to_bytes(3, 'little', signed=False) for _ in
                     points_num_list)
                ))
            bs.write(int(math.log2(sparse_tensor_coords_stride)).to_bytes(
                     1, 'little', signed=False))
            bs.write(b''.join(
                [_.to_bytes(1, 'little', signed=False)
                 for _ in (sparse_tensor_coords[0, 1:] // sparse_tensor_coords_stride).tolist()]
            ))
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

        concat_bytes = b''.join((len(s).to_bytes(3, 'little', signed=False) + s
                                 for s in compressed_bytes_list))
        return concat_bytes, sparse_tensor_coords_list

    def decompress(self, compressed_bytes: bytes, sparse_tensor_coords: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        with io.BytesIO(compressed_bytes) as bs:
            if self.cfg.adaptive_pruning:
                points_num_list = []
                for idx in range(1):
                    points_num_list.append([int.from_bytes(bs.read(3), 'little', signed=False)])
            else:
                points_num_list = None
            tensor_stride = 2 ** int.from_bytes(bs.read(1), 'little', signed=False)
            sparse_tensor_coords_bytes = bs.read(3)
            em_bytes = bs.read()

        fea_recon = self.em_lossless_based.decompress(
            em_bytes,
            self.get_sparse_pc(
                sparse_tensor_coords,
                tensor_stride=tensor_stride,
                only_return_coords=True
            ))

        decoder_fea = self.decoder(fea_recon, points_num_list)
        coord_recon = decoder_fea.C[:, 1:]
        color_recon_raw = decoder_fea.F
        color_recon = color_recon_raw.round_()

        return coord_recon, color_recon

    def decompress_partitions(self, concat_bytes: bytes,
                              sparse_tensor_coords_list: List[torch.Tensor]
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        color_recon_concat = torch.cat(color_recon_list, 0)
        return coord_recon_concat, color_recon_concat

    def train(self, mode: bool = True):
        """
        Use model.train() to reset evaluator.
        """
        if mode is True:
            self.evaluator.reset()
        return super(PCC, self).train(mode=mode)
