from typing import List, Union, Tuple, Generator, Iterator
import os
import time
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensorQuantizationMode

from lib.mpeg_gpcc_utils import gpcc_octree_lossless_geom_encode, gpcc_decode
from lib.torch_utils import MLPBlock
from lib.data_utils import PCData, write_ply_file
from lib.evaluators import PCGCEvaluator
from lib.entropy_models.continuous_batched import \
    NoisyDeepFactorizedEntropyModel as PriorEM

from ..lossy_coord.generative_upsample import GenerativeUpsampleMessage
from ..lossy_coord.layers import \
    Encoder, Decoder, HyperDecoderUpsample, EncoderRecurrent
from .model_config import ModelConfig
from .geo_lossl_em import GeoLosslessNoisyDeepFactorizedEntropyModel


class PCC(nn.Module):

    @staticmethod
    def params_divider(s: str) -> int:
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
            cfg.recurrent_part_channels,
            cfg.encoder_channels,
            cfg.first_conv_kernel_size,
            cfg.adaptive_pruning,
            cfg.adaptive_pruning_num_scaler,
            *self.basic_block_args,
            cfg.activation
        )
        self.decoder = Decoder(
            cfg.compressed_channels,
            cfg.decoder_channels,
            *self.basic_block_args,
            loss_type=cfg.coord_recon_loss_type,
            dist_upper_bound=cfg.dist_upper_bound
        )
        enc_lossl = self.init_enc_rec()
        hyper_dec_fea = self.init_hyper_dec_up()

        self.em_lossless_based = self.init_em_lossless_based(
            self.init_em(), enc_lossl,
            hyper_dec_fea, parameter_fns_factory
        )

    def init_em(self) -> nn.Module:
        em = PriorEM(
            batch_shape=torch.Size([self.cfg.compressed_channels]),
            coding_ndim=2,
            bottleneck_process=self.cfg.bottleneck_process,
            bottleneck_scaler=self.cfg.bottleneck_scaler,
            init_scale=2 / self.cfg.bottleneck_scaler,
            broadcast_shape_bytes=(0,),
        )
        return em

    def init_em_lossless_based(
            self, bottom_fea_entropy_model, encoder_geo_lossless,
            hyper_decoder_fea_geo_lossless, parameter_fns_factory
    ):
        em_lossless_based = GeoLosslessNoisyDeepFactorizedEntropyModel(
            bottom_fea_entropy_model=bottom_fea_entropy_model,
            encoder=encoder_geo_lossless,
            hyper_decoder_fea=hyper_decoder_fea_geo_lossless,
            hybrid_hyper_decoder_fea=self.cfg.hybrid_hyper_decoder_fea,
            fea_index_ranges=self.cfg.prior_indexes_range,
            fea_parameter_fns_type='transform',
            fea_parameter_fns_factory=parameter_fns_factory,
            fea_num_filters=self.cfg.lossless_fea_num_filters,
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
                return self.test_partitions_forward([sparse_pc], pc_data)
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

        bottleneck_feature, loss_dict = self.em_lossless_based(feature, batch_size)
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

    def test_forward(self, sparse_pc: ME.SparseTensor):
        strided_fea_list, points_num_list = self.encoder(sparse_pc)
        feature = strided_fea_list[-1]

        bottleneck_feature, em_bytes = self.em_lossless_based(feature, 1)
        decoder_message: GenerativeUpsampleMessage = self.decoder(
            GenerativeUpsampleMessage(
                fea=bottleneck_feature,
                target_key=sparse_pc.coordinate_map_key,
                max_stride_lossy_recon=[2 ** len(self.cfg.decoder_channels)] * 3,
                points_num_list=points_num_list
            )
        )

        h = hashlib.md5()
        h.update(str(time.time()).encode())
        tmp_file_path = 'tmp-' + h.hexdigest()
        write_ply_file(feature.C[:, 1:] // feature.tensor_stride[0], f'{tmp_file_path}.ply')
        gpcc_octree_lossless_geom_encode(
            f'{tmp_file_path}.ply', f'{tmp_file_path}.bin',
            command=self.cfg.mpeg_gpcc_command
        )
        with open(f'{tmp_file_path}.bin', 'rb') as f:
            sparse_tensor_coords_bytes = f.read()
        os.remove(f'{tmp_file_path}.ply')
        os.remove(f'{tmp_file_path}.bin')
        em_bytes += len(sparse_tensor_coords_bytes).to_bytes(3, 'little', signed=False) + \
            sparse_tensor_coords_bytes

        return decoder_message.fea.C[:, 1:], em_bytes

    def test_partitions_forward(self, sparse_pc_partitions: Iterator, pc_data: PCData):
        coord_recon_list = []
        compressed_bytes_list = []
        for sparse_pc in sparse_pc_partitions:
            coord_recon, compressed_bytes = self.test_forward(sparse_pc)
            ME.clear_global_coordinate_manager()
            coord_recon_list.append(coord_recon)
            compressed_bytes_list.append(compressed_bytes)
        concat_bytes = b''.join((len(s).to_bytes(3, 'little', signed=False) + s
                                 for s in compressed_bytes_list))
        ret = self.evaluator.log_batch(
            preds=[torch.cat(coord_recon_list, 0)],
            targets=[pc_data.xyz[0] if isinstance(pc_data.xyz, List) else pc_data.xyz[:, 1:]],
            compressed_bytes_list=[concat_bytes],
            pc_data=pc_data
        )
        return ret

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

    def train(self, mode: bool = True):
        """
        Use model.train() to reset evaluator.
        """
        if mode is True:
            self.evaluator.reset()
        return super(PCC, self).train(mode=mode)
