import io
from typing import Union, Tuple

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensorQuantizationMode

from lib.torch_utils import concat_loss_dicts
from lib.data_utils import PCData
from lib.evaluators import PCCEvaluator

from .geo_lossl_em import GeoLosslessEntropyModel
from ..lossy_coord_v2.layers import Encoder, Decoder, \
    HyperDecoderUpsample, EncoderGeoLossl, \
    ResidualGeoLossl, DecoderGeoLossl
from ..lossy_coord_v2.model_config import ModelConfig


class PCC(nn.Module):

    @staticmethod
    def params_divider(s: str) -> int:
        if 'em_lossless_based' in s:
            if 'bottom_fea_entropy_model' in s:
                return 2
            else:
                return 1
        else:
            return 0

    def __init__(self, cfg: ModelConfig):
        super(PCC, self).__init__()
        self.cfg = cfg
        ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
        self.minkowski_algorithm = getattr(ME.MinkowskiAlgorithm, cfg.minkowski_algorithm)
        self.evaluator = PCCEvaluator()
        assert len(cfg.compressed_channels) == len(cfg.geo_lossl_channels)
        assert len(cfg.geo_lossl_if_sample) == len(cfg.geo_lossl_channels) - 1
        assert cfg.compressed_channels[-1] == cfg.geo_lossl_channels[-1]

        self.encoder = Encoder(
            1,
            cfg.encoder_channels,
            cfg.adaptive_pruning,
            cfg.adaptive_pruning_scaler,
            cfg.conv_region_type,
            cfg.activation
        )
        self.decoder = Decoder(
            cfg.geo_lossl_channels[0],
            cfg.decoder_channels,
            cfg.conv_region_type,
            cfg.activation
        )
        enc_lossl = EncoderGeoLossl(
            cfg.geo_lossl_channels[:-1],
            cfg.geo_lossl_channels,
            cfg.geo_lossl_if_sample,
            cfg.conv_region_type,
            cfg.activation,
            cfg.bottleneck_value_bound,
            cfg.skip_encoding_fea
        )
        hyper_dec_fea = HyperDecoderUpsample(
            cfg.geo_lossl_channels[1:],
            cfg.geo_lossl_channels[:-1],
            cfg.geo_lossl_if_sample,
            cfg.conv_region_type,
            cfg.activation
        )
        self.em_lossless_based = self.init_em_lossless_based(
            enc_lossl,
            ResidualGeoLossl(
                cfg.geo_lossl_channels[:-1],
                cfg.compressed_channels[:-1],
                cfg.conv_region_type, cfg.activation,
                cfg.bottleneck_value_bound,
                cfg.skip_encoding_fea
            ),
            DecoderGeoLossl(
                cfg.compressed_channels[:-1],
                cfg.geo_lossl_channels[:-1],
                cfg.geo_lossl_channels[:-1],
                cfg.conv_region_type,
                cfg.activation,
                cfg.skip_encoding_fea
            ),
            hyper_dec_fea
        )
        self.linear_warmup_fea_step = (self.cfg.warmup_fea_loss_factor -
                                       self.cfg.bits_loss_factor) / self.cfg.warmup_fea_loss_steps

    def init_em_lossless_based(
            self, encoder_geo_lossless, residual_block, decoder_block,
            hyper_decoder_fea_geo_lossless,
    ):
        em_lossless_based = GeoLosslessEntropyModel(
            self.cfg.compressed_channels[0],
            self.cfg.bottleneck_process,
            self.cfg.bottleneck_scaler,
            self.cfg.skip_encoding_fea,
            encoder=encoder_geo_lossless,
            residual_block=residual_block,
            decoder_block=decoder_block,
            hyper_decoder_fea=hyper_decoder_fea_geo_lossless
        )
        return em_lossless_based

    def forward(self, pc_data: PCData):
        if self.training:
            return self.train_forward(pc_data.xyz, pc_data.training_step, pc_data.batch_size)
        else:
            assert pc_data.batch_size == 1, 'Only supports batch size == 1 during testing.'
            if isinstance(pc_data.xyz, torch.Tensor):
                return self.test_forward(pc_data)
            else:
                raise NotImplementedError

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

    def get_sparse_pc(self, xyz: torch.Tensor) -> ME.SparseTensor:
        global_coord_mg = self.set_global_cm()
        sparse_pc_feature = torch.full(
            (xyz.shape[0], 1), fill_value=1,
            dtype=torch.float, device=xyz.device
        )
        sparse_pc = ME.SparseTensor(
            features=sparse_pc_feature,
            coordinates=xyz,
            tensor_stride=[1] * 3,
            coordinate_manager=global_coord_mg,
            quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        )
        return sparse_pc

    def train_forward(self, batched_coord: torch.Tensor, training_step: int, batch_size: int):
        sparse_pc = self.get_sparse_pc(batched_coord)
        feature, points_num_list = self.encoder(sparse_pc)

        bottleneck_feature, loss_dict = self.em_lossless_based(feature, batch_size)

        decoder_loss_dict = self.decoder(
            bottleneck_feature, points_num_list,
            sparse_pc.coordinate_map_key
        )
        concat_loss_dicts(loss_dict, decoder_loss_dict)

        self.warmup_forward = False
        if training_step < self.cfg.warmup_fea_loss_steps:
            self.warmup_forward = True
            if self.cfg.linear_warmup:
                fea_loss_factor = self.cfg.warmup_fea_loss_factor - \
                    self.linear_warmup_fea_step * training_step
            else:
                fea_loss_factor = self.cfg.warmup_fea_loss_factor
        else:
            fea_loss_factor = self.cfg.bits_loss_factor

        for key in loss_dict:
            if key.endswith('bits_loss'):
                if 'fea' in key:
                    loss_dict[key] *= fea_loss_factor
                else:
                    loss_dict[key] *= self.cfg.bits_loss_factor
            if key.startswith('coord_recon_loss'):
                loss_dict[key] *= self.cfg.coord_recon_loss_factor

        loss_dict['loss'] = sum(loss_dict.values())
        for key in loss_dict:
            if key != 'loss':
                loss_dict[key] = loss_dict[key].item()
        return loss_dict

    def test_forward(self, pc_data: PCData):
        feature, points_num_list = self.encoder(self.get_sparse_pc(pc_data.xyz))
        feature_recon, em_bytes = self.em_lossless_based(feature, 1)
        decoder_fea = self.decoder(feature_recon, points_num_list)
        coord_recon = decoder_fea.C[:, 1:]

        with io.BytesIO() as bs:
            if self.cfg.adaptive_pruning:
                bs.write(b''.join(
                    (_[0].to_bytes(3, 'little', signed=False) for _ in points_num_list)
                ))
            bs.write(em_bytes)
            compressed_bytes = bs.getvalue()

        ret = self.evaluator.log_batch(
            preds=[coord_recon],
            targets=[pc_data.xyz[:, 1:]],
            compressed_bytes_list=[compressed_bytes],
            pc_data=pc_data
        )
        return ret

    def train(self, mode: bool = True):
        """
        Use model.train() to reset evaluator.
        """
        if mode is True:
            self.evaluator.reset()
        return super(PCC, self).train(mode=mode)
