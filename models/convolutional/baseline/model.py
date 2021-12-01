import io
from functools import partial, reduce
from typing import List, Callable, Union, Tuple, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
import MinkowskiEngine as ME

from lib.torch_utils import MLPBlock
from lib.data_utils import PCData
from lib.evaluators import PCGCEvaluator
from lib.sparse_conv_layers import GenerativeUpsampleMessage
from lib.entropy_models.continuous_batched import \
    NoisyDeepFactorizedEntropyModel as NoisyDeepFactorizedPriorEntropyModel
from lib.entropy_models.hyperprior.noisy_deep_factorized import \
    ScaleNoisyNormalEntropyModel as HyperPriorScaleNoisyNormalEntropyModel, \
    NoisyDeepFactorizedEntropyModel as HyperPriorNoisyDeepFactorizedEntropyModel
from lib.entropy_models.hyperprior.sparse_tensor_specialized.noisy_deep_factorized import \
    GeoLosslessNoisyDeepFactorizedEntropyModel

from models.convolutional.baseline.layers import \
    Encoder, Decoder, \
    HyperEncoder, HyperDecoder, \
    HyperDecoderCoordForGeoLossLess, HyperDecoderFeaForGeoLossLess, \
    EncoderForGeoLossLess
from models.convolutional.baseline.model_config import ModelConfig


class PCC(nn.Module):

    def params_divider(self, s: str) -> int:
        if not self.cfg.lossless_compression_based:
            if s.endswith("aux_param"): return 2

            else:
                if 'entropy_bottleneck' not in s: return 0
                else: return 1

        else:
            if s.endswith("aux_param"): return 2

            else:
                if 'entropy_bottleneck' not in s: return 0

                else:
                    if '.encoder' in s or '.decoder' in s: return 0
                    else: return 1

    def __init__(self, cfg: ModelConfig):
        super(PCC, self).__init__()
        ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

        self.minkowski_algorithm = getattr(ME.MinkowskiAlgorithm, cfg.minkowski_algorithm)

        self.evaluator = PCGCEvaluator(
            cfg.mpeg_pcc_error_command,
            cfg.mpeg_pcc_error_threads,
            cfg.chamfer_dist_test_phase
        )

        basic_block_args = (
            cfg.basic_block_type,
            cfg.conv_region_type,
            cfg.basic_block_num,
            cfg.use_batch_norm,
            cfg.activation
        )

        self.encoder = Encoder(
            1 if cfg.input_feature_type == 'Occupation' else 3,
            cfg.compressed_channels if not cfg.lossless_compression_based
            else cfg.lossless_coder_channels,
            cfg.encoder_channels,
            cfg.adaptive_pruning,
            cfg.adaptive_pruning_num_scaler,
            cfg.encoder_scaler if not cfg.lossless_compression_based else 1,
            *basic_block_args,
            None if not cfg.lossless_compression_based else cfg.activation
        )

        self.decoder = Decoder(
            cfg.compressed_channels,
            cfg.decoder_channels,
            cfg.conv_trans_near_pruning,
            1 / cfg.encoder_scaler,
            *basic_block_args,
            loss_type='BCE' if cfg.reconstruct_loss_type == 'Focal'
            else cfg.reconstruct_loss_type,
            dist_upper_bound=cfg.dist_upper_bound
        )

        if cfg.hyperprior == 'None':
            entropy_bottleneck = NoisyDeepFactorizedPriorEntropyModel(
                batch_shape=torch.Size([cfg.compressed_channels]),
                coding_ndim=2,
                init_scale=2)

        else:
            hyper_encoder = HyperEncoder(
                1 / cfg.encoder_scaler,
                cfg.hyper_encoder_scaler,
                cfg.compressed_channels,
                cfg.hyper_compressed_channels,
                cfg.hyper_encoder_channels,
                *basic_block_args
            )

            hyper_decoder = HyperDecoder(
                1 / cfg.hyper_encoder_scaler,
                cfg.prior_indexes_scaler,
                cfg.hyper_compressed_channels,
                cfg.compressed_channels * len(cfg.prior_indexes_range),
                cfg.hyper_decoder_channels,
                *basic_block_args
            )

            if cfg.hyperprior == 'ScaleNoisyNormal':
                assert len(cfg.prior_indexes_range) == 1
                entropy_bottleneck = HyperPriorScaleNoisyNormalEntropyModel(
                        hyper_encoder=hyper_encoder,
                        hyper_decoder=hyper_decoder,
                        hyperprior_batch_shape=torch.Size([cfg.hyper_compressed_channels]),
                        prior_bytes_num_bytes=4,
                        coding_ndim=2,
                        num_scales=cfg.prior_indexes_range[0],
                        scale_min=0.11,
                        scale_max=64
                    )

            elif cfg.hyperprior == 'NoisyDeepFactorized':
                def parameter_fns_factory(in_channels, out_channels):
                    return nn.Sequential(
                        MLPBlock(in_channels, out_channels,
                                 bn=None, act=cfg.activation),
                        nn.Linear(out_channels, out_channels,
                                  bias=True)
                    )

                entropy_bottleneck = HyperPriorNoisyDeepFactorizedEntropyModel(
                        hyper_encoder=hyper_encoder,
                        hyper_decoder=hyper_decoder,
                        hyperprior_batch_shape=torch.Size([cfg.hyper_compressed_channels]),
                        hyperprior_broadcast_shape_bytes=(4,),  # TODO: lower?
                        prior_bytes_num_bytes=4,
                        coding_ndim=2,
                        index_ranges=cfg.prior_indexes_range,
                        parameter_fns_type='transform',
                        parameter_fns_factory=parameter_fns_factory,
                        num_filters=(1, 3, 3, 3, 1),
                        quantize_indexes=True
                    )

            else: raise NotImplementedError

        if cfg.lossless_compression_based:
            encoder_geo_lossless = EncoderForGeoLossLess(
                cfg.lossless_coder_channels,
                cfg.compressed_channels,
                cfg.lossless_coder_num,
                cfg.encoder_scaler,
                cfg.lossless_detach_higher_fea,
                *basic_block_args
            )
            hyper_decoder_coord_geo_lossless = HyperDecoderCoordForGeoLossLess(
                1 / cfg.encoder_scaler,
                cfg.prior_indexes_scaler,
                cfg.hyper_compressed_channels,
                1 * len(cfg.lossless_coord_indexes_range),
                (cfg.lossless_coder_channels,),
                *basic_block_args
            )
            hyper_decoder_fea_geo_lossless = HyperDecoderFeaForGeoLossLess(
                1 / cfg.encoder_scaler,
                cfg.prior_indexes_scaler,
                cfg.hyper_compressed_channels,
                cfg.compressed_channels * (
                    len(cfg.prior_indexes_range)
                    if not cfg.lossless_hybrid_hyper_decoder_fea
                    else len(cfg.prior_indexes_range) + 1
                ),
                (cfg.lossless_coder_channels,),
                *basic_block_args
            )

            def parameter_fns_factory(in_channels, out_channels):
                return nn.Sequential(
                    MLPBlock(in_channels, out_channels,
                             bn=None, act=cfg.activation),
                    nn.Linear(out_channels, out_channels,
                              bias=True)
                )

            self.entropy_bottleneck = GeoLosslessNoisyDeepFactorizedEntropyModel(
                bottom_fea_entropy_model=entropy_bottleneck,
                encoder=encoder_geo_lossless,
                detach_higher_fea=cfg.lossless_detach_higher_fea,
                hyper_decoder_coord=hyper_decoder_coord_geo_lossless,
                hyper_decoder_fea=hyper_decoder_fea_geo_lossless,
                hybrid_hyper_decoder_fea=cfg.lossless_hybrid_hyper_decoder_fea,
                fea_bytes_num_bytes=4,  # TODO: lower?
                coord_bytes_num_bytes=4,
                coord_index_ranges=cfg.lossless_coord_indexes_range,
                coord_parameter_fns_type='transform',
                coord_parameter_fns_factory=parameter_fns_factory,
                coord_num_filters=(1, 3, 3, 3, 1),
                fea_index_ranges=cfg.prior_indexes_range,
                fea_parameter_fns_type='transform',
                fea_parameter_fns_factory=parameter_fns_factory,
                fea_num_filters=(1, 3, 3, 3, 3, 1),
                quantize_indexes=True
            )

        else:
            self.entropy_bottleneck = entropy_bottleneck

        self.cfg = cfg

    def forward(self, pc_data: PCData):
        if self.training:
            sparse_pc = self.get_sparse_pc(pc_data.xyz)
            return self.train_forward(sparse_pc, pc_data.training_step)

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

        pc_coord_key = global_coord_mg.insert_and_map(xyz, [tensor_stride] * 3)[0]

        if only_return_coords:
            return pc_coord_key, global_coord_mg

        else:
            if self.cfg.input_feature_type == 'Occupation':
                sparse_pc_feature = torch.ones(
                    xyz.shape[0], 1,
                    dtype=torch.float,
                    device=xyz.device
                )

            else:
                raise NotImplementedError

            sparse_pc = ME.SparseTensor(
                features=sparse_pc_feature,
                coordinate_map_key=pc_coord_key,
                coordinate_manager=global_coord_mg
            )
            return sparse_pc

    def get_sparse_pc_partitions(self, xyz: List[torch.Tensor]) -> Generator:
        # The first one is supposed to be the original coordinates.
        for sub_xyz in xyz[1:]:
            yield self.get_sparse_pc(sub_xyz)

    def train_forward(self, sparse_pc: ME.SparseTensor, training_step: int):
        warmup_forward = training_step < self.cfg.warmup_steps

        encoder_feature, cached_encoder_fea_list, points_num_list = self.encoder(sparse_pc)
        bottleneck_feature, loss_dict = self.entropy_bottleneck(encoder_feature)

        for key in loss_dict:
            if key.endswith('bits_loss'):
                loss_dict[key] = loss_dict[key] * (
                        (self.cfg.warmup_bpp_loss_factor if warmup_forward
                         else self.cfg.bpp_loss_factor) / sparse_pc.shape[0]
                )

        decoder_message = self.decoder(
            GenerativeUpsampleMessage(
                fea=bottleneck_feature,
                target_key=sparse_pc.coordinate_map_key,
                points_num_list=points_num_list
            )
        )

        loss_dict[f'reconstruct_loss'] = self.get_reconstruct_loss(
            decoder_message.cached_pred_list,
            decoder_message.cached_target_list
        )

        loss_dict['loss'] = sum(loss_dict.values())
        for key in loss_dict:
            if key != 'loss':
                loss_dict[key] = loss_dict[key].item()

        return loss_dict

    def test_forward(self, sparse_pc: ME.SparseTensor, pc_data: PCData):
        compressed_string, sparse_tensor_coords = self.compress(sparse_pc)

        del sparse_pc
        ME.clear_global_coordinate_manager()
        torch.cuda.empty_cache()

        pc_recon = self.decompress(compressed_string, sparse_tensor_coords)

        ret = self.evaluator.log_batch(
            preds=[pc_recon],
            targets=[pc_data.xyz[:, 1:]],
            compressed_strings=[compressed_string],
            pc_data=pc_data
        )

        return ret

    def test_partitions_forward(self, sparse_pc_partitions: Generator, pc_data: PCData):
        compressed_string, sparse_tensor_coords_list = self.compress_partitions(sparse_pc_partitions)
        pc_recon = self.decompress_partitions(compressed_string, sparse_tensor_coords_list)

        ret = self.evaluator.log_batch(
            preds=[pc_recon],
            targets=[pc_data.xyz[0]],
            compressed_strings=[compressed_string],
            pc_data=pc_data
        )

        return ret

    def compress(self, sparse_pc: ME.SparseTensor) -> Tuple[bytes, torch.Tensor]:
        feature, cached_feature_list, points_num_list = self.encoder(sparse_pc)

        if self.cfg.lossless_compression_based:
            em_string, bottom_rounded_fea, _ = \
                self.entropy_bottleneck.compress(feature)

            sparse_tensor_coords = bottom_rounded_fea.C

        else:
            em_strings, coding_batch_shape, _ = \
                self.entropy_bottleneck.compress(feature)
            assert coding_batch_shape == torch.Size([1])

            sparse_tensor_coords = feature.C

            em_string = em_strings[0]

        with io.BytesIO() as bs:
            if self.cfg.adaptive_pruning:
                bs.write(reduce(
                    lambda i, j: i + j,
                    [_[0].to_bytes(4, 'little', signed=False)
                     for _ in points_num_list]
                ))

            bs.write(len(em_string).to_bytes(4, 'little', signed=False))
            bs.write(em_string)

            # TODO: sparse_tensor_coords_list -> strings

            compressed_string = bs.getvalue()

        return compressed_string, sparse_tensor_coords

    def compress_partitions(self, sparse_pc_partitions: Generator) \
            -> Tuple[bytes, List[torch.Tensor]]:
        compressed_string_list = []
        sparse_tensor_coords_list = []

        for sparse_pc in sparse_pc_partitions:
            compressed_string, sparse_tensor_coords = self.compress(sparse_pc)

            del sparse_pc
            ME.clear_global_coordinate_manager()
            torch.cuda.empty_cache()

            compressed_string_list.append(compressed_string)
            sparse_tensor_coords_list.append(sparse_tensor_coords)

        # Log bytes of each partitions.
        concat_string = reduce(lambda i, j: i + j,
                               (len(s).to_bytes(4, 'little', signed=False) + s
                                for s in compressed_string_list))

        return concat_string, sparse_tensor_coords_list

    def decompress(self, compressed_string: bytes, sparse_tensor_coords: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device

        with io.BytesIO(compressed_string) as bs:
            if self.cfg.adaptive_pruning:
                points_num_list = []
                for idx in range(len(self.cfg.decoder_channels)):
                    points_num_list.append([int.from_bytes(bs.read(4), 'little', signed=False)])
            else:
                points_num_list = None

            em_string_len = int.from_bytes(bs.read(4), 'little', signed=False)
            em_string = bs.read(em_string_len)

        if self.cfg.lossless_compression_based:
            assert isinstance(self.entropy_bottleneck,
                              GeoLosslessNoisyDeepFactorizedEntropyModel)
            fea_recon = self.entropy_bottleneck.decompress(
                em_string,
                device,
                self.get_sparse_pc(
                    sparse_tensor_coords,
                    tensor_stride=2 ** (
                            len(self.cfg.decoder_channels)
                            + self.cfg.lossless_coder_num),
                    only_return_coords=True),
                self.cfg.lossless_coder_num
            )

        else:
            fea_recon = self.entropy_bottleneck.decompress(
                [em_string],
                torch.Size([1]),
                device,
                sparse_tensor_coords_tuple=self.get_sparse_pc(
                    sparse_tensor_coords,
                    tensor_stride=2 ** len(self.cfg.decoder_channels),
                    only_return_coords=True
                )
            )

        decoder_message = self.decoder(
            GenerativeUpsampleMessage(
                fea=fea_recon,
                points_num_list=points_num_list
            )
        )

        # The last one is supposed to be the final output of the decoder.
        pc_recon = decoder_message.cached_pred_list[-1].C[:, 1:]
        return pc_recon

    def decompress_partitions(self, concat_string: bytes,
                              sparse_tensor_coords_list: List[torch.Tensor]) \
            -> torch.Tensor:
        pc_recon_list = []
        concat_string_len = len(concat_string)

        with io.BytesIO(concat_string) as bs:
            while bs.tell() != concat_string_len:
                length = int.from_bytes(bs.read(4), 'little', signed=False)
                pc_recon_list.append(self.decompress(
                    bs.read(length), sparse_tensor_coords_list.pop(0)
                ))

                ME.clear_global_coordinate_manager()
                torch.cuda.empty_cache()

        pc_recon = torch.cat([pc_recon for pc_recon in pc_recon_list], 0)

        return pc_recon

    def get_reconstruct_loss(self, cached_pred_list, cached_target_list):
        if self.cfg.reconstruct_loss_type == 'BCE':
            loss_func = F.binary_cross_entropy_with_logits
        elif self.cfg.reconstruct_loss_type == 'Focal':
            loss_func = partial(sigmoid_focal_loss,
                                alpha=0.25,
                                gamma=2.0,
                                reduction='mean')
        elif self.cfg.reconstruct_loss_type == 'Dist':
            loss_func = F.smooth_l1_loss
        else:
            raise NotImplementedError

        reconstruct_loss_list = [loss_func(
                pred.F.squeeze(dim=1),
                target.type(pred.F.dtype))
             for pred, target in zip(cached_pred_list, cached_target_list)]

        reconstruct_loss = sum(reconstruct_loss_list) / len(reconstruct_loss_list)

        factor = self.cfg.reconstruct_loss_factor
        if factor != 1:
            reconstruct_loss *= factor

        return reconstruct_loss

    def train(self, mode: bool = True):
        """
        Use model.train() to reset evaluator.
        """
        if mode is True:
            self.evaluator.reset()
        return super(PCC, self).train(mode=mode)


def model_debug():
    cfg = ModelConfig()
    cfg.resolution = 128
    model = PCC(cfg).cuda()
    xyz_c = [ME.utils.sparse_quantize(torch.randint(0, 128, (100, 3))) for _ in range(16)]
    xyz = ME.utils.batched_coordinates(xyz_c).cuda()
    out = model(PCData(xyz=xyz, resolution=[128] * 16))
    out['loss'].backward()
    model.eval()
    with torch.no_grad():
        test_out = model(PCData(xyz=xyz[xyz[:, 0] == 0, :], file_path=[''] * 16,
                                ori_resolution=[0] * 16, resolution=[128] * 16))
    print('Done')


if __name__ == '__main__':
    model_debug()
