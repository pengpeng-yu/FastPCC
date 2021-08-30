from functools import partial
from typing import List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
import MinkowskiEngine as ME

from lib.torch_utils import MLPBlock
from lib.data_utils import PCData
from lib.evaluators import PCGCEvaluator
from lib.sparse_conv_layers import GenerativeUpsampleMessage
from lib.entropy_models.continuous_batched import NoisyDeepFactorizedEntropyModel
from lib.entropy_models.hyperprior import \
    NoisyDeepFactorizedHyperPriorScaleNoisyNormalEntropyModel, \
    NoisyDeepFactorizedHyperPriorNoisyDeepFactorizedEntropyModel

from models.convolutional.baseline.layers import Encoder, Decoder, ConvBlock, BLOCKS_DICT
from models.convolutional.baseline.model_config import ModelConfig


class PCC(nn.Module):
    params_divisions: List[Callable[[str], bool]] = [
        lambda s: 'entropy_bottleneck.' not in s and not s.endswith("aux_param"),
        lambda s: 'entropy_bottleneck.' in s and not s.endswith("aux_param"),
        lambda s: s.endswith("aux_param")
    ]

    def __init__(self, cfg: ModelConfig):
        super(PCC, self).__init__()
        ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

        self.evaluator = PCGCEvaluator(
            cfg.mpeg_pcc_error_command,
            cfg.mpeg_pcc_error_threads,
            cfg.chamfer_dist_test_phase
        )
        self.encoder = Encoder(1 if cfg.input_feature_type == 'Occupation' else 3,
                               cfg.compressed_channels,
                               cfg.encoder_channels,
                               cfg.basic_block_type,
                               cfg.basic_block_num,
                               cfg.use_batch_norm,
                               cfg.activation,
                               cfg.use_skip_connection,
                               cfg.skip_connection_channels)

        self.decoder = Decoder(cfg.compressed_channels,
                               cfg.decoder_channels,
                               cfg.basic_block_type,
                               cfg.basic_block_num,
                               cfg.use_batch_norm,
                               cfg.activation,
                               cfg.use_skip_connection,
                               cfg.skipped_fea_fusion_method,
                               cfg.skip_connection_channels,
                               loss_type='BCE' if cfg.reconstruct_loss_type == 'Focal'
                               else cfg.reconstruct_loss_type,
                               dist_upper_bound=cfg.dist_upper_bound)

        if cfg.hyperprior == 'None':
            prior_channels = cfg.compressed_channels
            if cfg.use_skip_connection:
                prior_channels += sum(cfg.skip_connection_channels)

            self.entropy_bottleneck = NoisyDeepFactorizedEntropyModel(
                batch_shape=torch.Size([prior_channels]),
                coding_ndim=2,
                init_scale=2)

        else:
            # BN is always performed for the last conv of hyper encoder and hyper decoder
            def make_hyper_coder(in_channels, intra_channels, out_channels):
                return nn.Sequential(
                    *[nn.Sequential(

                        ConvBlock(
                            intra_channels[idx - 1] if idx != 0 else in_channels,
                            intra_channels[idx],
                            3, 1, bn=cfg.use_batch_norm, act=cfg.activation
                        ),

                        *[BLOCKS_DICT[cfg.basic_block_type](
                            intra_channels[idx],
                            bn=cfg.use_batch_norm, act=cfg.activation
                        ) for _ in range(cfg.basic_block_num)]

                    ) for idx in range(len(intra_channels))],

                    ConvBlock(intra_channels[-1],
                              out_channels,
                              3, 1, bn=True, act=None)
                )

            hyper_decoder_out_channels = cfg.compressed_channels * len(cfg.prior_indexes_range)

            hyper_encoder = make_hyper_coder(
                cfg.compressed_channels,
                cfg.hyper_encoder_channels,
                cfg.hyper_compressed_channels)

            hyper_decoder = make_hyper_coder(
                cfg.hyper_compressed_channels,
                cfg.hyper_decoder_channels,
                hyper_decoder_out_channels,
            )

            if cfg.hyperprior == 'ScaleNoisyNormal':
                assert len(cfg.prior_indexes_range) == 1
                self.entropy_bottleneck = \
                    NoisyDeepFactorizedHyperPriorScaleNoisyNormalEntropyModel(
                        hyper_encoder=hyper_encoder,
                        hyper_decoder=hyper_decoder,
                        hyperprior_batch_shape=torch.Size([cfg.hyper_compressed_channels]),
                        coding_ndim=2,
                        num_scales=cfg.prior_indexes_range[0],
                        scale_min=0.11,
                        scale_max=64
                    )

            elif cfg.hyperprior == 'NoisyDeepFactorized':
                # No bn is performed.
                def parameter_fns_factory(in_channels, out_channels):
                    return nn.Sequential(
                        MLPBlock(in_channels, out_channels,
                                 bn=None, act=cfg.activation),
                        nn.Linear(out_channels, out_channels,
                                  bias=True)
                    )

                self.entropy_bottleneck = \
                    NoisyDeepFactorizedHyperPriorNoisyDeepFactorizedEntropyModel(
                        hyper_encoder=hyper_encoder,
                        hyper_decoder=hyper_decoder,
                        hyperprior_batch_shape=torch.Size([cfg.hyper_compressed_channels]),
                        coding_ndim=2,
                        index_ranges=cfg.prior_indexes_range,
                        parameter_fns_type='transform',
                        parameter_fns_factory=parameter_fns_factory,
                        num_filters=(1, 3, 3, 3, 1),
                        quantize_indexes=True
                    )

            else: raise NotImplementedError

        self.cfg = cfg
        self.init_parameters()

    def forward(self, pc_data: PCData):
        ME.clear_global_coordinate_manager()

        if self.cfg.input_feature_type == 'Occupation':
            sparse_pc = ME.SparseTensor(
                features=torch.ones(pc_data.xyz.shape[0], 1,
                                    dtype=torch.float,
                                    device=pc_data.xyz.device),
                coordinates=pc_data.xyz)

        elif self.cfg.input_feature_type == 'Coordinate':
            input_coords_scaler = torch.tensor(
                pc_data.resolution,
                device=pc_data.xyz.device)[pc_data.xyz[:, 0].type(torch.long), None]
            sparse_pc = ME.SparseTensor(
                features=pc_data.xyz[:, 1:].type(torch.float) / input_coords_scaler,
                coordinates=pc_data.xyz)

        else: raise NotImplementedError

        feature, cached_feature_list, points_num_list = self.encoder(sparse_pc)

        # points_num_list type: List[List[int]]
        if not self.cfg.adaptive_pruning:
            points_num_list = None
        else:
            points_num_list = [points_num_list[0]] + \
                              [[int(n * self.cfg.adaptive_pruning_num_scaler) for n in _]
                               for _ in points_num_list[1:]]

        if self.cfg.use_skip_connection:
            # feature = ME.cat(feature, *cached_feature_list)
            raise NotImplementedError

        if self.training:
            fea_tilde, loss_dict = self.entropy_bottleneck(feature)

            if self.cfg.use_skip_connection:
                # fea_tilde, *cached_feature_list = minkowski_tensor_split(
                #     fea_tilde, [self.cfg.compressed_channels,
                #                 *self.cfg.skip_connection_channels])
                raise NotImplementedError

            decoder_message = self.decoder(
                GenerativeUpsampleMessage(
                    fea=fea_tilde,
                    target_key=sparse_pc.coordinate_map_key,
                    points_num_list=points_num_list,
                    cached_fea_list=cached_feature_list))

            loss_dict['reconstruct_loss'] = self.get_reconstruct_loss(
                decoder_message.cached_pred_list,
                decoder_message.cached_target_list)

            loss_dict['bits_loss'] = loss_dict['bits_loss'] * \
                (self.cfg.bpp_loss_factor / sparse_pc.shape[0])

            if 'hyper_bits_loss' in loss_dict:
                loss_dict['hyper_bits_loss'] = loss_dict['hyper_bits_loss'] * \
                    (self.cfg.hyper_bpp_loss_factor / sparse_pc.shape[0])

            excluded_loss = {}
            if self.cfg.bpp_target != 0:
                raise NotImplementedError
            #     assert self.cfg.bpp_loss_factor == self.cfg.hyper_bpp_loss_factor == 1
            #     total_bpp_loss = loss_dict['bits_loss']
            #     excluded_loss['bits_loss'] = loss_dict['bits_loss'].detach().item()
            #
            #     if 'hyper_bits_loss' in loss_dict:
            #         total_bpp_loss += loss_dict['hyper_bits_loss']
            #         excluded_loss['hyper_bits_loss'] = loss_dict['hyper_bits_loss'].detach().item()
            #
            #     loss_dict['bpp_target_loss'] = F.l1_loss(
            #         total_bpp_loss,
            #         torch.tensor(self.cfg.bpp_target,
            #                      dtype=total_bpp_loss.dtype,
            #                      device=total_bpp_loss.device)
            #     )
            #
            #     del loss_dict['bits_loss'], loss_dict['hyper_bits_loss']

            loss_dict['loss'] = sum(loss_dict.values())
            for key in loss_dict:
                if key != 'loss':
                    loss_dict[key] = loss_dict[key].detach().item()

            loss_dict.update(excluded_loss)

            return loss_dict

        # Only supports batch size == 1.
        elif not self.training:
            fea_reconstructed, loss_dict, compressed_strings = self.entropy_bottleneck(feature)

            if self.cfg.use_skip_connection:
                # fea_reconstructed, *cached_feature_list = minkowski_tensor_split(
                #     fea_reconstructed, [self.cfg.compressed_channels,
                #                         *self.cfg.skip_connection_channels])
                raise NotImplementedError

            decoder_message = self.decoder(
                GenerativeUpsampleMessage(
                    fea=fea_reconstructed,
                    target_key=sparse_pc.coordinate_map_key,
                    points_num_list=points_num_list,
                    cached_fea_list=cached_feature_list))

            # the last one is supposed to be the final output of decoder
            pc_reconstructed = decoder_message.cached_pred_list[-1]

            ret = self.evaluator.log_batch(
                preds=pc_reconstructed.decomposed_coordinates,
                targets=sparse_pc.decomposed_coordinates,
                compressed_strings=compressed_strings,
                bits_preds=[loss_dict['bits_loss'].item() +
                            loss_dict.get('hyper_bits_loss', torch.tensor([0])).item()],
                feature_points_numbers=[_.shape[0] for _ in feature.decomposed_coordinates],
                pc_data=pc_data
            )

            return ret

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

        if self.cfg.reconstruct_loss_factor != 1:
            reconstruct_loss *= self.cfg.reconstruct_loss_factor

        return reconstruct_loss

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (ME.MinkowskiConvolution,
                              ME.MinkowskiGenerativeConvolutionTranspose)):
                torch.nn.init.normal_(m.kernel, 0, 0.08)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        for m in self.modules():
            if not isinstance(m, PCC) and hasattr(m, 'init_parameters'):
                m.init_parameters()

    def train(self, mode: bool = True):
        """
        Use model.train() to reset evaluator.
        """
        if mode is True:
            self.evaluator.reset()
        return super(PCC, self).train(mode=mode)


def main_debug():
    ME.clear_global_coordinate_manager()
    ME.set_sparse_tensor_operation_mode(
        ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

    coords, feats = ME.utils.sparse_collate(
        coords=[torch.tensor([[3, 5, 7], [8, 9, 1], [-2, 3, 3], [-2, 3, 4]],
                             dtype=torch.int32)],
        feats=[torch.tensor([[0.5, 0.4], [0.1, 0.6], [-0.9, 10], [-0.9, 8]])])
    batch_coords = torch.tensor(
        [[0, 3, 5, 7], [0, 8, 9, 1], [0, -2, 3, 2], [0, -2, 3, 4]],
        dtype=torch.int32)

    pc = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1)
    cm = ME.global_coordinate_manager()
    coord_map_key = cm.insert_and_map(batch_coords, tensor_stride=1)[0]

    print(cm.kernel_map(pc.coordinate_map_key, coord_map_key, kernel_size=1))

    conv11 = ME.MinkowskiConvolution(2, 2, 1, 1, dimension=3)
    conv_trans22 = ME.MinkowskiConvolutionTranspose(2, 6, 2, 2, dimension=3)

    pc = conv11(pc)
    m_out_trans = conv_trans22(pc)
    pc = pc.dense(shape=torch.Size([1, 3, 10, 10, 10]),
                  min_coordinate=torch.IntTensor([0, 0, 0]))

    print('Done')


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
    # main_debug()
    model_debug()
