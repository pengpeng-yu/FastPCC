import os
from collections import defaultdict
from typing import List

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from compressai.models.utils import update_registered_buffers

from lib.torch_utils import MLPBlock
from lib.data_utils import PCData
from lib.metric import mpeg_pc_error
from lib.loss_function import chamfer_loss
from lib.sparse_conv_layers import GenerativeUpsampleMessage
from lib.entropy_models.continuous_batched import NoisyDeepFactorizedEntropyModel
from models.convolutional.PCGCv2.layers import Encoder, DecoderBlock
from models.convolutional.PCGCv2.model_config import ModelConfig


class PCC(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(PCC, self).__init__()
        ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
        self.encoder = Encoder(1 if cfg.input_feature_type == 'Occupation' else 3,
                               cfg.compressed_channels,
                               res_blocks_num=3,
                               res_block_type=cfg.res_block_type)

        self.decoder = nn.Sequential(
            DecoderBlock(cfg.compressed_channels,
                         64, 3, cfg.res_block_type,
                         loss_type=cfg.reconstruct_loss_type,
                         dist_upper_bound=cfg.dist_upper_bound),

            DecoderBlock(64, 32, 3, cfg.res_block_type,
                         loss_type=cfg.reconstruct_loss_type,
                         dist_upper_bound=cfg.dist_upper_bound),

            DecoderBlock(32, 16, 3, cfg.res_block_type,
                         loss_type=cfg.reconstruct_loss_type,
                         dist_upper_bound=cfg.dist_upper_bound,
                         is_last_layer=True))

        self.entropy_bottleneck = NoisyDeepFactorizedEntropyModel(
            prior_batch_shape=torch.Size([cfg.compressed_channels]),
            coding_ndim=2,
            init_scale=5)

        self.cfg = cfg
        self.log_pred_res('init')
        self.init_parameters()

    def log_pred_res(self, mode, preds=None, targets=None,
                     compressed_strings: List[bytes] = None,
                     fea_points_num: int = None,
                     pc_data: PCData = None,
                     bits_preds: List[float] = None):
        if mode == 'init' or mode == 'reset':
            self.total_reconstruct_loss = 0.0
            self.total_bpp = 0.0
            self.samples_num = 0
            self.total_metric_values = defaultdict(float)
            self.total_bpp_pred = 0.0

        elif mode == 'log':
            assert not self.training
            assert isinstance(preds, list) and isinstance(targets, list)

            if len(preds) != 1: raise NotImplementedError

            for pred, target, file_path, ori_resolution, resolution, bits_pred \
                    in zip(preds, targets, pc_data.file_path,
                           pc_data.ori_resolution, pc_data.resolution, bits_preds):
                com_string = compressed_strings[0]

                if self.cfg.chamfer_dist_test_phase is True:
                    self.total_reconstruct_loss += chamfer_loss(
                        pred.unsqueeze(0).type(torch.float) / resolution,
                        target.unsqueeze(0).type(torch.float) / resolution).item()

                bpp = len(com_string) * 8 / target.shape[0]
                self.total_bpp += bpp

                self.total_bpp_pred += bits_pred / target.shape[0]

                if hasattr(pc_data, 'results_dir'):
                    out_file_path = os.path.join(pc_data.results_dir, os.path.splitext(file_path)[0])
                    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
                    compressed_path = out_file_path + '.txt'
                    fileinfo_path = out_file_path + '_info.txt'
                    reconstructed_path = out_file_path + '_recon.ply'

                    if not file_path.endswith('.ply') or ori_resolution != resolution:
                        file_path = out_file_path + '.ply'
                        o3d.io.write_point_cloud(file_path,
                                                 o3d.geometry.PointCloud(
                                                     o3d.utility.Vector3dVector(
                                                         target.cpu())), write_ascii=True)

                    with open(compressed_path, 'wb') as f:
                        f.write(com_string)

                    fileinfo = f'fea_points_num: {fea_points_num}\n' \
                               f'\n' \
                               f'input_points_num: {target.shape[0]}\n' \
                               f'output_points_num: {pred.shape[0]}\n' \
                               f'compressed_bytes: {len(com_string)}\n' \
                               f'bpp: {bpp}\n' \
                               f'\n'

                    o3d.io.write_point_cloud(
                        reconstructed_path,
                        o3d.geometry.PointCloud(
                            o3d.utility.Vector3dVector(
                                pred.detach().clone().cpu())), write_ascii=True)

                    mpeg_pc_error_dict = mpeg_pc_error(
                        os.path.abspath(file_path),
                        os.path.abspath(reconstructed_path),
                        resolution=resolution, normal=False,
                        command=self.cfg.mpeg_pcc_error_command,
                        threads=self.cfg.mpeg_pcc_error_threads)
                    assert mpeg_pc_error_dict != {}, \
                        f'Error when call mpeg pc error software with ' \
                        f'infile1={os.path.abspath(file_path)} ' \
                        f'infile2={os.path.abspath(reconstructed_path)}'

                    for key, value in mpeg_pc_error_dict.items():
                        self.total_metric_values[key] += value
                        fileinfo += f'{key}: {value} \n'

                    with open(fileinfo_path, 'w') as f:
                        f.write(fileinfo)

            self.samples_num += len(targets)

            return True

        elif mode == 'show':
            metric_dict = {'samples_num': self.samples_num,
                           'mean_bpp': (self.total_bpp / self.samples_num),
                           'mean_bpp_pred': (self.total_bpp_pred / self.samples_num)}

            for key, value in self.total_metric_values.items():
                metric_dict[key] = value / self.samples_num

            if self.cfg.chamfer_dist_test_phase > 0:
                metric_dict['mean_reconstruct_loss'] = self.total_reconstruct_loss / self.samples_num

            return metric_dict

        else:
            raise NotImplementedError

    def forward(self, pc_data: PCData):
        ME.clear_global_coordinate_manager()

        if self.cfg.input_feature_type == 'Occupation':
            xyz = ME.SparseTensor(
                features=torch.ones(pc_data.xyz.shape[0], 1,
                                    dtype=torch.float,
                                    device=pc_data.xyz.device),
                coordinates=pc_data.xyz)

        elif self.cfg.input_feature_type == 'Coordinates':
            input_coords_scaler = torch.tensor(
                pc_data.resolution,
                device=pc_data.xyz.device)[pc_data.xyz[:, 0].type(torch.long), None]
            xyz = ME.SparseTensor(
                features=pc_data.xyz[:, 1:].type(torch.float) / input_coords_scaler,
                coordinates=pc_data.xyz)

        else: raise NotImplementedError

        fea, points_num_list = self.encoder(xyz)  # TODO: rename
        if not self.cfg.adaptive_pruning: points_num_list = None

        # bottleneck, decoder and loss in training
        if self.training:
            fea_tilde, loss_dict = self.entropy_bottleneck(fea)

            message: GenerativeUpsampleMessage = self.decoder(
                GenerativeUpsampleMessage(
                    fea=fea_tilde,
                    target_key=xyz.coordinate_map_key,
                    points_num_list=points_num_list))

            loss_dict['reconstruct_loss'] = self.get_reconstruct_loss(
                message.cached_pred,
                message.cached_target)

            # cross entropy loss
            loss_dict['bits_loss'] = loss_dict['bits_loss'] * \
                (self.cfg.bpp_loss_factor / xyz.shape[0])

            loss_dict['loss'] = sum(loss_dict.values())
            for key in loss_dict:
                if key != 'loss':
                    loss_dict[key] = loss_dict[key].detach().cpu().item()
            return loss_dict

        # bottleneck, decoder and metric in testing
        elif not self.training:
            fea, loss_dict, compressed_strings = self.entropy_bottleneck(fea)

            cached_pred = self.decoder(
                GenerativeUpsampleMessage(
                    fea=fea,
                    points_num_list=points_num_list)).cached_pred

            decoder_output = cached_pred[-1].decomposed_coordinates
            items_dict = self.log_pred_res(
                'log', preds=decoder_output,
                targets=xyz.decomposed_coordinates,
                compressed_strings=compressed_strings,
                fea_points_num=fea.shape[0],
                pc_data=pc_data,
                bits_preds=[loss_dict['bits_loss'].cpu().item()]
            )

            return items_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        # update_registered_buffers(
        #     self.entropy_bottleneck,
        #     "entropy_bottleneck",
        #     ["_quantized_cdf", "_offset", "_cdf_length"],
        #     state_dict
        # )
        # TODO
        return super().load_state_dict(state_dict, strict=strict)

    def get_reconstruct_loss(self, cached_pred, cached_target):
        if self.cfg.reconstruct_loss_type == 'BCE':
            reconstruct_loss = sum(
                [F.binary_cross_entropy_with_logits(pred.F.squeeze(),
                                                    target.type(pred.F.dtype))
                 for pred, target in zip(cached_pred, cached_target)])
            reconstruct_loss /= len(cached_pred)

        elif self.cfg.reconstruct_loss_type == 'Dist':
            reconstruct_loss = sum(
                [F.smooth_l1_loss(pred.F.squeeze(),
                                  target.type(pred.F.dtype))
                 for pred, target in zip(cached_pred, cached_target)])
            reconstruct_loss /= len(cached_pred)

        else:
            raise NotImplementedError

        if self.cfg.reconstruct_loss_factor != 1:
            reconstruct_loss *= self.cfg.reconstruct_loss_factor

        return reconstruct_loss

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (ME.MinkowskiConvolution,
                              ME.MinkowskiGenerativeConvolutionTranspose)):
                torch.nn.init.uniform_(m.kernel, -0.05, 0.05)
                torch.nn.init.zeros_(m.bias)


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
    # cfg.bottleneck = 'BinaryClassification'
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
