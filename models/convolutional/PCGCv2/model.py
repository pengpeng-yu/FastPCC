import os
from collections import defaultdict
from typing import Union, List
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from compressai.entropy_models import EntropyBottleneck
from compressai.models.utils import update_registered_buffers

from lib.metric import mpeg_pc_error
from lib.loss_function import chamfer_loss
from lib.sparse_conv_layers import GenerativeUpsampleMessage
from models.convolutional.PCGCv2.layers import Encoder, DecoderBlock
from models.convolutional.PCGCv2.model_config import ModelConfig


class PCC(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(PCC, self).__init__()
        ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
        self.encoder = Encoder(cfg.compressed_channels,
                               res_blocks_num=3,
                               res_block_type=cfg.res_block_type)
        self.decoder = self.layers = nn.Sequential(DecoderBlock(cfg.compressed_channels,
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
        self.entropy_bottleneck = EntropyBottleneck(cfg.compressed_channels)
        self.cfg = cfg
        self.log_pred_res('init')

    def log_pred_res(self, mode, preds=None, targets=None,
                     file_path_list: str = None, compressed_strings: List[bytes] = None,
                     fea_points_num: int = None, resolutions: Union[int, torch.Tensor] = None, results_dir: str = None):
        if mode == 'init' or mode == 'reset':
            self.total_reconstruct_loss = 0.0
            self.total_bpp = 0.0
            self.samples_num = 0
            self.totall_metric_values = defaultdict(float)

        elif mode == 'log':
            assert not self.training
            assert isinstance(preds, list) and isinstance(targets, list)

            if preds.shape[0] != 1: raise NotImplementedError

            resolutions = resolutions if isinstance(resolutions, torch.Tensor) \
                else torch.tensor([resolutions], dtype=torch.int32).expand(len(preds))

            for pred, target, file_path, resolution \
                    in zip(preds, targets, file_path_list, resolutions):
                resolution = resolution.item()
                com_string = compressed_strings[0]

                if self.cfg.chamfer_dist_test_phase is True:
                    self.total_reconstruct_loss += chamfer_loss(
                        pred.unsqueeze(0).type(torch.float) / resolution,
                        target.unsqueeze(0).type(torch.float) / resolution).item()

                bpp = len(com_string) * 8 / target.shape[0]
                self.total_bpp += bpp

                if results_dir is not None:
                    out_file_path = os.path.join(results_dir, os.path.splitext(file_path)[0])
                    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
                    compressed_path = out_file_path + '.txt'
                    fileinfo_path = out_file_path + '_info.txt'
                    reconstructed_path = out_file_path + '_recon.ply'

                    if not file_path.endswith('.ply'):
                        file_path = out_file_path + '.ply'
                        o3d.io.write_point_cloud(file_path,
                                                 o3d.geometry.PointCloud(
                                                     o3d.utility.Vector3dVector(
                                                         target.cpu())), write_ascii=True)

                    with open(compressed_path, 'wb') as f:
                        f.write(com_string)

                    fileinfo = f'fea_points_num: {fea_points_num}\n' \
                               f'scaler: {self.cfg.bottleneck_scaler}\n' \
                               f'\n' \
                               f'input_points_num: {target.shape[0]}\n' \
                               f'output_points_num: {pred.shape[0]}\n' \
                               f'compressed_bytes: {len(com_string)}\n' \
                               f'bpp: {bpp}\n' \
                               f'\n'

                    o3d.io.write_point_cloud(reconstructed_path,
                                             o3d.geometry.PointCloud(
                                                 o3d.utility.Vector3dVector(
                                                     pred.detach().clone().cpu())), write_ascii=True)

                    mpeg_pc_error_dict = mpeg_pc_error(os.path.abspath(file_path), os.path.abspath(reconstructed_path),
                                                       resolution=resolution, normal=False,
                                                       command=self.cfg.mpeg_pcc_error_command,
                                                       threads=self.cfg.mpeg_pcc_error_threads)
                    assert mpeg_pc_error_dict != {}, f'Error when call mpeg pc error software with ' \
                                                     f'infile1={os.path.abspath(file_path)} ' \
                                                     f'infile2={os.path.abspath(reconstructed_path)}'

                    for key, value in mpeg_pc_error_dict.items():
                        self.totall_metric_values[key] += value
                        fileinfo += f'{key}: {value} \n'

                    with open(fileinfo_path, 'w') as f:
                        f.write(fileinfo)

            self.samples_num += len(targets)

            return True

        elif mode == 'show':
            metric_dict = {'samples_num': self.samples_num,
                           'mean_bpp': (self.total_bpp / self.samples_num)}

            for key, value in self.totall_metric_values.items():
                metric_dict[key] = value / self.samples_num

            if self.cfg.chamfer_dist_test_phase > 0:
                metric_dict['mean_recontruct_loss'] = self.total_reconstruct_loss / self.samples_num

            return metric_dict

        else:
            raise NotImplementedError

    def forward(self, x):
        ME.clear_global_coordinate_manager()
        if self.training:
            xyz, file_path_list, resolutions = x
            results_dir = None
        else:
            xyz, file_path_list, resolutions, results_dir = x
        xyz = ME.SparseTensor(torch.ones(xyz.shape[0], 1, dtype=torch.float, device=xyz.device),
                              xyz)
        fea, points_num_list = self.encoder(xyz)
        if not self.cfg.adaptive_pruning: points_num_list = None

        if self.training:
            # TODO: scaler?
            fea_tilde, likelihood = self.entropy_bottleneck(fea.F.T.unsqueeze(0) * self.cfg.bottleneck_scaler)
            fea_tilde = fea_tilde / self.cfg.bottleneck_scaler
            fea_tilde = ME.SparseTensor(fea_tilde.squeeze(0).T,
                                        coordinate_map_key=fea.coordinate_map_key,
                                        coordinate_manager=ME.global_coordinate_manager())

            message: GenerativeUpsampleMessage = \
                self.decoder(GenerativeUpsampleMessage(fea=fea_tilde,
                                                       target_key=xyz.coordinate_map_key,
                                                       points_num_list=points_num_list))
            cached_pred = message.cached_pred
            cached_target = message.cached_target

            bpp_loss = torch.log2(likelihood).sum() * (
                    -self.cfg.bpp_loss_factor / xyz.shape[0])

            if self.cfg.reconstruct_loss_type == 'BCE':
                reconstruct_loss = sum([nn.functional.binary_cross_entropy_with_logits(pred.F.squeeze(),
                                                                                       target.type(pred.F.dtype))
                                        for pred, target in zip(cached_pred, cached_target)])
                reconstruct_loss /= len(cached_pred)
            elif self.cfg.reconstruct_loss_type == 'Dist':
                reconstruct_loss = sum([nn.functional.smooth_l1_loss(pred.F.squeeze(),
                                                                     target.type(pred.F.dtype))
                                        for pred, target in zip(cached_pred, cached_target)])
                reconstruct_loss /= len(cached_pred)
            else: raise NotImplementedError
            if self.cfg.reconstruct_loss_factor != 1:
                reconstruct_loss *= self.cfg.reconstruct_loss_factor

            aux_loss = self.entropy_bottleneck.loss() * self.cfg.aux_loss_factor
            loss = bpp_loss + reconstruct_loss + aux_loss

            return {'loss': loss,
                    'bpp_loss': bpp_loss.detach().cpu().item(),
                    'reconstruct_loss': reconstruct_loss.detach().cpu().item(),
                    'aux_loss': aux_loss.detach().cpu().item()}

        else:
            cached_map_key = fea.coordinate_map_key
            compressed_strings = self.entropy_bottleneck_compress(fea.F * self.cfg.bottleneck_scaler)
            fea = self.entropy_bottleneck_decompress(compressed_strings, fea.shape[0])
            fea = ME.SparseTensor(fea / self.cfg.bottleneck_scaler,
                                  coordinate_map_key=cached_map_key,
                                  coordinate_manager=ME.global_coordinate_manager())
            cached_pred = \
                self.decoder(GenerativeUpsampleMessage(fea=fea,
                                                       points_num_list=points_num_list)).cached_pred

            decoder_output = cached_pred[-1].decomposed_coordinates
            items_to_save = self.log_pred_res('log', decoder_output, xyz.decomposed_coordinates,
                                              file_path_list, compressed_strings, fea.shape[0],
                                              resolutions, results_dir)

            return items_to_save

    def load_state_dict(self, state_dict, strict: bool = True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    def entropy_bottleneck_compress(self, encoder_output):
        assert not self.training
        encoder_output = encoder_output.T[None, :, :, None].contiguous()
        return self.entropy_bottleneck.compress(encoder_output)

    def entropy_bottleneck_decompress(self, compressed_strings, points_num):
        assert not self.training
        decompressed_tensors = self.entropy_bottleneck.decompress(compressed_strings, size=(points_num, 1))
        decompressed_tensors = decompressed_tensors.squeeze().T
        return decompressed_tensors


def main_debug():
    ME.clear_global_coordinate_manager()
    ME.set_sparse_tensor_operation_mode(
        ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

    coords, feats = ME.utils.sparse_collate(coords=[torch.tensor([[3, 5, 7], [8, 9, 1], [-2, 3, 3], [-2, 3, 4]],
                                                                 dtype=torch.int32)],
                                            feats=[torch.tensor([[0.5, 0.4], [0.1, 0.6], [-0.9, 10], [-0.9, 8]])])
    batch_coords = torch.tensor([[0, 3, 5, 7], [0, 8, 9, 1], [0, -2, 3, 2], [0, -2, 3, 4]], dtype=torch.int32)

    pc = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1)
    cm = ME.global_coordinate_manager()
    coord_map_key = cm.insert_and_map(batch_coords, tensor_stride=1)[0]

    print(cm.kernel_map(pc.coordinate_map_key, coord_map_key, kernel_size=1))

    conv11 = ME.MinkowskiConvolution(2, 2, 1, 1, dimension=3)
    conv_trans22 = ME.MinkowskiConvolutionTranspose(2, 6, 2, 2, dimension=3)

    pc = conv11(pc)
    m_out_trans = conv_trans22(pc)
    pc = pc.dense(shape=torch.Size([1, 3, 10, 10, 10]), min_coordinate=torch.IntTensor([0, 0, 0]))

    print('Done')


def model_debug():
    cfg = ModelConfig()
    cfg.resolution = 128
    model = PCC(cfg)
    xyz_c = [ME.utils.sparse_quantize(torch.randint(-128, 128, (100, 3))) for _ in range(16)]
    xyz_f = [torch.ones((_.shape[0], 1), dtype=torch.float32) for _ in xyz_c]
    xyz = ME.utils.sparse_collate(coords=xyz_c, feats=xyz_f)
    out = model((xyz[0], [''], 0))
    out['loss'].backward()
    model.eval()
    model.entropy_bottleneck.update()
    test_out = model((xyz[0], [''], 0, None))
    print('Done')


if __name__ == '__main__':
    # main_debug()
    model_debug()

