import os
from functools import reduce
from collections import defaultdict
from typing import Tuple, List, Optional, Union

import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import compressai
from compressai.models.utils import update_registered_buffers

from lib.metric import mpeg_pc_error
from lib.loss_function import chamfer_loss
from lib.points_layers import PointLayerMessage, TransitionDown, RandLANeighborFea, \
    LocalFeatureAggregation as LFA
from lib.torch_utils import MLPBlock
from models.exp4.model_config import ModelConfig


class GenerativeTransitionUp(nn.Module):
    def __init__(self, lfa: LFA, upsample_rate: int = 2,):
        super(GenerativeTransitionUp, self).__init__()
        self.lfa = lfa
        self.mlp_pred = MLPBlock(lfa.out_channels // upsample_rate, 3, activation=None, batchnorm='nn.bn1d')
        self.upsample_rate = upsample_rate

    def forward(self, msg: PointLayerMessage):
        msg = self.lfa(msg)  # type: PointLayerMessage
        batch_size, points_num, channels = msg.feature.shape

        msg.raw_neighbors_feature = msg.neighbors_idx = None

        msg.feature = msg.feature.view(batch_size, points_num, self.upsample_rate, channels // self.upsample_rate)
        pred_offset = self.mlp_pred(msg.feature)

        pred_coord = msg.xyz.unsqueeze(2) + pred_offset
        pred_coord = pred_coord.reshape(batch_size, points_num * self.upsample_rate, 3)
        msg.cached_feature.append(pred_coord)

        msg.xyz = pred_coord.detach()
        msg.feature = msg.feature.reshape(batch_size, points_num * self.upsample_rate, channels // self.upsample_rate)

        return msg


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        neighbor_fea_generator = RandLANeighborFea(cfg.neighbor_num)

        self.encoder = [LFA(3, neighbor_fea_generator, 8, 16),
                        LFA(16, neighbor_fea_generator, 8, 16),

                        TransitionDown(cfg.sample_method, 0.25),
                        LFA(16, neighbor_fea_generator, 16, 32),
                        LFA(32, neighbor_fea_generator, 16, 32),

                        TransitionDown(cfg.sample_method, 0.25),
                        LFA(32, neighbor_fea_generator, 32, 64),
                        LFA(64, neighbor_fea_generator, 32, 64),

                        TransitionDown(cfg.sample_method, 0.25),
                        LFA(64, neighbor_fea_generator, 32, 64),
                        LFA(64, neighbor_fea_generator, 32, 64)]

        self.encoder = nn.Sequential(*self.encoder)
        self.mlp_enc_out = nn.Sequential(MLPBlock(64, 32, activation=None, batchnorm='nn.bn1d'))

        self.entropy_bottleneck = compressai.entropy_models.EntropyBottleneck(self.mlp_enc_out[-1].out_channels)

        self.decoder = nn.Sequential(
            GenerativeTransitionUp(LFA(32, neighbor_fea_generator, 32, 64),
                                   upsample_rate=4),
            GenerativeTransitionUp(LFA(16, neighbor_fea_generator, 16, 32),
                                   upsample_rate=4),
            GenerativeTransitionUp(LFA(8, neighbor_fea_generator, 8, 16),
                                   upsample_rate=4)
        )

    def forward(self, x):
        if self.training:
            raw_xyz, file_path_list, resolutions = x
            results_dir = None
        else:
            raw_xyz, file_path_list, resolutions, results_dir = x
        raw_fea = raw_xyz

        # encode
        msg = self.encoder(PointLayerMessage(xyz=raw_xyz, feature=raw_fea))  # type: PointLayerMessage
        fea = self.mlp_enc_out(msg.feature)
        fea = fea * self.cfg.bottleneck_scaler

        if self.training:
            fea, likelihoods = self.entropy_bottleneck(fea.permute(0, 2, 1).unsqueeze(3).contiguous())
            fea = fea / self.cfg.bottleneck_scaler
            fea = fea.squeeze(3).permute(0, 2, 1).contiguous()
            likelihoods = likelihoods.squeeze(3).permute(0, 2, 1).contiguous()
            msg = self.decoder(PointLayerMessage(xyz=msg.xyz, feature=fea))

            bpp_loss = torch.log2(likelihoods).sum() * (-self.cfg.bpp_loss_factor / (fea.shape[0] * fea.shape[1]))
            reconstruct_loss = sum([chamfer_loss(p, raw_xyz) for p in msg.cached_feature]) \
                * self.cfg.reconstruct_loss_factor
            aux_loss = self.entropy_bottleneck.loss()
            loss = reconstruct_loss + bpp_loss + aux_loss

            return {'aux_loss': aux_loss.detach().cpu().item(),
                    'bpp_loss': bpp_loss.detach().cpu().item(),
                    'reconstruct_loss': reconstruct_loss.detach().cpu().item(),
                    'loss': loss}
        else:
            compressed_strings = self.entropy_bottleneck_compress(fea)
            decompressed_tensors = self.entropy_bottleneck_decompress(compressed_strings, fea.shape[1])
            fea = decompressed_tensors / self.cfg.bottleneck_scaler
            msg = self.decoder(PointLayerMessage(xyz=msg.xyz, feature=fea))  # type: PointLayerMessage
            decoder_output = msg.cached_feature[-1]

            self.log_pred_res('log', decoder_output, raw_xyz,
                              file_path_list, compressed_strings,
                              resolutions, results_dir)

            return True

    def log_pred_res(self, mode, preds=None, targets=None,
                     file_path_list: str = None, compressed_strings: List[bytes] = None,
                     resolutions: Union[int, torch.Tensor] = None, results_dir: str = None):
        if mode == 'init' or mode == 'reset':
            self.total_reconstruct_loss = 0.0
            self.total_bpp = 0.0
            self.samples_num = 0
            self.total_metric_values = defaultdict(float)

        elif mode == 'log':
            assert not self.training

            resolutions = resolutions if isinstance(resolutions, torch.Tensor) \
                else torch.tensor([resolutions], dtype=torch.int32).expand(len(preds))

            for pred, target, file_path, com_string, resolution \
                    in zip(preds, targets, file_path_list, compressed_strings, resolutions):
                resolution = resolution.item()

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
                        target = torch.round(target * resolution).type(torch.int32)
                        file_path = out_file_path + '.ply'
                        o3d.io.write_point_cloud(file_path,
                                                 o3d.geometry.PointCloud(
                                                     o3d.utility.Vector3dVector(
                                                         target.cpu())), write_ascii=True)

                    with open(compressed_path, 'wb') as f:
                        f.write(com_string)

                    fileinfo = f'scaler: {self.cfg.bottleneck_scaler}\n' \
                               f'\n' \
                               f'input_points_num: {target.shape[0]}\n' \
                               f'output_points_num: {pred.shape[0]}\n' \
                               f'compressed_bytes: {len(com_string)}\n' \
                               f'bpp: {bpp}\n' \
                               f'\n'

                    pred = torch.round(pred * resolution).type(torch.int32)
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
                        self.total_metric_values[key] += value
                        fileinfo += f'{key}: {value} \n'

                    with open(fileinfo_path, 'w') as f:
                        f.write(fileinfo)

            self.samples_num += len(targets)

            return True

        elif mode == 'show':
            metric_dict = {'samples_num': self.samples_num,
                           'mean_bpp': (self.total_bpp / self.samples_num)}

            for key, value in self.total_metric_values.items():
                metric_dict[key] = value / self.samples_num

            if self.cfg.chamfer_dist_test_phase > 0:
                metric_dict['mean_reconstruct_loss'] = self.total_reconstruct_loss / self.samples_num

            return metric_dict

        else:
            raise NotImplementedError

    def load_state_dict(self, state_dict, strict: bool = True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        return super().load_state_dict(state_dict, strict=strict)

    def entropy_bottleneck_compress(self, encoder_output):
        assert not self.training
        encoder_output = encoder_output.permute(0, 2, 1).unsqueeze(3).contiguous()
        return self.entropy_bottleneck.compress(encoder_output)

    def entropy_bottleneck_decompress(self, compressed_strings, fea_points_num):
        assert not self.training
        decompressed_tensors = self.entropy_bottleneck.decompress(compressed_strings, size=(fea_points_num, 1))
        decompressed_tensors = decompressed_tensors.squeeze(3).permute(0, 2, 1)
        return decompressed_tensors


def main_t():
    from thop import profile
    from thop import clever_format

    cfg = ModelConfig()
    torch.cuda.set_device('cuda:3')
    model = PointCompressor(cfg).cuda()
    model.train()
    batch_points = torch.rand(2, 1024, 3).cuda()
    out = model((batch_points, [''], 0))
    model.entropy_bottleneck.update()
    model.eval()
    test_out = model((batch_points, [''], 0, None))

    macs, params = profile(model, inputs=(batch_points,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f'macs: {macs}, params: {params}')  # macs: 10.924G, params: 67.639M

    print('Done')


if __name__ == '__main__':
    main_t()
