from collections import defaultdict
from typing import Tuple, List, Union
import os

import open3d as o3d
import torch

from lib.data_utils import PCData
from lib.loss_function import chamfer_loss
from lib.metric import mpeg_pc_error


class Evaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def log_batch(self, *args, **kwargs):
        raise NotImplementedError

    def show(self):
        raise NotImplementedError


class PCGCEvaluator(Evaluator):
    def __init__(self, mpeg_pcc_error_command: str, mpeg_pcc_error_threads: int):
        super(PCGCEvaluator, self).__init__()
        self.mpeg_pcc_error_command = mpeg_pcc_error_command
        self.mpeg_pcc_error_threads = mpeg_pcc_error_threads

    def reset(self):
        self.total_chamfer_loss = 0.0
        self.total_bpp = 0.0
        self.samples_num = 0
        self.total_metric_values = defaultdict(float)
        self.total_bpp_pred = 0.0

    def log_batch(self,
                  preds: Union[List[torch.Tensor], torch.Tensor],
                  targets: Union[List[torch.Tensor], torch.Tensor],
                  compressed_strings: List[bytes],
                  bits_preds: List[float],
                  feature_points_numbers: List[int],
                  pc_data: PCData,
                  compute_chamfer_loss: bool = False):
        """
        "preds" and "targets" are supposed to be list contains unnormalized discrete
        coordinates with resolution specified in pc_data.resolution.
        """
        batch_size = len(preds)

        assert batch_size == \
               len(targets) == \
               len(compressed_strings) == \
               len(bits_preds) == \
               len(feature_points_numbers)

        if compute_chamfer_loss:
            assert batch_size == len(pc_data.resolution)

        if hasattr(pc_data, 'results_dir'):
            assert batch_size == \
                   len(pc_data.resolution) == \
                   len(pc_data.file_path)

        for idx in range(batch_size):
            pred = preds[idx]
            target = targets[idx]
            assert pred.ndim == target.ndim == 2
            assert pred.shape[1] == target.shape[1] == 3

            compressed_string = compressed_strings[idx]
            resolution = pc_data.resolution[idx]

            if compute_chamfer_loss:
                self.total_chamfer_loss += chamfer_loss(
                    pred[None].to(torch.float) / resolution,
                    target[None].to(torch.float) / resolution
                ).item()

            bpp = len(compressed_string) * 8 / target.shape[0]
            self.total_bpp += bpp

            self.total_bpp_pred += bits_preds[idx] / target.shape[0]

            if hasattr(pc_data, 'results_dir'):
                file_path = pc_data.file_path[idx]

                out_file_path = os.path.join(
                    pc_data.results_dir, os.path.splitext(file_path)[0]
                )

                os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
                compressed_path = out_file_path + '.txt'
                fileinfo_path = out_file_path + '_info.txt'
                reconstructed_path = out_file_path + '_recon.ply'

                with open(compressed_path, 'wb') as f:
                    f.write(compressed_string)

                fileinfo = f'fea_points_num: {feature_points_numbers[idx]}\n' \
                           f'\n' \
                           f'input_points_num: {target.shape[0]}\n' \
                           f'output_points_num: {pred.shape[0]}\n' \
                           f'compressed_bytes: {len(compressed_string)}\n' \
                           f'bpp: {bpp}\n'

                o3d.io.write_point_cloud(
                    reconstructed_path,
                    o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(
                            pred.detach().clone().cpu()
                        )
                    ), write_ascii=True
                )

                if self.mpeg_pcc_error_command != '':
                    if not file_path.endswith('.ply') or \
                            pc_data.ori_resolution is None or \
                            pc_data.ori_resolution[idx] != resolution:
                        file_path = out_file_path + '.ply'
                        o3d.io.write_point_cloud(
                            file_path,
                            o3d.geometry.PointCloud(
                                o3d.utility.Vector3dVector(
                                    target.cpu()
                                )
                            ), write_ascii=True
                        )

                    mpeg_pc_error_dict = mpeg_pc_error(
                        os.path.abspath(file_path),
                        os.path.abspath(reconstructed_path),
                        resolution=resolution, normal=False,
                        command=self.mpeg_pcc_error_command,
                        threads=self.mpeg_pcc_error_threads)

                    assert mpeg_pc_error_dict != {}, \
                        f'Error when call mpeg pc error software with ' \
                        f'infile1={os.path.abspath(file_path)} ' \
                        f'infile2={os.path.abspath(reconstructed_path)}'

                    for key, value in mpeg_pc_error_dict.items():
                        self.total_metric_values[key] += value
                        fileinfo += f'{key}: {value} \n'

                with open(fileinfo_path, 'w') as f:
                    f.write(fileinfo)

        self.samples_num += batch_size

        return True

    def show(self):
        metric_dict = {'samples_num': self.samples_num,
                       'mean_bpp': (self.total_bpp / self.samples_num),
                       'mean_bpp_pred': (self.total_bpp_pred / self.samples_num)}

        for key, value in self.total_metric_values.items():
            metric_dict[key] = value / self.samples_num

        if self.total_chamfer_loss != 0:
            metric_dict['mean_reconstruct_loss'] = self.total_chamfer_loss / self.samples_num

        return metric_dict




