import json
from collections import defaultdict
from typing import Tuple, List, Union, Dict
import os

import cv2
import numpy as np
import torch

from lib.data_utils import PCData, write_xyz_to_ply_file
try:
    from lib.loss_functions import chamfer_loss
except ImportError:
    chamfer_loss = None
from lib.metrics.misc import batch_image_psnr
from lib.metrics.pc_error_wapper import mpeg_pc_error


class Evaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def log_batch(self, *args, **kwargs):
        raise NotImplementedError

    def show(self, results_dir: str):
        raise NotImplementedError


class PCGCEvaluator(Evaluator):
    def __init__(self,
                 mpeg_pcc_error_command: str,
                 mpeg_pcc_error_threads: int,
                 compute_chamfer_loss: bool = False):
        super(PCGCEvaluator, self).__init__()
        self.mpeg_pcc_error_command = mpeg_pcc_error_command
        self.mpeg_pcc_error_threads = mpeg_pcc_error_threads
        self.compute_chamfer_loss = compute_chamfer_loss

    def reset(self):
        self.file_path_to_info: Dict[str, Dict[str, Union[int, float]]] = {}

    @torch.no_grad()
    def log_batch(self,
                  preds: Union[List[torch.Tensor], torch.Tensor],
                  targets: Union[List[torch.Tensor], torch.Tensor],
                  compressed_strings: List[bytes],
                  pc_data: PCData,
                  extra_info_dicts: List[Dict[str, Union[str, int, float]]] = None):
        """
        "preds" and "targets" are supposed to be list contains unnormalized discrete
        coordinates with resolution specified in pc_data.resolution.
        """
        batch_size = len(preds)
        assert batch_size == \
               len(targets) == \
               len(compressed_strings)
        if self.compute_chamfer_loss:
            assert batch_size == len(pc_data.resolution)
        if pc_data.results_dir is not None:
            assert batch_size == \
                   len(pc_data.resolution) == \
                   len(pc_data.file_path)

        for idx in range(batch_size):
            file_info_dict = {}

            file_path = pc_data.file_path[idx]
            pred = preds[idx]
            target = targets[idx]
            assert pred.ndim == target.ndim == 2
            assert pred.shape[1] == target.shape[1] == 3
            compressed_string = compressed_strings[idx]
            resolution = pc_data.resolution[idx]

            if self.compute_chamfer_loss:
                file_info_dict['chamfer_loss'] = chamfer_loss(
                    pred[None].to(torch.float) / resolution,
                    target[None].to(torch.float) / resolution
                ).item()

            bpp = len(compressed_string) * 8 / target.shape[0]

            if pc_data.results_dir is not None:
                out_file_path = os.path.join(
                    pc_data.results_dir, os.path.splitext(file_path)[0]
                )
                os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
                compressed_path = out_file_path + '.bin'
                reconstructed_path = out_file_path + '_recon.ply'
                with open(compressed_path, 'wb') as f:
                    f.write(compressed_string)
                file_info_dict.update(
                    {
                        'input_points_num': target.shape[0],
                        'output_points_num': pred.shape[0],
                        'compressed_bytes': len(compressed_string),
                        'bpp': bpp
                    }
                )
                if extra_info_dicts is not None:
                    file_info_dict.update(extra_info_dicts[idx])
                write_xyz_to_ply_file(pred, reconstructed_path)

                if self.mpeg_pcc_error_command != '':
                    if not file_path.endswith('.ply') or \
                            pc_data.ori_resolution is None or \
                            pc_data.ori_resolution[idx] != resolution:
                        file_path = out_file_path + '.ply'
                        write_xyz_to_ply_file(target, file_path)
                    mpeg_pc_error_dict = mpeg_pc_error(
                        os.path.abspath(file_path),
                        os.path.abspath(reconstructed_path),
                        resolution=resolution, normal=False,
                        command=self.mpeg_pcc_error_command,
                        threads=self.mpeg_pcc_error_threads)
                    assert mpeg_pc_error_dict != {}, \
                        f'Error when calling mpeg pc error software with ' \
                        f'infile1={os.path.abspath(file_path)} ' \
                        f'infile2={os.path.abspath(reconstructed_path)}'
                    file_info_dict.update(mpeg_pc_error_dict)
            assert file_path not in self.file_path_to_info
            self.file_path_to_info[file_path] = file_info_dict

        return True

    def show(self, results_dir: str) -> Dict[str, Union[int, float]]:
        if results_dir is not None:
            with open(os.path.join(results_dir, 'metric.txt'), 'w') as f:
                f.write(json.dumps(self.file_path_to_info, indent=2, sort_keys=False))

        mean_dict = defaultdict(float)
        exclusion = ['fea_points_num',
                     'input_points_num',
                     'output_points_num']
        if not self.compute_chamfer_loss:
            exclusion.append('chamfer_loss')
        for _, info in self.file_path_to_info.items():
            for key, value in info.items():
                if key not in exclusion:
                    mean_dict[key + '(mean)'] += value
        samples_num = len(self.file_path_to_info)
        for key in mean_dict:
            mean_dict[key] /= samples_num
        mean_dict['samples_num'] = samples_num
        if results_dir is not None:
            with open(os.path.join(results_dir, 'mean_metric.txt'), 'w') as f:
                f.write(json.dumps(mean_dict, indent=2, sort_keys=False))
        return mean_dict


class ImageCompressionEvaluator(Evaluator):
    def __init__(self):
        super(ImageCompressionEvaluator, self).__init__()

    def reset(self):
        self.file_path_to_info = {}

    def log_batch(self,
                  batch_im_recon,
                  batch_im,
                  compressed_strings,
                  file_paths,
                  valid_ranges,
                  results_dir):
        if not batch_im_recon.shape[0] == 1:
            raise NotImplementedError
        assert len(batch_im_recon) == len(compressed_strings) == len(file_paths)

        valid_range = valid_ranges[0]
        batch_psnr = batch_image_psnr(
            batch_im_recon[:, :, valid_range[0][0]: valid_range[0][1],
                           valid_range[1][0]: valid_range[1][1]],
            batch_im[:, :, valid_range[0][0]: valid_range[0][1],
                     valid_range[1][0]: valid_range[1][1]],
            max_val=255
        )
        batch_im_recon = batch_im_recon.to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        pixels_num = batch_im_recon.shape[1] * batch_im_recon.shape[2]

        for idx in range(len(batch_psnr)):
            psnr = batch_psnr[idx].item()
            if results_dir is not None:
                out_file_path = os.path.join(results_dir, file_paths[idx])
                os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
                cv2.imwrite(out_file_path, batch_im_recon[idx])
            self.file_path_to_info[file_paths[idx]] = \
                {
                    'psnr': psnr,
                    'bpp': len(compressed_strings[idx]) * 8 / pixels_num
                }
        return True

    def show(self, results_dir: str) -> Dict[str, Union[int, float]]:
        if results_dir is not None:
            with open(os.path.join(results_dir, 'test_metric.txt'), 'w') as f:
                f.write(json.dumps(self.file_path_to_info, indent=2, sort_keys=False))

        mean_dict = defaultdict(float)
        for _, info in self.file_path_to_info.items():
            for key, value in info.items():
                mean_dict[key + '(mean)'] += value
        samples_num = len(self.file_path_to_info)
        for key in mean_dict:
            mean_dict[key] /= samples_num
        if results_dir is not None:
            with open(os.path.join(results_dir, 'mean.txt'), 'w') as f:
                f.write(json.dumps(mean_dict, indent=2, sort_keys=False))
        return mean_dict
