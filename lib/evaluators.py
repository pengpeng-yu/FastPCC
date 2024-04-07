import json
from collections import defaultdict
from typing import List, Union, Dict
import os
import os.path as osp
import multiprocessing as mp

try:
    import cv2
except ImportError: cv2 = None
import numpy as np
import torch

from lib.data_utils import write_ply_file, if_ply_has_vertex_normal
from lib.metrics.pc_error_wapper import mpeg_pc_error
from lib.loss_functions import chamfer_loss


class Evaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def log(self, *args, **kwargs):
        raise NotImplementedError

    def log_batch(self, *args, **kwargs):
        raise NotImplementedError

    def show(self, results_dir: str):
        raise NotImplementedError


class PCCEvaluator(Evaluator):
    def __init__(self,
                 cal_mpeg_pc_error: bool = True,
                 mpeg_pc_error_processes: int = 16):
        super(PCCEvaluator, self).__init__()
        self.cal_mpeg_pc_error = cal_mpeg_pc_error
        self.mpeg_pc_error_pool = mp.Pool(mpeg_pc_error_processes)

    def reset(self):
        self.file_path_to_info: Dict[str, Dict[str, Union[int, float]]] = {}
        self.file_path_to_info_run_res: Dict[str, mp.pool.AsyncResult] = {}

    @torch.no_grad()
    def log(self,
            pred: torch.Tensor,
            target: torch.Tensor,
            compressed_bytes: bytes,
            file_path: str,
            resolution: int,
            results_dir: str = None,
            pred_color: torch.Tensor = None,
            target_color: torch.Tensor = None,
            extra_info_dict: Dict[str, Union[str, int, float]] = None):
        """
        "pred" and "target" are coordinates with a specified resolution.
        "pred_color" and "target_color" are RGB colors. (0 ~ 255).
        """
        have_color = pred_color is not None and target_color is not None
        assert pred.ndim == target.ndim == 2
        assert pred.shape[1] == target.shape[1] == 3

        file_info_dict = {
            'input_points_num': target.shape[0],
            'output_points_num': pred.shape[0],
            'compressed_bytes': len(compressed_bytes),
            'bpp': len(compressed_bytes) * 8 / target.shape[0]
        }
        if target.dtype.is_floating_point:  # For LiDAR datasets.
            try:
                file_info_dict['chamfer'] = chamfer_loss(
                    pred[None].to(target.dtype),
                    target[None].to(pred.device)).item()
            except Exception as e:
                print(e)
        if extra_info_dict is not None:
            file_info_dict.update(extra_info_dict)

        if results_dir is not None:
            out_file_path = osp.join(
                results_dir, osp.splitext(file_path)[0]
            )
            os.makedirs(osp.dirname(out_file_path), exist_ok=True)
            compressed_path = out_file_path + '.bin'
            reconstructed_path = out_file_path + '_recon.ply'
            with open(compressed_path, 'wb') as f:
                f.write(compressed_bytes)
            write_ply_file(pred, reconstructed_path, rgb=pred_color if have_color else None)

            if self.cal_mpeg_pc_error:
                write_ply_for_orig_pc = False
                if file_path.endswith('.ply'):
                    if_target_has_normal = if_ply_has_vertex_normal(file_path)
                    if if_target_has_normal:
                        normal_file_path = file_path
                    else:
                        normal_file_path = osp.splitext(file_path)[0] + '_n.ply'
                        if not osp.isfile(normal_file_path):
                            normal_file_path = out_file_path + '.ply'
                            write_ply_for_orig_pc = True
                else:
                    normal_file_path = out_file_path + '.ply'
                    write_ply_for_orig_pc = True
                if write_ply_for_orig_pc:
                    file_path = out_file_path + '.ply'
                    write_ply_file(
                        target, file_path, rgb=target_color if have_color else None,
                        estimate_normals=True
                    )
                    print(f'Wrote Ply file to {file_path} with normals estimation')
                    normal_file_path = file_path
                self.file_path_to_info_run_res[file_path] = self.mpeg_pc_error_pool.apply_async(
                    mpeg_pc_error,
                    (osp.abspath(file_path),
                     osp.abspath(reconstructed_path),
                     resolution, normal_file_path, False, have_color)
                )

        if file_path in self.file_path_to_info:
            print(f'Warning: Duplicated test sample {file_path}')
        self.file_path_to_info[file_path] = file_info_dict

        return True

    def show(self, results_dir: str) -> Dict[str, Union[int, float]]:
        for file_path, run_res in self.file_path_to_info_run_res.items():
            mpeg_pc_error_dict = run_res.get()
            assert mpeg_pc_error_dict != {}, \
                f'Error when calling mpeg pc error software with ' \
                f'infile1={osp.abspath(file_path)} '
            self.file_path_to_info[file_path].update(mpeg_pc_error_dict)
            self.file_path_to_info[file_path]["mse1+mse2 (p2point)"] = \
                mpeg_pc_error_dict["mse1      (p2point)"] + \
                mpeg_pc_error_dict["mse2      (p2point)"]

        if results_dir is not None:
            with open(osp.join(results_dir, 'metric.txt'), 'w') as f:
                f.write(json.dumps(self.file_path_to_info, indent=2, sort_keys=False))

        mean_dict = defaultdict(float)
        count_dist = defaultdict(int)
        exclusion = ['fea_points_num',
                     'input_points_num',
                     'output_points_num']
        for _, info in self.file_path_to_info.items():
            for key, value in info.items():
                if key not in exclusion:
                    key = key + '(mean)'
                    count_dist[key] += 1
                    mean_dict[key] += value
        samples_num = len(self.file_path_to_info)
        for key in mean_dict:
            mean_dict[key] /= samples_num
        mean_dict = {k: v for k, v in mean_dict.items() if count_dist[k] == samples_num}
        mean_dict['samples_num'] = samples_num
        if results_dir is not None:
            with open(osp.join(results_dir, 'mean_metric.txt'), 'w') as f:
                f.write(json.dumps(mean_dict, indent=2, sort_keys=False))
        return mean_dict


class ImageCompressionEvaluator(Evaluator):
    def __init__(self):
        super(ImageCompressionEvaluator, self).__init__()

    def reset(self):
        self.file_path_to_info = {}

    def log(self, im_recon, im, compressed_bytes, file_path, results_dir):
        im = im.cpu().numpy()
        im_recon = im_recon.cpu().numpy()
        psnr = (np.log10(255 / np.linalg.norm(im.astype(np.double) - im_recon) * np.sqrt(im.size)) * 20).item()
        pixels_num = im_recon.shape[1] * im_recon.shape[2]

        if results_dir is not None:
            out_file_path = osp.join(results_dir, file_path)
            os.makedirs(osp.dirname(out_file_path), exist_ok=True)
            cv2.imwrite(out_file_path, im_recon.astype(np.uint8).transpose(1, 2, 0))
        self.file_path_to_info[file_path] = \
            {
                'psnr': psnr,
                'bpp': len(compressed_bytes) * 8 / pixels_num
            }
        return True

    def show(self, results_dir: str) -> Dict[str, Union[int, float]]:
        if results_dir is not None:
            with open(osp.join(results_dir, 'metric.txt'), 'w') as f:
                f.write(json.dumps(self.file_path_to_info, indent=2, sort_keys=False))

        mean_dict = defaultdict(float)
        for _, info in self.file_path_to_info.items():
            for key, value in info.items():
                mean_dict[key + '(mean)'] += value
        samples_num = len(self.file_path_to_info)
        for key in mean_dict:
            mean_dict[key] /= samples_num
        if results_dir is not None:
            with open(osp.join(results_dir, 'mean_metric.txt'), 'w') as f:
                f.write(json.dumps(mean_dict, indent=2, sort_keys=False))
        return mean_dict
