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

from lib.data_utils import PCData, write_ply_file, if_ply_has_vertex_normal
from lib.metrics.misc import batch_image_psnr
from lib.metrics.pc_error_wapper import mpeg_pc_error


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
    def log_batch(self,
                  preds: Union[List[torch.Tensor], torch.Tensor],
                  targets: Union[List[torch.Tensor], torch.Tensor],
                  compressed_bytes_list: List[bytes],
                  pc_data: PCData,
                  preds_color: Union[List[torch.Tensor], torch.Tensor] = None,
                  targets_color: Union[List[torch.Tensor], torch.Tensor] = None,
                  extra_info_dicts: List[Dict[str, Union[str, int, float]]] = None):
        """
        "preds" and "targets" are supposed to be lists contain unnormalized
        coordinates with resolution specified in pc_data.resolution (GPU or CPU torch.int32).
        "color" are supposed to be unnormalized RGBs (GPU or CPU torch.float32).
        """
        batch_size = len(preds)
        assert batch_size == len(targets) == len(compressed_bytes_list)
        if pc_data.results_dir is not None:
            assert batch_size == len(pc_data.resolution) == len(pc_data.file_path)
        if preds_color is not None or targets_color is not None:
            have_color = True
            assert batch_size == len(preds_color) == len(targets_color)
        else:
            have_color = False

        for idx in range(batch_size):
            file_info_dict = {}

            file_path = pc_data.file_path[idx]
            pred = preds[idx]
            target = targets[idx]
            assert pred.ndim == target.ndim == 2
            assert pred.shape[1] == target.shape[1] == 3
            compressed_bytes = compressed_bytes_list[idx]
            resolution = pc_data.resolution[idx]

            bpp = len(compressed_bytes) * 8 / target.shape[0]

            if pc_data.results_dir is not None:
                out_file_path = osp.join(
                    pc_data.results_dir, osp.splitext(file_path)[0]
                )
                os.makedirs(osp.dirname(out_file_path), exist_ok=True)
                compressed_path = out_file_path + '.bin'
                reconstructed_path = out_file_path + '_recon.ply'
                with open(compressed_path, 'wb') as f:
                    f.write(compressed_bytes)
                file_info_dict.update(
                    {
                        'input_points_num': target.shape[0],
                        'output_points_num': pred.shape[0],
                        'compressed_bytes': len(compressed_bytes),
                        'bpp': bpp
                    }
                )
                if extra_info_dicts is not None:
                    file_info_dict.update(extra_info_dicts[idx])
                write_ply_file(pred, reconstructed_path, rgb=preds_color[idx] if have_color else None)

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
                            target, file_path, rgb=targets_color[idx] if have_color else None,
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
        psnr = (np.log10(255 / np.linalg.norm(im.astype(np.double) - im_recon) * np.sqrt(im.size)) * 10).item()
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
            with open(osp.join(results_dir, 'test_metric.txt'), 'w') as f:
                f.write(json.dumps(self.file_path_to_info, indent=2, sort_keys=False))

        mean_dict = defaultdict(float)
        for _, info in self.file_path_to_info.items():
            for key, value in info.items():
                mean_dict[key + '(mean)'] += value
        samples_num = len(self.file_path_to_info)
        for key in mean_dict:
            mean_dict[key] /= samples_num
        if results_dir is not None:
            with open(osp.join(results_dir, 'mean.txt'), 'w') as f:
                f.write(json.dumps(mean_dict, indent=2, sort_keys=False))
        return mean_dict
