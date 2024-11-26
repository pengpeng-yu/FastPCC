import json
from collections import defaultdict
from typing import Union, Dict
import os
import os.path as osp
import multiprocessing as mp

import torch

from lib.data_utils import write_ply_file
from lib.metrics.pc_error_wrapper import mpeg_pc_error


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
                 mpeg_pc_error_processes: int = 8):
        super(PCCEvaluator, self).__init__()
        self.cal_mpeg_pc_error = cal_mpeg_pc_error
        self.mpeg_pc_error_processes = mpeg_pc_error_processes
        self.working = False
        self.mp_ctx = mp.get_context('forkserver')

    def reset(self):
        self.file_path_to_info: Dict[str, Dict[str, Union[int, float]]] = {}
        self.file_path_to_info_run_res: Dict[str, mp.pool.AsyncResult] = {}

    @torch.no_grad()
    def log(self,
            pred: torch.Tensor,
            target: torch.Tensor,
            compressed_bytes: bytes,
            file_path: str,
            resolution: float,
            results_dir: str = None,
            pred_color: torch.Tensor = None,
            target_color: torch.Tensor = None,
            extra_info_dict: Dict[str, Union[str, int, float]] = None):
        """
        "pred" and "target" are coordinates with a specified resolution.
        "pred_color" and "target_color" are RGB colors. (0 ~ 255).
        """
        if not self.working:
            self.reset()
            self.mpeg_pc_error_pool = self.mp_ctx.Pool(self.mpeg_pc_error_processes)
            self.working = True

        have_color = pred_color is not None and target_color is not None
        assert pred.ndim == target.ndim == 2
        assert pred.shape[1] == target.shape[1] == 3

        file_info_dict = {
            'input_points_num': target.shape[0],
            'output_points_num': pred.shape[0],
            'compressed_bytes': len(compressed_bytes),
            'bpp': len(compressed_bytes) * 8 / target.shape[0]
        }
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
                assert file_path.endswith('.ply'), file_path
                self.file_path_to_info_run_res[file_path] = self.mpeg_pc_error_pool.apply_async(
                    mpeg_pc_error,
                    (osp.abspath(file_path), osp.abspath(reconstructed_path), resolution, '', False, have_color)
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

        if results_dir is not None:
            with open(osp.join(results_dir, '..', 'metric_dict.json'), 'w') as f:
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
            with open(osp.join(results_dir, '..', 'mean_metric.json'), 'w') as f:
                f.write(json.dumps(mean_dict, indent=2, sort_keys=False))

        self.reset()
        self.mpeg_pc_error_pool.close()
        self.mpeg_pc_error_pool.join()
        self.working = False
        return mean_dict
