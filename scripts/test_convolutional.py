import shutil
import os.path as osp
from typing import Tuple, Dict, Callable, Union
from glob import glob
import subprocess
import json

from scripts.log_extract_utils import concat_values_for_dict
from scripts.shared_config import conda_prefix, metric_dict_filename


conda_env_name = 'py37torch110'
glob_weights_paths = [
    'weights/lossl_based/*.pt',
    'weights/hyperprior_factorized/*.pt',
    'weights/hyperprior_scale_normal/*.pt',
    'weights/baseline/*.pt',
    'weights/baseline_4x/*.pt'
]
config_paths = [
    'configs/train/convolutional/lossl_based',
    'configs/train/convolutional/hyperprior_factorized',
    'configs/train/convolutional/hyperprior_scale_normal',
    'configs/train/convolutional/baseline',
    'configs/train/convolutional/baseline_4x'
]


def rename_dict_key(d: Dict, mappings: Dict[str, Tuple[str, Union[None, Callable]]]):
    for key, (new_key, map_fn) in mappings.items():
        if key in d:
            assert new_key not in d
            if map_fn is not None:
                d[key] = map_fn(d[key])
            d[new_key] = d[key]
            del d[key]
    return d


key_mappings = {
    "encoder_elapsed_time": ('encode time', None),
    "encoder_max_cuda_memory_allocated": ('encode memory', lambda v: v / 1024),  # B -> KB
    "decoder_elapsed_time": ('decode time', None),
    "decoder_max_cuda_memory_allocated": ('decode memory', lambda v: v / 1024)
}


def test():
    if conda_prefix and conda_env_name:
        python_pre_command = f'. {osp.join(conda_prefix, "bin", "activate")} {conda_env_name};'
    else:
        python_pre_command = ';'

    for run_dirname, par_num in (
        ('convolutional_all_no_par', 0),
        ('convolutional_all_par6e5', 600000),
        ('convolutional_all_par15e4', 150000),
    ):
        run_dir = osp.join('runs', 'tests', run_dirname)
        for config_path, glob_weights_path in zip(config_paths, glob_weights_paths):
            all_file_metric_dict = {}
            config_name = osp.split(config_path)[1]
            sub_run_dir = osp.join(run_dir, config_name)
            for weight_path in glob(glob_weights_path):
                print(f'\nTest config: "{config_path}", par num {par_num}, weight "{weight_path}"\n')
                sub_sub_run_dir = osp.join(
                    sub_run_dir,
                    osp.splitext(osp.split(weight_path)[1])[0]
                )
                if osp.exists(sub_sub_run_dir):
                    shutil.rmtree(sub_sub_run_dir)
                subprocess.run(
                    f'{python_pre_command}'
                    f'python test.py {config_path}'
                    f' test.weights_from_ckpt={weight_path}'
                    f' test.rundir_name={sub_sub_run_dir.replace("runs/", "", 1)}'
                    f' test.dataset.kd_tree_partition_max_points_num={par_num}'
                    f' test.device=2',
                    shell=True, check=True, executable=shutil.which('bash')
                )
                sub_metric_dict_path = osp.join(sub_sub_run_dir, 'results', 'metric.txt')
                with open(sub_metric_dict_path, 'rb') as f:
                    sub_metric_dict = json.load(f)
                for key in sub_metric_dict:
                    all_file_metric_dict[key] = concat_values_for_dict(
                        all_file_metric_dict[key] if key in all_file_metric_dict else {},
                        rename_dict_key(sub_metric_dict[key], key_mappings)
                    )

            print(f'config "{config_path}", par num {par_num} Done')
            with open(osp.join(sub_run_dir, metric_dict_filename), 'w') as f:
                f.write(json.dumps(all_file_metric_dict, indent=2, sort_keys=False))
    print('All Done')


if __name__ == '__main__':
    test()
