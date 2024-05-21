import shutil
import sys
import os.path as osp
from glob import glob
import subprocess
import json

sys.path.append(osp.dirname(osp.dirname(__file__)))
from scripts.log_extract_utils import concat_values_for_dict
from scripts.shared_config import metric_dict_filename, cuda_device, test_dir


weight_prefix = 'weights/convolutional'
config_prefix = 'config/convolutional'
output_prefix = f'{test_dir}/convolutional'
config_paths = [
    'lossy_coord_v2/baseline_r*.yaml',
    'lossy_coord_v2/baseline_part6e5_r*.yaml',
    'lossy_coord_v2/gpcc_based_r*.yaml',
    'lossy_coord_v2/wo_residual_r*.yaml',
    'lossy_coord/baseline.yaml',
    'lossy_coord_lossy_color/baseline_r*.yaml',
    'lossy_coord_v2/baseline_kitti_r*.yaml',
    'lossy_coord_v2/baseline_kitti_q1mm_r*.yaml',
]
sub_config_to_weight_path_maps = {
    'lossy_coord_v2/baseline_part6e5_r*.yaml': lambda _: _.replace('_part6e5', '', 1),
    'lossy_coord_v2/baseline_kitti_q1mm_r*.yaml': lambda _: _.replace('_q1mm', '', 1)
}


def test():
    for glob_config_path in config_paths:
        glob_config_path_org = glob_config_path
        glob_config_path = osp.join(config_prefix, glob_config_path)
        sub_config_paths = sorted(glob(glob_config_path))
        for config_path in sub_config_paths:
            config_name = config_path[len(config_prefix)+1: -5]
            sub_run_dir = osp.join(output_prefix, config_name)
            all_file_metric_dict = {}

            glob_weight_path = osp.join(
                weight_prefix,
                sub_config_to_weight_path_maps.get(glob_config_path_org, lambda _: _)(config_name) + '.pt'
            )
            weights_paths = sorted(glob(glob_weight_path))
            if len(weights_paths) == 0:
                weights_paths = sorted(glob(glob_weight_path[:-3] + '/*.pt'))
            if len(weights_paths) == 0:
                print(f'Warning: weights of {glob_weight_path} missing!')
                continue

            for weight_path in sorted(weights_paths):
                print(f'\nTest config: "{config_path}", weight "{weight_path}"\n')
                if len(weights_paths) != 1:
                    sub_sub_run_dir = osp.join(
                        sub_run_dir,
                        osp.splitext(osp.split(weight_path)[1])[0]
                    )
                else: sub_sub_run_dir = sub_run_dir
                if osp.exists(sub_sub_run_dir):
                    shutil.rmtree(sub_sub_run_dir)
                subprocess.run(
                    f'{sys.executable} test.py {config_path}'
                    f' test.weights_from_ckpt={weight_path}'
                    f' test.rundir_name={sub_sub_run_dir.replace("runs/", "", 1)}'
                    f' test.device={cuda_device}',
                    shell=True, check=True, executable=shutil.which('bash')
                )
                if len(weights_paths) != 1:
                    sub_metric_dict_path = osp.join(sub_sub_run_dir, 'metric_dict.json')
                    with open(sub_metric_dict_path, 'rb') as f:
                        sub_metric_dict = json.load(f)
                    for key in sub_metric_dict:
                        all_file_metric_dict[key] = concat_values_for_dict(
                            all_file_metric_dict[key] if key in all_file_metric_dict else {},
                            sub_metric_dict[key]
                        )

            print(f'config "{config_path}" Done')
            if len(weights_paths) != 1:
                with open(osp.join(sub_run_dir, metric_dict_filename), 'w') as f:
                    f.write(json.dumps(all_file_metric_dict, indent=2, sort_keys=False))
    print('All Done')


if __name__ == '__main__':
    test()
