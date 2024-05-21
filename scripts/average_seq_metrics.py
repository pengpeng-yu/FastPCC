import sys
import os
import os.path as osp
from glob import glob
import json
from typing import List

sys.path.append(osp.dirname(osp.dirname(__file__)))
from scripts.shared_config import test_dir, metric_dict_filename


def main():
    """
    Averaging the metrics of each rate-distortion points for sequences.
    input: a list of metric dicts ('metric_dict.json').
    output: averaged dicts ('metric_dict.json', which will be used by compare_performance.py)
            and original dicts ('metric_dict_bak.json')
    If 'metric_dict_bak.json' already exists, I will skip that folder to avoid the original dicts being overwritten.
    """
    average_targets = {
        lambda s: 'basketball_player_vox11' in s and 'Owlii' in s: 'basketball_player_vox11',
        lambda s: 'dancer_vox11' in s and 'Owlii' in s: 'dancer_vox11',
        lambda s: 'loot' in s and '8iVFBv2' in s: 'loot_vox10',
        lambda s: 'redandblack' in s and '8iVFBv2' in s: 'redandblack_vox10',
        lambda s: 'KITTI' in s and 'q1mm' not in s: 'KITTI',
        lambda s: 'KITTI' in s and 'q1mm' in s: 'KITTI q1mm'
    }
    input_files = (
        f'{test_dir}/convolutional/lossy_coord_v2/baseline_kitti_q1mm_r*/{metric_dict_filename}',
        f'{test_dir}/convolutional/lossy_coord_v2/baseline_kitti_r*/{metric_dict_filename}',
        f'{test_dir}/convolutional/lossy_coord_lossy_color/baseline_r*/{metric_dict_filename}',
        f'{test_dir}/tmc3_geo/octree/{metric_dict_filename}',
        f'{test_dir}/tmc3/octree-predlift/{metric_dict_filename}',
        f'{test_dir}/tmc3/octree-raht/{metric_dict_filename}',
        f'{test_dir}/tmc2/{metric_dict_filename}',
        f'{test_dir}/OctAttention-lidar/{metric_dict_filename}',
        f'{test_dir}/pcc-geo-color/{metric_dict_filename}',
    )
    for files in input_files:
        files = sorted(glob(files))
        for file in files:
            outfile = osp.splitext(file)[0] + '_bak' + osp.splitext(file)[1]
            if osp.isfile(outfile):
                print(f'{outfile} already exists!')
                print(f'Skip {file}\n')
                continue
            with open(file, 'rb') as f:
                try:
                    metric_dict = json.load(f)
                except Exception as e:
                    print(file)
                    raise e
            new_metric_dict = {}
            counts = {v: 0 for v in average_targets.values()}
            for sample_name, sample_metric in metric_dict.items():
                for target_fn, target in average_targets.items():
                    if target_fn(sample_name):
                        if target not in new_metric_dict:
                            new_metric_dict[target] = {}
                        for metric_key, metric_values in sample_metric.items():
                            if metric_key not in new_metric_dict[target]:
                                new_metric_dict[target][metric_key] = metric_values
                            else:
                                if isinstance(metric_values, List):
                                    for i, _ in enumerate(metric_values):
                                        new_metric_dict[target][metric_key][i] += _
                                else:
                                    new_metric_dict[target][metric_key] += metric_values
                        counts[target] += 1
                        break
                else:
                    new_metric_dict[sample_name] = sample_metric
            for k, v in counts.items():
                if k not in new_metric_dict: continue
                for metric_key, metric_values in new_metric_dict[k].items():
                    if isinstance(new_metric_dict[k][metric_key], List):
                        for i in range(len(new_metric_dict[k][metric_key])):
                            new_metric_dict[k][metric_key][i] /= v
                    else:
                        new_metric_dict[k][metric_key] /= v
            outfile = osp.splitext(file)[0] + '_bak' + osp.splitext(file)[1]
            os.rename(file, outfile)
            with open(file, 'w') as f:
                f.write(json.dumps(new_metric_dict, indent=2, sort_keys=False))
    print('All Done')


if __name__ == '__main__':
    main()
