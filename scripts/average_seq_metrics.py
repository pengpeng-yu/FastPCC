import sys
import os
import os.path as osp
from glob import glob
import json
from copy import deepcopy
from typing import List

sys.path.append(osp.dirname(osp.dirname(__file__)))
from scripts.script_config import test_dir, metric_dict_filename


def main():
    """
    Averaging the metrics of each rate-distortion points for sequences.
    input: a list of metric dicts ('metric_dict.json').
    output: averaged dicts ('metric_dict.json', which will be used by compare_performance.py)
            and original dicts ('metric_dict_bak.json')
    If 'metric_dict_bak.json' already exists, I will use 'metric_dict_bak.json' as input.
    """
    average_targets = {
        lambda s: 'basketball_player_vox11' in s and 'Owlii' in s: 'basketball_player_vox11',
        lambda s: 'dancer_vox11' in s and 'Owlii' in s: 'dancer_vox11',
        lambda s: 'loot' in s and '8iVFBv2' in s: 'loot_vox10',
        lambda s: 'redandblack' in s and '8iVFBv2' in s: 'redandblack_vox10',
        lambda s: 'KITTI' in s and 'q1mm' not in s and 'AVS' not in s: 'KITTI',
        lambda s: 'KITTI' in s and 'q1mm' in s and 'AVS' not in s: 'KITTI q1mm',
        lambda s: 'Ford' in s and 'AVS' not in s: 'Ford',
        lambda s: 'KITTI/sequences/11' in s and 'q1mm' not in s and 'AVS' not in s: 'KITTI 11',
        lambda s: 'KITTI/sequences/12' in s and 'q1mm' not in s and 'AVS' not in s: 'KITTI 12',
        lambda s: 'KITTI/sequences/13' in s and 'q1mm' not in s and 'AVS' not in s: 'KITTI 13',
        lambda s: 'KITTI/sequences/14' in s and 'q1mm' not in s and 'AVS' not in s: 'KITTI 14',
        lambda s: 'KITTI/sequences/15' in s and 'q1mm' not in s and 'AVS' not in s: 'KITTI 15',
        lambda s: 'KITTI/sequences/16' in s and 'q1mm' not in s and 'AVS' not in s: 'KITTI 16',
        lambda s: 'KITTI/sequences/17' in s and 'q1mm' not in s and 'AVS' not in s: 'KITTI 17',
        lambda s: 'KITTI/sequences/18' in s and 'q1mm' not in s and 'AVS' not in s: 'KITTI 18',
        lambda s: 'KITTI/sequences/19' in s and 'q1mm' not in s and 'AVS' not in s: 'KITTI 19',
        lambda s: 'KITTI/sequences/20' in s and 'q1mm' not in s and 'AVS' not in s: 'KITTI 20',
        lambda s: 'KITTI/sequences/21' in s and 'q1mm' not in s and 'AVS' not in s: 'KITTI 21',
        lambda s: 'KITTI/sequences/11' in s and 'q1mm' in s and 'AVS' not in s: 'KITTI q1mm 11',
        lambda s: 'KITTI/sequences/12' in s and 'q1mm' in s and 'AVS' not in s: 'KITTI q1mm 12',
        lambda s: 'KITTI/sequences/13' in s and 'q1mm' in s and 'AVS' not in s: 'KITTI q1mm 13',
        lambda s: 'KITTI/sequences/14' in s and 'q1mm' in s and 'AVS' not in s: 'KITTI q1mm 14',
        lambda s: 'KITTI/sequences/15' in s and 'q1mm' in s and 'AVS' not in s: 'KITTI q1mm 15',
        lambda s: 'KITTI/sequences/16' in s and 'q1mm' in s and 'AVS' not in s: 'KITTI q1mm 16',
        lambda s: 'KITTI/sequences/17' in s and 'q1mm' in s and 'AVS' not in s: 'KITTI q1mm 17',
        lambda s: 'KITTI/sequences/18' in s and 'q1mm' in s and 'AVS' not in s: 'KITTI q1mm 18',
        lambda s: 'KITTI/sequences/19' in s and 'q1mm' in s and 'AVS' not in s: 'KITTI q1mm 19',
        lambda s: 'KITTI/sequences/20' in s and 'q1mm' in s and 'AVS' not in s: 'KITTI q1mm 20',
        lambda s: 'KITTI/sequences/21' in s and 'q1mm' in s and 'AVS' not in s: 'KITTI q1mm 21',
        lambda s: 'AVS' in s and 'kitti' in s: 'AVS KITTI',
        lambda s: 'AVS' in s and 'kitti' in s and '/11/' in s: 'AVS KITTI 11',
        lambda s: 'AVS' in s and 'kitti' in s and '/12/' in s: 'AVS KITTI 12',
        lambda s: 'AVS' in s and 'kitti' in s and '/13/' in s: 'AVS KITTI 13',
        lambda s: 'AVS' in s and 'kitti' in s and '/14/' in s: 'AVS KITTI 14',
        lambda s: 'AVS' in s and 'kitti' in s and '/15/' in s: 'AVS KITTI 15',
        lambda s: 'AVS' in s and 'kitti' in s and '/16/' in s: 'AVS KITTI 16',
        lambda s: 'AVS' in s and 'kitti' in s and '/17/' in s: 'AVS KITTI 17',
        lambda s: 'AVS' in s and 'kitti' in s and '/18/' in s: 'AVS KITTI 18',
        lambda s: 'AVS' in s and 'kitti' in s and '/19/' in s: 'AVS KITTI 19',
        lambda s: 'AVS' in s and 'kitti' in s and '/20/' in s: 'AVS KITTI 20',
        lambda s: 'AVS' in s and 'kitti' in s and '/21/' in s: 'AVS KITTI 21',
        lambda s: 'AVS' in s and 'Ford_03' in s: 'AVS Ford 03',
        lambda s: 'AVS' in s and 'Livox_02' in s: 'AVS Livox 02',
        lambda s: 'AVS' in s and 'kitti_det_val' in s: 'AVS KITTI VAL',
    }
    input_files = (
        f'{test_dir}/convolutional/lossy_coord_v2/baseline_kitti_q1mm_r*/{metric_dict_filename}',
        f'{test_dir}/convolutional/lossy_coord_v2/baseline_kitti_r*/{metric_dict_filename}',
        f'{test_dir}/convolutional/lossy_coord_lossy_color/baseline_r*/{metric_dict_filename}',
        f'{test_dir}/convolutional/lossl_coord/*/{metric_dict_filename}',
        f'{test_dir}/convolutional/lossl_coord_unicorn_test_cond/*/{metric_dict_filename}',
        f'{test_dir}/tmc3_geo/octree/{metric_dict_filename}',
        f'{test_dir}/tmc3_geo/octree/Ford_low_rate/{metric_dict_filename}',
        f'{test_dir}/tmc3/octree-predlift/{metric_dict_filename}',
        f'{test_dir}/tmc3/octree-raht/{metric_dict_filename}',
        f'{test_dir}/tmc2/{metric_dict_filename}',
        f'{test_dir}/OctAttention-lidar/{metric_dict_filename}',
        f'{test_dir}/pcc-geo-color/{metric_dict_filename}',
        f'{test_dir}/convolutional/lossl_coord/avs*/kitti_ford_livox_test_r*/{metric_dict_filename}',
    )
    for files in input_files:
        files = sorted(glob(files))
        for file in files:
            outfile = osp.splitext(file)[0] + '_bak' + osp.splitext(file)[1]
            if osp.isfile(outfile):
                print(f'{outfile} already exists!')
                infile = outfile
            else:
                infile = file
            with open(infile, 'rb') as f:
                try:
                    metric_dict = json.load(f)
                except Exception as e:
                    print(infile)
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
                                new_metric_dict[target][metric_key] = deepcopy(metric_values)
                            else:
                                if isinstance(metric_values, List):
                                    for i, _ in enumerate(metric_values):
                                        new_metric_dict[target][metric_key][i] += _
                                else:
                                    new_metric_dict[target][metric_key] += metric_values
                        counts[target] += 1
            for k, v in counts.items():
                if k not in new_metric_dict: continue
                for metric_key, metric_values in new_metric_dict[k].items():
                    if metric_key == 'compressed_bytes':
                        tmp_compressed_bytes = deepcopy(new_metric_dict[k][metric_key])
                    if isinstance(new_metric_dict[k][metric_key], List):
                        for i in range(len(new_metric_dict[k][metric_key])):
                            new_metric_dict[k][metric_key][i] /= v
                    else:
                        new_metric_dict[k][metric_key] /= v
                new_metric_dict[k]['total_compressed_bytes'] = tmp_compressed_bytes
            if not osp.isfile(outfile):
                os.rename(infile, outfile)
            with open(file, 'w') as f:
                f.write(json.dumps(new_metric_dict, indent=2, sort_keys=False))
    print('All Done')


if __name__ == '__main__':
    main()
