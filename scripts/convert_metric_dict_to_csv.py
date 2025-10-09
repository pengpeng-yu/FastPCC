import sys
import os
import os.path as osp
from glob import glob
import json

sys.path.append(osp.dirname(osp.dirname(__file__)))
from scripts.script_config import test_dir


def main(input_paths_, target_filenames=None):
    input_paths = []
    for p in input_paths_:
        input_paths.extend(sorted(glob(p)))
    assert len(input_paths) > 0

    os.makedirs(test_dir, exist_ok=True)
    fw = open(osp.join(test_dir, 'concat_metric.csv'), 'w')
    metric_dicts = []
    for p in input_paths:
        with open(p, 'rb') as fr:
            metric_dicts.append(json.load(fr))

    metric_keys = list(metric_dicts[0][next(iter(metric_dicts[0]))].keys())
    if 'compressed_bits' not in metric_keys:
        metric_keys.append('compressed_bits')
    fw.write(',' + ','.join([k.replace(',', '_') for k in metric_keys]))
    sample_keys = list(metric_dicts[0].keys())
    target_sample_keys = []
    if target_filenames is not None:
        for target_key in target_filenames:
            for sample_key in sample_keys:
                if target_key in sample_key:
                    target_sample_keys.append(sample_key)
                    break
    else:
        target_sample_keys = sample_keys
    for sample_key in target_sample_keys:
        for metric_dict in metric_dicts:
            fw.write('\n')
            fw.write(sample_key)
            for metric_key in metric_keys:
                fw.write(',')
                if metric_key == 'compressed_bits' and metric_key not in metric_dict[sample_key]:
                    fw.write(str(metric_dict[sample_key]['compressed_bytes'] * 8))
                else:
                    fw.write(str(metric_dict[sample_key][metric_key]))
    fw.close()


if __name__ == '__main__':
    input_paths = [
        'xxx/metric_dict.json',
        'xxx/metric_dict.json'
    ]
    target_filenames = [
        # 'RWTT_059_tomb_vox10',
        # 'RWTT_156_vishnu_vox10',
        # 'RWTT_211_foxstatue_vox10',
        # 'bicycle_vox11',
        # 'puppet_vox12',
        # 'QQdog_vox12',
        # 'exercise_00000001_vox11',
        # 'model_00000001_vox11',
        # 'sara_23_vox10',
        # 'phil_139_vox10',
        # 'helicopter_00000001_vox10',
        # 'clock1_vox11',
        # 'candlestick_vox11',
    ]
    main(input_paths, target_filenames)
