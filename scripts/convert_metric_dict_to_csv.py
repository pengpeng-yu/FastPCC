import os
import os.path as osp
from glob import glob
import json

from scripts.script_config import test_dir


def main(input_paths_):
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
    fw.write(',' + ','.join(metric_keys))
    sample_keys = list(metric_dicts[0].keys())
    for sample_key in sample_keys:
        for metric_dict in metric_dicts:
            fw.write('\n')
            fw.write(sample_key)
            for metric_key in metric_keys:
                fw.write(',')
                fw.write(str(metric_dict[sample_key][metric_key]))
    fw.close()


if __name__ == '__main__':
    input_paths = ['xxx/metric_dict.json', 'xxx/metric_dict.json']
    main(input_paths)
