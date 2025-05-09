"""
This script is based on the commit 19c881937d2204700262a72b6db3963c2959dd73 of Unicorn.
Converting the csv format of rate-distortion used in Unicorn to the json format I use.
"""
import os
import os.path as osp
import json

import pandas as pd

from scripts.script_config import metric_dict_filename


Unicorn_path = '../Unicorn'


def convert_intra_kitti_ford_lossy():
    key_maps = {
        'encode time': 'enc_time',
        'decode time': 'dec_time',
        'mseF,PSNR (p2point)': 'mseF,PSNR (p2point)',
        'mseF,PSNR (p2plane)': 'mseF,PSNR (p2plane)',
        'bpp': 'bpp'
    }
    df = pd.read_csv(osp.join(Unicorn_path, 'results/Unicorn_v1/PCGC/csvfiles/lidar/intra/kitti1mm.csv'))
    kitti = {k: [] for k in key_maps}
    for key, tgt_key in key_maps.items():
        kitti[key] = df[tgt_key].values.tolist()[1:]

    df = pd.read_csv(osp.join(Unicorn_path, 'results/Unicorn_v1/PCGC/csvfiles/lidar/intra/ford1mm.csv'))
    ford = {k: [] for k in key_maps}
    for key, tgt_key in key_maps.items():
        ford[key] = df[tgt_key].values.tolist()[1:]


    os.makedirs('runs/tests/Unicorn/intra', exist_ok=True)
    with open(f'runs/tests/Unicorn/intra/{metric_dict_filename}', 'w') as f:
        f.write(json.dumps(
            {
                'KITTI q1mm': kitti,
                'Ford': ford
            }, indent=2, sort_keys=False))
    print('Done')


if __name__ == '__main__':
    convert_intra_kitti_ford_lossy()
