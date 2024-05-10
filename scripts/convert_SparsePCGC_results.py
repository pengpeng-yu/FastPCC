import os

import pandas as pd
import json


def convert_dense_lossy():
    file_paths = [
        'datasets/MPEG_GPCC_CTC/Solid/longdress_vox10_1300.ply',
        "datasets/MPEG_GPCC_CTC/Solid/loot_vox10_1200.ply",
        "datasets/MPEG_GPCC_CTC/Solid/queen_0200.ply",
        "datasets/MPEG_GPCC_CTC/Solid/redandblack_vox10_1550.ply",
        "datasets/MPEG_GPCC_CTC/Solid/soldier_vox10_0690.ply",
        "datasets/MPEG_GPCC_CTC/Solid/basketball_player_vox11_00000200.ply",
        "datasets/MPEG_GPCC_CTC/Solid/dancer_vox11_00000001.ply"
    ]
    key_maps = {
        'encode time': 'R{i}_enc_time',
        'decode time': 'R{i}_dec_time',
        'mseF,PSNR (p2point)': 'R{i}_mseF,PSNR (p2point)',
        'mseF,PSNR (p2plane)': 'R{i}_mseF,PSNR (p2plane)',
        'bpp': 'R{i}_bpp'
    }
    path = '../SparsePCGC/results/dense_lossy/ours.csv'
    res = pd.read_csv(path)
    d = {}

    for idx in range(res.shape[0]):
        line = res.iloc[idx]
        sub_d = {}
        for key, tgt_key in key_maps.items():
            sub_d[key] = []
            for i in range(99):
                if tgt_key.format(i=i) in line:
                    sub_d[key].append(line[tgt_key.format(i=i)].item())
                else:
                    break
        filename = line.filedir.rsplit('/', 1)[1]
        for file_path in file_paths:
            if filename in file_path:
                d[file_path] = sub_d
                break

    os.makedirs('runs/tests/SparsePCGC/dense_lossy', exist_ok=True)
    with open('runs/tests/SparsePCGC/dense_lossy/metric_dict.json', 'w') as f:
        f.write(json.dumps(d, indent=2, sort_keys=False))

    print('Done')


def convert_kitti_lossy():
    key_maps = {
        'encode time': 'R{i}_enc_time',
        'decode time': 'R{i}_dec_time',
        'mseF,PSNR (p2point)': 'R{i}_mseF,PSNR (p2point)',
        'mseF,PSNR (p2plane)': 'R{i}_mseF,PSNR (p2plane)',
        'bpp': 'R{i}_bpp'
    }
    path = '../SparsePCGC/results/comparison_gpcc_sparse/ours_lossy_kitti_data110.csv'
    df = pd.read_csv(path)
    d = {k: [] for k in key_maps}

    for key, tgt_key in key_maps.items():
        for i in range(99):
            tgt_key_i = tgt_key.format(i=i)
            if tgt_key_i in df:
                d[key].append(df[tgt_key_i].mean())

    os.makedirs('runs/tests/SparsePCGC/kitti_q1mm', exist_ok=True)
    with open('runs/tests/SparsePCGC/kitti_q1mm/metric_dict.json', 'w') as f:
        f.write(json.dumps({'KITTI q1mm': d}, indent=2, sort_keys=False))

    print('Done')


if __name__ == '__main__':
    convert_dense_lossy()
    convert_kitti_lossy()
