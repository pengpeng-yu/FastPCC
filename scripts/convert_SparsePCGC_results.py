import pandas as pd
import json
import os.path as osp


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
path = 'tmp/SparsePCGCResults/ours.xlsx'
res = pd.read_excel(path)
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

with open(osp.join(osp.split(path)[0], 'metric_dict.json'), 'w') as f:
    f.write(json.dumps(d, indent=2, sort_keys=False))

print('Done')
