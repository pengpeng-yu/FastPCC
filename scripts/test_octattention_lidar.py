"""
This script is based on the commit 505ceb855d4d231e5fcb8030354167cc75937d80 of OctAttention-lidar.
"""
import sys
import os
import os.path as osp
import shutil
import subprocess
import json

import numpy as np
import open3d as o3d

sys.path.append(osp.dirname(osp.dirname(__file__)))
from scripts.log_extract_utils import read_file_list_with_rel_path
from scripts.shared_config import metric_dict_filename, cuda_device, test_dir
from lib.data_utils.utils import write_ply_file


octattention_path = '../OctAttention-lidar'
reuse_distortion_metric = 'runs/tests/tmc3_geo/octree/metric_dict.json'  # or metric_dict_bak.json?
# I assume that you HAVE RUN test_tcm3.py on KITTI. So I simply reuse its distortion metrics.
file_lists = (
    'datasets/KITTI/sequences/test_list.txt',
    'datasets/KITTI/sequences/test_list_SparsePCGC110.txt'
)
resolutions = (59.70 + 1, 30000 + 1)
output_dir = f'{test_dir}/OctAttention-lidar'


def test():
    print('Test OctAttention-LiDAR encoding')
    with open(reuse_distortion_metric, 'rb') as f:
        reused_metric_dict = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    octattention_input = osp.join(octattention_path, 'file/Ply/2851.ply')
    octattention_input_bak = osp.join(octattention_path, 'file/Ply/2851_bak.ply')
    octattention_output = osp.join(octattention_path, 'Exp/Obj/data/2851.bin')
    if osp.isfile(octattention_input) and not osp.isfile(octattention_input_bak):
        shutil.move(octattention_input, octattention_input_bak)

    all_file_metric_dict = {}
    for resolution, file_list in zip(resolutions, file_lists):
        file_paths = read_file_list_with_rel_path(file_list)
        for file_path in file_paths:
            target_path = osp.splitext(file_path)[0] + ('_q1mm_n.ply' if 'SparsePCGC' in file_path else '_n.ply')
            target_points_num = np.asarray(o3d.io.read_point_cloud(target_path).points).shape[0]
            bpp_list = []
            for rate_flag in range(1, 7, 1):
                print(f'\n    Test file {file_path}, res {resolution}, {rate_flag}\n')
                # Same quantization as test_tmc3.py
                temp_xyz = np.fromfile(file_path, '<f4').reshape(-1, 4)[:, :3]
                scale_for_kitti = (2 ** (int(rate_flag) + 9) - 1) / 400
                temp_xyz *= scale_for_kitti
                temp_xyz.round(out=temp_xyz)
                temp_xyz = np.unique(temp_xyz, axis=0)
                write_ply_file(temp_xyz, octattention_input)

                subprocess.run(
                    f'export CUDA_VISIBLE_DEVICES={cuda_device};'
                    f'cd {octattention_path};'
                    f'{sys.executable} encoder.py',
                    shell=True, check=True, executable=shutil.which('bash')
                )
                bpp_list.append(osp.getsize(octattention_output) * 8 / target_points_num)
            all_file_metric_dict[target_path] = reused_metric_dict[target_path]
            all_file_metric_dict[target_path]['bpp'] = bpp_list

    shutil.move(octattention_input_bak, octattention_input)
    with open(osp.join(output_dir, metric_dict_filename), 'w') as f:
        f.write(json.dumps(all_file_metric_dict, indent=2, sort_keys=False))
    print('All Done')


if __name__ == '__main__':
    test()
