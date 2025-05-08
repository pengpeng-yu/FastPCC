"""
This script is based on the commit 436376cd95dea17096a0463116eb718af8d40dff of OctAttention-lidar (lidar branch).
"""
import sys
import os
import os.path as osp
import json
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
sys.path.append(osp.dirname(osp.dirname(__file__)))
from scripts.log_extract_utils import read_file_list_with_rel_path
from scripts.script_config import metric_dict_filename, cuda_device, test_dir


octattention_path = '../OctAttention-lidar'
reuse_distortion_metric = 'runs/tests/tmc3_geo/octree/metric_dict_bak.json'
# I assume that you HAVE RUN test_tmc3.py on KITTI. So I simply reuse its distortion metrics.
if not osp.isfile(reuse_distortion_metric):
    reuse_distortion_metric = f'runs/tests/tmc3_geo/octree/{metric_dict_filename}'
file_lists = (
    'datasets/KITTI/sequences/test_list.txt',
    'datasets/KITTI/sequences/test_list_SparsePCGC110.txt',
)
# I look for keywords ('SparsePCGC') in the paths of file lists
#   to flag whether special handling is needed.
# Testing on the full KITTI dataset (> 20000 point clouds) is time-consuming,
#   so I skip the actual entropy coding to run faster.
output_dir = f'{test_dir}/OctAttention-lidar'
processes_num = 8
restart_steps = 10  # A simple workaround for the memory leaky issue (seems to be) of module "Octree"


sys.path.append(octattention_path)
import encoderTool
from networkTool import reload
from Octree import GenOctree, GenKparentSeq


def test():
    encoderTool.device = f"cuda:{cuda_device}"
    from octAttention import model  # will consume a little memory of cuda:0
    model = model.to(f"cuda:{cuda_device}")
    saved = reload(None, osp.join(octattention_path, 'modelsave/lidar/encoder_epoch_00801460.pth'))['encoder']
    model.load_state_dict(saved)

    print('Test OctAttention-LiDAR encoding')
    with open(reuse_distortion_metric, 'rb') as f:
        reused_metric_dict = json.load(f)
    os.makedirs(output_dir, exist_ok=True)

    all_file_metric_dict = {}
    data_queue = mp.Queue(32)
    subp = mp.Process(target=data_loader_manager, args=(data_queue,))
    subp.start()

    for file_list in file_lists:
        file_paths = read_file_list_with_rel_path(file_list)
        for file_path in tqdm(file_paths):
            bpp_list = []
            oct_data_list, target_path, target_points_num = data_queue.get()
            for rate_flag in range(1, 7, 1):
                bpp_list.append(encoderTool.compress(
                    oct_data_list[rate_flag - 1], None, model, False, print, False
                )[0] / target_points_num)
            sub_metric_dict = reused_metric_dict[target_path]
            del sub_metric_dict['encode time']
            del sub_metric_dict['decode time']
            del sub_metric_dict['encode memory']
            del sub_metric_dict['decode memory']
            sub_metric_dict['bpp'] = bpp_list
            all_file_metric_dict[target_path] = sub_metric_dict

    with open(osp.join(output_dir, metric_dict_filename), 'w') as f:
        f.write(json.dumps(all_file_metric_dict, indent=2, sort_keys=False))
    print('All Done')


def data_loader_manager(data_queue: mp.Queue):
    for file_list in file_lists:
        flag_sparsepcgc = 'SparsePCGC' in file_list
        file_paths = read_file_list_with_rel_path(file_list)
        pool = None
        for i in range((len(file_paths) // processes_num) + 1):
            if i % restart_steps == 0:
                if pool is not None: pool.close()
                pool = mp.Pool(processes_num)
            for res in pool.imap(
                    data_loader, [(file_path, flag_sparsepcgc)
                                  for file_path in file_paths[processes_num * i: processes_num * (i + 1)]]):
                data_queue.put(res)
        pool.close()


def data_loader(args):
    file_path, flag_sparsepcgc = args
    target_path = osp.splitext(file_path)[0] + ('_q1mm_n.ply' if flag_sparsepcgc else '_n.ply')
    org_xyz = np.fromfile(file_path, '<f4').reshape(-1, 4)[:, :3]
    if flag_sparsepcgc:
        # Quantization may reduce points num.
        target_points_num = np.unique((org_xyz * 1000).round(), axis=0).shape[0]
    else:
        target_points_num = org_xyz.shape[0]
    oct_data_list = []
    for rate_flag in range(1, 7, 1):
        # Same quantization as test_tmc3.py
        scale_for_kitti = (2 ** (int(rate_flag) + 10) - 1) / 400
        temp_xyz_q = org_xyz * scale_for_kitti
        temp_xyz_q.round(out=temp_xyz_q)
        temp_xyz_q = np.unique(temp_xyz_q, axis=0)

        oct_data_seq = dataPrepare(temp_xyz_q).astype(int)[:, -encoderTool.levelNumK:, 0:6]
        oct_data_list.append(oct_data_seq)
    return oct_data_list, target_path, target_points_num


def dataPrepare(p, qs=1, offset='min', qlevel=None, rotation=False, normalize=False):
    refPt = p
    if normalize is True:  # normalize pc to [-1,1]^3
        p = p - np.mean(p, axis=0)
        p = p / abs(p).max()
        refPt = p

    if rotation:
        refPt = refPt[:, [0, 2, 1]]
        refPt[:, 2] = - refPt[:, 2]

    if offset == 'min':
        offset = np.min(refPt, 0)

    points = refPt - offset

    if qlevel is not None:
        qs = (points.max() - points.min()) / (2 ** qlevel - 1)

    pt = np.round(points / qs)
    pt = np.unique(pt, axis=0)
    pt = pt.astype(int)

    code, Octree, QLevel = GenOctree(pt)
    DataSturct = GenKparentSeq(Octree, 4)

    patchFile = np.concatenate((np.expand_dims(DataSturct['Seq'], 2), DataSturct['Level'], DataSturct['Pos']), 2)
    return patchFile


if __name__ == '__main__':
    test()
