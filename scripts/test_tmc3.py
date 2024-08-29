"""
This script is based on the version 26rc2 of mpeg-pcc-tmc13.
"""
import math
from glob import glob
import sys
import os
import os.path as osp
import subprocess
import json
import multiprocessing as mp

import numpy as np

sys.path.append(osp.dirname(osp.dirname(__file__)))
from lib.metrics.pc_error_wapper import mpeg_pc_error
from scripts.log_extract_utils import *
from scripts.shared_config import pc_error_path, metric_dict_filename, test_dir
from lib.data_utils.utils import write_ply_file


geo_only = True
single_frame_only = True

processes_num = mp.cpu_count() // 4
tmc3_path = '../mpeg-pcc-tmc13/build/tmc3/tmc3'

if single_frame_only:
    file_lists = (
        'datasets/MPEG_GPCC_CTC/Dense/Dense_16384.txt',
        "datasets/MPEG_GPCC_CTC/Solid/Solid_4096.txt",
        "datasets/MPEG_GPCC_CTC/Solid/Solid_2048.txt",
        "datasets/MPEG_GPCC_CTC/Solid/Solid_1024.txt",
        "datasets/MVUB/list.txt",
        'datasets/KITTI/sequences/test_list.txt',
        'datasets/KITTI/sequences/test_list_SparsePCGC110.txt',
    )
    # ↑ Two different calculation of distortion metrics on KITTI.
    # I put them in a single json file since they have different reconstruction targets (*_n.ply / *_q1mm_n.ply)
    #   and won't cause key conflicts.
    # I look for keywords ('MVUB'/'KITTI'/'SparsePCGC') in the paths of file lists
    #   to flag whether special handling is needed.
    resolutions = (16384, 4096, 2048, 1024, 512, 59.70 + 1, 30000 + 1)
else:
    file_lists = (
        'datasets/Owlii/list_basketball_player_dancer.txt',
        'datasets/8iVFBv2/list_loot_redandblack.txt'
    )
    resolutions = (2048, 1024)

if geo_only:
    config_dirs = (
        '../mpeg-pcc-tmc13/cfg/octree-predlift/lossy-geom-lossy-attrs',
    )
    output_dirs = (
        f'{test_dir}/tmc3_geo/octree',
    )
else:
    config_dirs = (
        '../mpeg-pcc-tmc13/cfg/octree-predlift/lossy-geom-lossy-attrs',
        '../mpeg-pcc-tmc13/cfg/octree-raht/lossy-geom-lossy-attrs'
    )
    output_dirs = (
        f'{test_dir}/tmc3/octree-predlift',
        f'{test_dir}/tmc3/octree-raht',
    )
assert len(file_lists) == len(resolutions)
assert len(config_dirs) == len(output_dirs)


class TMC3LogExtractor(LogExtractor):
    default_enc_log_mappings: log_mappings_type = {
        'Total bitstream size': ('bits', lambda l: float(l.split()[-2]) * 8),
        'Processing time (user)': ('encode time', lambda l: float(l.split()[-2])),
        'Peak memory': ('encode memory', lambda l: int(l.split()[-2]))
    }
    default_dec_log_mappings: log_mappings_type = {
        'Processing time (user)': ('decode time', lambda l: float(l.split()[-2])),
        'Peak memory': ('decode memory', lambda l: int(l.split()[-2]))
    }

    def __init__(self):
        self.enc_log_mappings = self.default_enc_log_mappings
        self.dec_log_mappings = self.default_dec_log_mappings
        super(TMC3LogExtractor, self).__init__()

    def extract_enc_log(self, log: str):
        return self.extract_log(log, self.enc_log_mappings)

    def extract_dec_log(self, log: str):
        return self.extract_log(log, self.dec_log_mappings)


def get_tmc3_octree_positionQuantizationScale(src_geometry_precision, rate_flag):
    # From octree-liftt-ctc-lossy-geom-lossy-attrs.yaml
    rp = 6 - int(rate_flag)
    gp = src_geometry_precision
    p_min = max(gp - 9, 7)
    start = min(1, gp - (p_min + 6))
    step = max(1., (min(gp - 1, p_min + 7) - p_min) / 5)
    y = start + round(rp * step)
    div = 1 << (abs(y) + 1)
    return ((1 - 2 * (y < 0)) % div) / div


def get_tmc3_trisoup_trisoupNodeSizeLog2(rate_flag):
    # From trisoup-liftt-ctc-lossy-geom-lossy-attrs.yaml
    d = {1: 5, 2: 4, 3: 3, 4: 2}
    return d[int(rate_flag)]


def test_intra():
    print(f'Test tmc3 {"geo " if geo_only else ""}coding')
    pool = mp.get_context('forkserver').Pool(processes_num)

    for config_dir, output_dir in zip(config_dirs, output_dirs):
        default_config_paths = glob(osp.join(config_dir, 'longdress_vox10_1300', '*', 'encoder.cfg'))
        default_config_paths.sort()
        print(f'Test config: "{config_dir}"')
        print(f'Output to "{output_dir}"')
        all_file_metric_dict: all_file_metric_dict_type = {}
        all_file_run_res = []

        for resolution, file_list in zip(resolutions, file_lists):
            file_paths = read_file_list_with_rel_path(file_list)
            for file_path in file_paths:
                all_file_run_res.append(pool.apply_async(
                    run_single_file,
                    (file_path, resolution, file_list, default_config_paths, config_dir, output_dir)
                ))
        for run_res in all_file_run_res:
            ret = run_res.get()
            if ret is not None:
                all_file_metric_dict[ret[0]] = ret[1]

        print(f'{config_dir} Done')
        with open(osp.join(output_dir, metric_dict_filename), 'w') as f:
            f.write(json.dumps(all_file_metric_dict, indent=2, sort_keys=False))
    pool.close()
    print('All Done')


def run_single_file(file_path, resolution, file_list, default_config_paths, config_dir, output_dir):
    log_extractor = TMC3LogExtractor()
    sub_metric_dict: one_file_metric_dict_type = {}
    file_basename = osp.splitext(osp.split(file_path)[1])[0]
    if 'queen/frame' in file_path:
        # queen/frame_xxx -> queen_0200
        file_basename = f'queen_{file_path.rsplit("_", 1)[1]}'
    config_paths = glob(osp.join(config_dir, file_basename.lower(), '*', 'encoder.cfg'))
    flag_mvub = False
    flag_kitti = False
    flag_sparsepcgc = False
    if len(config_paths) == 0:
        if 'MVUB' in file_list:
            flag_mvub = True
            config_paths = default_config_paths
        elif 'KITTI' in file_list:
            flag_kitti = True
            if 'SparsePCGC' in file_list:
                flag_sparsepcgc = True
            config_paths = default_config_paths
        else:
            assert single_frame_only is False
            # e.g., filename: basketball_player_vox11_xxx -> config: basketball_player_vox11
            config_basename = file_basename.lower().rsplit('_', 1)[0]
            config_paths = glob(osp.join(config_dir, config_basename, '*', 'encoder.cfg'))
            # e.g., filename: loot_vox10_xxx -> config: loot_vox10_1200
            if len(config_paths) == 0:
                config_paths = glob(osp.join(config_dir, config_basename + '*', '*', 'encoder.cfg'))
            if len(config_paths) != 6:
                print(f'\nlen(config_paths) == {len(config_paths)} != 6 for {file_basename}\n')
                return None
    config_paths.sort()
    sub_output_dir = osp.join(output_dir, osp.splitext(file_path)[0])
    if flag_sparsepcgc:
        sub_output_dir += '_q1mm'
    os.makedirs(sub_output_dir, exist_ok=True)
    if flag_kitti:
        org_xyz = np.fromfile(file_path, '<f4').reshape(-1, 4)[:, :3]

    for config_path in config_paths:
        rate_flag = osp.split(osp.split(config_path)[0])[1]
        print(f'    Test file {file_path}, res {resolution}, {rate_flag}')
        command_enc = \
            f'{tmc3_path}' \
            f' --config={config_path}' \
            f' --disableAttributeCoding={1 if geo_only else 0}' \
            f' --uncompressedDataPath={file_path}' \
            f' --compressedStreamPath={osp.join(sub_output_dir, f"{rate_flag}.bin")}'
        org_pc_for_pc_error = file_path
        if flag_mvub:
            if 'octree' in config_dir:
                command_enc += \
                    f' --positionQuantizationScale=' \
                    f'{get_tmc3_octree_positionQuantizationScale(round(math.log2(resolution)), rate_flag[1:])}'
            elif 'trisoup' in config_dir:
                command_enc += \
                    f' --trisoupNodeSizeLog2=' \
                    f'{get_tmc3_trisoup_trisoupNodeSizeLog2(rate_flag[1:])}'
            else:
                raise NotImplementedError
        if flag_kitti:
            scale_for_kitti = (2 ** (int(rate_flag[1:]) + 9) - 1) / 400
            temp_xyz_q = org_xyz * scale_for_kitti
            temp_xyz_q.round(out=temp_xyz_q)
            temp_xyz_q = np.unique(temp_xyz_q, axis=0)
            temp_file_path_for_kitti = osp.join(sub_output_dir, f"{rate_flag}_scaled_input.ply")
            write_ply_file(temp_xyz_q, temp_file_path_for_kitti)
            command_enc += f' --uncompressedDataPath={temp_file_path_for_kitti} --positionQuantizationScale=1'
            if not flag_sparsepcgc:
                org_pc_for_pc_error = osp.splitext(file_path)[0] + '_n.ply'
                if not osp.isfile(org_pc_for_pc_error):
                    write_ply_file(org_xyz, org_pc_for_pc_error, estimate_normals=True)
            else:
                org_pc_for_pc_error = osp.splitext(file_path)[0] + '_q1mm_n.ply'
                if not osp.isfile(org_pc_for_pc_error):
                    write_ply_file(np.unique((org_xyz * 1000).round(), axis=0),
                                   org_pc_for_pc_error, estimate_normals=True)

        subp_enc = subprocess.run(
            command_enc, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True, check=True, text=True
        )
        with open(osp.join(sub_output_dir, f'log_{rate_flag}_enc.txt'), 'w') as f:
            f.write(subp_enc.stdout)
        sub_metric_dict = concat_values_for_dict(
            sub_metric_dict,
            log_extractor.extract_enc_log(subp_enc.stdout), False
        )
        command_dec = \
            f'{tmc3_path}' \
            f' --mode=1' \
            f' --outputBinaryPly=1' \
            f' --compressedStreamPath={osp.join(sub_output_dir, f"{rate_flag}.bin")}' \
            f' --reconstructedDataPath={osp.join(sub_output_dir, f"{rate_flag}_recon.ply")}'
        if flag_kitti:
            os.remove(temp_file_path_for_kitti)
            command_dec += ' --outputScaling=1'
            if not flag_sparsepcgc:
                command_dec += f' --outputUnitLength={scale_for_kitti}'
            else:
                command_dec += f' --outputUnitLength={scale_for_kitti / 1000}'
        subp_dec = subprocess.run(
            command_dec, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True, check=True, text=True
        )
        with open(osp.join(sub_output_dir, f'log_{rate_flag}_dec.txt'), 'w') as f:
            f.write(subp_dec.stdout)
        sub_metric_dict = concat_values_for_dict(
            sub_metric_dict,
            log_extractor.extract_dec_log(subp_dec.stdout), False
        )
        sub_metric_dict = concat_values_for_dict(
            sub_metric_dict,
            mpeg_pc_error(
                org_pc_for_pc_error,
                osp.join(sub_output_dir, f'{rate_flag}_recon.ply'), resolution,
                color=False if geo_only else True,
                command=pc_error_path
            ), False
        )
    sub_metric_dict['bpp'] = [bits / org_points_num for bits, org_points_num in zip(
        sub_metric_dict['bits'], sub_metric_dict['org points num'])]
    del sub_metric_dict['bits'], sub_metric_dict['org points num']
    with open(osp.join(sub_output_dir, metric_dict_filename), 'w') as f:
        f.write(json.dumps(sub_metric_dict, indent=2, sort_keys=False))
    return org_pc_for_pc_error, sub_metric_dict


if __name__ == '__main__':
    test_intra()
