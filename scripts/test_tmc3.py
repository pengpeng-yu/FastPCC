"""
This script is based on commit c3c9798a0f63970bd17ce191900ded478a8aa0f6 of mpeg-pcc-tmc13.
"""
import math
from glob import glob
import os
import subprocess
import json
import multiprocessing as mp

from lib.metrics.pc_error_wapper import mpeg_pc_error
from scripts.log_extract_utils import *
from scripts.shared_config import pc_error_path, metric_dict_filename

geo_only = True
single_frame_only = True

tmc3_path = '../mpeg-pcc-tmc13/build/tmc3/tmc3'

if single_frame_only:
    file_lists = (
        "datasets/MVUB/list.txt",
        "datasets/MPEG_GPCC_CTC/Solid/Solid_1024.txt",
        "datasets/MPEG_GPCC_CTC/Solid/Solid_2048.txt",
        "datasets/MPEG_GPCC_CTC/Solid/Solid_4096.txt"
    )
    resolutions = (512, 1024, 2048, 4096)
    results_dir = "tests"
else:
    file_lists = (
        'datasets/Owlii/list_basketball_player_dancer_all.txt',
        'datasets/8iVFBv2/list_all.txt'
    )
    resolutions = (2048, 1024)
    results_dir = "tests_seq"

if geo_only:
    config_dirs = (
        '../mpeg-pcc-tmc13/cfg/octree-predlift/lossy-geom-lossy-attrs',
        '../mpeg-pcc-tmc13/cfg/trisoup-predlift/lossy-geom-lossy-attrs'
    )
    output_dirs = (
        f'runs/{results_dir}/tmc3_geo/octree',
        f'runs/{results_dir}/tmc3_geo/trisoup',
    )
else:
    config_dirs = (
        '../mpeg-pcc-tmc13/cfg/octree-predlift/lossy-geom-lossy-attrs',
        '../mpeg-pcc-tmc13/cfg/octree-raht/lossy-geom-lossy-attrs'
    )
    output_dirs = (
        f'runs/{results_dir}/tmc3/octree-predlift',
        f'runs/{results_dir}/tmc3/octree-raht',
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


def test_geo_intra(processes_num, immediate_dump):
    print(f'Test tmc3 {"geo " if geo_only else ""}coding')
    pool = mp.Pool(processes_num)

    for config_dir, output_dir in zip(config_dirs, output_dirs):
        default_config_paths = glob(osp.join(config_dir, 'basketball_player_vox11_00000200', '*', 'encoder.cfg'))
        default_config_paths.sort()
        print(f'Test config: "{config_dir}"')
        print(f'Output to "{output_dir}"')
        all_file_metric_dict: all_file_metric_dict_type = {}
        all_file_run_res = {}

        for resolution, file_list in zip(resolutions, file_lists):
            file_paths = read_file_list_with_rel_path(file_list)
            for file_path in file_paths:
                all_file_run_res[file_path] = pool.apply_async(
                    run_single_file,
                    (file_path, resolution, file_list, default_config_paths, config_dir, output_dir)
                )
        for file_path, run_res in all_file_run_res.items():
            ret = run_res.get()
            if ret is not None:
                all_file_metric_dict[file_path] = ret
                if immediate_dump:
                    with open(osp.join(output_dir, metric_dict_filename), 'w') as f:
                        f.write(json.dumps(all_file_metric_dict, indent=2, sort_keys=False))

        print(f'{config_dir} Done')
        with open(osp.join(output_dir, metric_dict_filename), 'w') as f:
            f.write(json.dumps(all_file_metric_dict, indent=2, sort_keys=False))
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
    if len(config_paths) == 0:
        if 'MVUB' in file_list:
            flag_mvub = True
            config_paths = default_config_paths
        else:
            assert single_frame_only is False
            # e.g. basketball_player_vox11_xxx -> basketball_player_vox11_00000200
            config_paths = glob(osp.join(
                config_dir, file_basename.lower().rsplit('_', 1)[0] + '*', '*', 'encoder.cfg'))
            if len(config_paths) == 0:
                print(f'\nlen(config_paths) == 0 for : {file_basename}\n')
                return None
    config_paths.sort()
    sub_output_dir = osp.join(output_dir, file_basename)
    os.makedirs(sub_output_dir, exist_ok=True)
    for config_path in config_paths:
        rate_flag = osp.split(osp.split(config_path)[0])[1]
        print(f'    Test file {file_path}, res {resolution}, {rate_flag}')
        command_enc = \
            f'{tmc3_path}' \
            f' --config={config_path}' \
            f' --disableAttributeCoding={1 if geo_only else 0}' \
            f' --uncompressedDataPath={file_path}' \
            f' --compressedStreamPath={osp.join(sub_output_dir, f"{rate_flag}.bin")}'
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
            f' --compressedStreamPath={osp.join(sub_output_dir, f"{rate_flag}.bin")}' \
            f' --reconstructedDataPath={osp.join(sub_output_dir, f"{rate_flag}_recon.ply")}'
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
                file_path,
                osp.join(sub_output_dir, f'{rate_flag}_recon.ply'), resolution,
                color=False if geo_only else True,
                normal_file=f'{osp.splitext(file_path)[0]}_n.ply',
                command=pc_error_path,
                hooks=(hook_for_org_points_num,)
            ), False
        )
    sub_metric_dict['bpp'] = [bits / org_points_num for bits, org_points_num in zip(
        sub_metric_dict['bits'], sub_metric_dict['org points num'])]
    del sub_metric_dict['bits'], sub_metric_dict['org points num']
    return sub_metric_dict


if __name__ == '__main__':
    test_geo_intra(32, True)
