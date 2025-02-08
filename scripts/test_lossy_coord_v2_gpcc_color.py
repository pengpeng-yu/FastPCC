"""
This script is based on the version 26rc2 of mpeg-pcc-tmc13.
"""
from glob import glob
import sys
import os
import os.path as osp
import shutil
import subprocess
import json
import multiprocessing as mp

import numpy as np
import open3d as o3d
import torch

sys.path.append(osp.dirname(osp.dirname(__file__)))
from lib.metrics.pc_error_wrapper import mpeg_pc_error
from lib.data_utils import write_ply_file
from scripts.script_config import pc_error_path, metric_dict_filename, test_dir, tmc3_path, cuda_device
from models.convolutional.lossy_coord_lossy_color.layers import sample_wise_recolor


cal_pcqm = False
cal_graph_sim = False
weight_prefix = 'weights/convolutional'
config_prefix = 'config/convolutional'
output_prefix = f'{test_dir}/convolutional'
config_paths = 'lossy_coord_v2/baseline_r*.yaml'
file_lists = (
    'datasets/Owlii/list_basketball_player_dancer.txt',
    'datasets/8iVFBv2/list_loot_redandblack.txt',
)
resolutions = (2048, 1024,)
tmc3_config_dirs = (
    '../mpeg-pcc-tmc13/cfg/octree-predlift/lossless-geom-lossy-attrs',
    '../mpeg-pcc-tmc13/cfg/octree-raht/lossless-geom-lossy-attrs',
)
processes_num = mp.cpu_count() // 2
assert len(file_lists) == len(resolutions)


def extract_color_bytes_log(log: str):
    lines = log.splitlines()
    bits = 0
    for idx, line in enumerate(lines):
        if line.startswith('colors bitstream size'):
            bits += float(line.split()[3])
    return bits


def extract_color_time_log(log: str):
    lines = log.splitlines()
    t = 0
    for idx, line in enumerate(lines):
        if line.startswith('colors processing time (user):'):
            t += float(line.split()[4])
    return t


def test_lossy_coord_v2():
    print(f'Test lossy coord v2 coding')

    for config_path in sorted(glob(osp.join(config_prefix, config_paths))):
        config_name = config_path[len(config_prefix) + 1: -5]
        sub_run_dir = osp.join(output_prefix, config_name).replace('lossy_coord_v2', 'lossy_coord_v2_recolor', 1)
        weight_path = osp.join(weight_prefix, config_name + '.pt')
        print(f'\nTest config: "{config_path}", weight "{weight_path}"\n')
        if osp.exists(sub_run_dir):
            shutil.rmtree(sub_run_dir)
        command = f'{sys.executable} test.py {config_path}' \
            f' test.from_ckpt={weight_path}' \
            f' test.rundir_name={sub_run_dir.replace("runs/", "", 1)}' \
            f' test.device={cuda_device}' \
            f' test.dataset.root=\"[{",".join([osp.dirname(_) for _ in file_lists])}]\"' \
            f' test.dataset.filelist_path=\"[{",".join([osp.split(_)[1] for _ in file_lists])}]\"' \
            f' test.dataset.resolution=\"[{",".join((str(_) for _ in resolutions))}]\"' \
            f' test.dataset.kd_tree_partition_max_points_num=0' \
            f' test.dataset.coord_scaler=1.0'
        print(command)
        subprocess.run(command, shell=True, check=True, executable=shutil.which('bash'))

        with open(osp.join(sub_run_dir, metric_dict_filename), 'rb') as f:
            sub_metric_dict = json.load(f)
        for org_file_path, file_metrics in sub_metric_dict.items():
            recon_file = osp.join(sub_run_dir, 'results', osp.splitext(org_file_path)[0] + '_recon.ply')
            recon_recolor_file = osp.join(sub_run_dir, 'results', osp.splitext(org_file_path)[0] + '_recolor.ply')
            recon_xyz = np.asarray(o3d.io.read_point_cloud(recon_file).points).astype(np.float32)
            org_pc = o3d.io.read_point_cloud(org_file_path)
            new_color = sample_wise_recolor(
                torch.from_numpy(recon_xyz).to(f'cuda:{cuda_device}'),
                torch.from_numpy(np.asarray(org_pc.points).astype(np.float32)).to(f'cuda:{cuda_device}'),
                torch.from_numpy(np.asarray(org_pc.colors)).to(f'cuda:{cuda_device}').to(torch.float32) * 255,
            )
            write_ply_file(recon_xyz, recon_recolor_file, rgb=new_color)


def test_tmc3_color():
    print(f'Test tmc3 color coding')
    pool = mp.get_context('forkserver').Pool(processes_num)
    for tmc3_config_dir in tmc3_config_dirs:
        print(f'Test config: "{tmc3_config_dir}"')

        for tmc3_rate_flag, config_path in zip(  # check if the tmc3 flags correspond to the lossy_coord_v2 paths
                ['r06', 'r05', 'r04', 'r03', 'r02', 'r01'], sorted(glob(osp.join(config_prefix, config_paths)))):
            config_name = config_path[len(config_prefix) + 1: -5]
            sub_run_dir = osp.join(output_prefix, config_name).replace('lossy_coord_v2', 'lossy_coord_v2_recolor', 1)

            all_file_run_res = []
            with open(osp.join(sub_run_dir, metric_dict_filename), 'rb') as f:
                sub_metric_dict = json.load(f)
            for org_file_path, file_metrics in sub_metric_dict.items():
                recon_recolor_file = osp.join(sub_run_dir, 'results', osp.splitext(org_file_path)[0] + '_recolor.ply')
                for idx, file_list in enumerate(file_lists):
                    file_list_basename = osp.dirname(file_list)
                    if file_list_basename in org_file_path:
                        break
                resolution = resolutions[idx]
                all_file_run_res.append(pool.apply_async(
                    run_single_file,
                    (recon_recolor_file, org_file_path, resolution, tmc3_config_dir, tmc3_rate_flag,
                     sub_metric_dict[org_file_path], osp.dirname(recon_recolor_file))
                ))
            all_file_metric_dict = {}
            for run_res in all_file_run_res:
                ret = run_res.get()
                if ret is not None:
                    all_file_metric_dict[ret[0]] = ret[1]

            tmc3_metric_dict_path = osp.join(
                output_prefix, config_name.replace(
                    'lossy_coord_v2', 'lossy_coord_v2_raht' if 'raht' in tmc3_config_dir
                    else 'lossy_coord_v2_predlift', 1),
                metric_dict_filename)
            os.makedirs(osp.dirname(tmc3_metric_dict_path), exist_ok=True)
            with open(tmc3_metric_dict_path, 'w') as f:
                f.write(json.dumps(all_file_metric_dict, indent=2, sort_keys=False))

        print(f'{tmc3_config_dir} Done')
    pool.close()
    print('All Done')


def run_single_file(file_path, org_file_path, resolution, config_dir, rate_flag, org_metrics, output_dir):
    file_basename = osp.splitext(osp.split(org_file_path)[1])[0]
    config_path = osp.join(config_dir, file_basename.lower(), rate_flag, 'encoder.cfg')
    if not osp.isfile(config_path):
        config_basename = file_basename.lower().rsplit('_', 1)[0]
        config_path = osp.join(config_dir, config_basename.lower(), rate_flag, 'encoder.cfg')
        if not osp.isfile(config_path):
            config_paths = glob(osp.join(config_dir, config_basename + '*', rate_flag, 'encoder.cfg'))
            if len(config_paths) == 0:
                raise RuntimeError
            else:
                assert len(config_paths) == 1
                config_path = config_paths[0]
    print(f'    Test file {file_path}, res {resolution}, {rate_flag}, {config_path}')
    command_enc = \
        f'{tmc3_path}' \
        f' --config={config_path}' \
        f' --uncompressedDataPath={file_path}' \
        f' --compressedStreamPath={osp.join(output_dir, f"{file_basename}_{rate_flag}.bin")}'
    subp_enc = subprocess.run(
        command_enc, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, check=True, text=True
    )
    with open(osp.join(output_dir, f'log_{file_basename}_{rate_flag}_enc.txt'), 'w') as f:
        f.write(subp_enc.stdout)
    command_dec = \
        f'{tmc3_path}' \
        f' --mode=1' \
        f' --outputBinaryPly=1' \
        f' --compressedStreamPath={osp.join(output_dir, f"{file_basename}_{rate_flag}.bin")}' \
        f' --reconstructedDataPath={osp.join(output_dir, f"{file_basename}_tmc3_recon.ply")}'
    subp_dec = subprocess.run(
        command_dec, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, check=True, text=True
    )
    metric_dict = mpeg_pc_error(
        org_file_path,
        osp.join(output_dir, f'{file_basename}_tmc3_recon.ply'), resolution,
        color=True, command=pc_error_path, cal_pcqm=cal_pcqm, cal_graph_sim=cal_graph_sim
    )
    metric_dict['compressed_bytes'] = org_metrics['compressed_bytes'] + extract_color_bytes_log(subp_enc.stdout)
    metric_dict['encode time'] = org_metrics['encode time'] + extract_color_time_log(subp_enc.stdout)
    metric_dict['decode time'] = org_metrics['decode time'] + extract_color_time_log(subp_dec.stdout)
    metric_dict['encode memory'] = org_metrics['encode memory']
    metric_dict['decode memory'] = org_metrics['decode memory']
    metric_dict['bpp'] = metric_dict['compressed_bytes'] * 8 / metric_dict['org points num']
    return org_file_path, metric_dict


if __name__ == '__main__':
    test_lossy_coord_v2()
    test_tmc3_color()
