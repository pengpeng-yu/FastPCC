import subprocess
from typing import Dict
import os
import os.path as osp

import numpy as np
import open3d as o3d

try:
    from lib.data_utils import write_ply_file
except ImportError: write_ply_file = None
from lib.metrics.pcqm_wrapper import pcqm
from lib.metrics.graph_sim_wrapper import graph_sim
from scripts.script_config import pc_error_path


def if_ply_has_vertex_normal(file_path: str):
    has = False
    with open(file_path, 'rb') as f:
        while True:
            try:
                line = f.readline()
                if line.strip() == b'end_header': break
                elif line.rsplit(maxsplit=1)[-1] == b'nx':
                    has = True
                    break
            except Exception as e:
                print(file_path)
                raise e
    return has


_DIVIDERS = ['1. Use infile1 (A) as reference, loop over A, use normals on B. (A->B).',
             '2. Use infile2 (B) as reference, loop over B, use normals on A. (B->A).',
             '3. Final (symmetric).',
             'Job done!']


def mpeg_pc_error(
        infile1: str, infile2: str, resolution: float, normal_file: str = '',
        hausdorff: bool = False, color: bool = False, threads: int = 2, command='',
        cal_pcqm=False, cal_graph_sim=False
) -> Dict[str, float]:
    if command == '': command = pc_error_path
    cmd_args = f'{command}' \
               f' -a {infile1}' \
               f' -b {infile2}' \
               f' --resolution={resolution - 1}' \
               f' --hausdorff={int(hausdorff)}' \
               f' --color={int(color)}' \
               f' --nbThreads={threads}'

    # Priority: arg "normal_file" -> infile1's normals -> existing *_n.ply -> generate *_n.ply using infile1
    if normal_file != '' and osp.exists(normal_file):
        cmd_args += ' -n ' + normal_file
    else:
        if if_ply_has_vertex_normal(infile1):
            pass
        else:
            normal_file = osp.splitext(infile1)[0] + '_n.ply'
            if osp.isfile(normal_file):
                pass
            else:
                pc = o3d.io.read_point_cloud(infile1)
                # https://github.com/isl-org/Open3D/wiki/Deadlock-with-multiprocessing-(using-fork)-and-OpenMP
                pc.estimate_normals()
                write_ply_file(np.asarray(pc.points), normal_file, normals=np.asarray(pc.normals))
                print(f'Warning: For computing point-to-plane loss, '
                      f'a PLY file is generated at {normal_file} with Open3D normal estimation.')
            cmd_args += ' -n ' + normal_file

    subp_stdout = subprocess.run(
        cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, check=True, text=True
    ).stdout

    metric_dict = {}
    flag_read = False
    for line in subp_stdout.splitlines():
        if line.startswith('Point cloud sizes for org version'):
            metric_dict['org points num'] = int(line.rstrip().rsplit(' ', 3)[1][:-1])
        elif line.startswith(_DIVIDERS[0]):
            flag_read = True
        elif line.startswith(_DIVIDERS[-1]):
            break
        elif flag_read and ':' in line:
            line = line.strip()
            key, value = line.split(':', 1)
            metric_dict[key.strip()] = float(value)

    if metric_dict == {}:
        raise RuntimeError(subp_stdout)
    metric_dict["mse1+mse2 (p2point)"] = metric_dict["mse1      (p2point)"] + metric_dict["mse2      (p2point)"]
    metric_dict["mse1+mse2/2(p2point)"] = metric_dict["mse1+mse2 (p2point)"] / 2
    if color:
        metric_dict["c[3],PSNRF"] = metric_dict["c[0],PSNRF"] * 0.75 + \
            metric_dict["c[1],PSNRF"] / 8 + metric_dict["c[2],PSNRF"] / 8

    if color:
        if cal_pcqm:
            metric_dict['PCQM'] = pcqm(infile1, infile2, threads)
        if cal_graph_sim:
            metric_dict['GraphSIM'] = graph_sim(infile1, infile2, threads)
    return metric_dict
