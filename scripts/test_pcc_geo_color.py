"""
This script is based on the commit 51d398dfccd62ac35828842effd35bc91c97abe0 of pcc-geo-color.
"""
import sys
import os
import os.path as osp
from glob import glob
import shutil
import subprocess
import multiprocessing as mp
import json


sys.path.append(osp.dirname(osp.dirname(__file__)))
from scripts.script_config import metric_dict_filename, cuda_device, test_dir, pc_error_path
from scripts.log_extract_utils import concat_values_for_dict
from lib.metrics.pc_error_wrapper import mpeg_pc_error


pcc_geo_color_src_path = '../pcc-geo-color/src'
pretrained_models_path = '../pcc-geo-color/src/pre-trained_models'
file_dirs = (
    'datasets/Owlii/basketball_player_vox11', 'datasets/Owlii/dancer_vox11',
    'datasets/8iVFBv2/loot/Ply', 'datasets/8iVFBv2/redandblack/Ply'
)
filename_pattern = '*[0-9].ply'  # excludes *_n.ply
resolutions = (2048, 2048, 1024, 1024)
output_dir = f'{test_dir}/pcc-geo-color'
output_dir = osp.abspath(output_dir)
processes_num = mp.cpu_count() * 2 // 3


def test():
    print('Test pcc-geo-color')
    os.makedirs(output_dir, exist_ok=True)
    pool = mp.get_context('forkserver').Pool(processes_num)

    pretrained_paths = sorted(glob(osp.join(pretrained_models_path, 'geom_color', '*')),
                              key=lambda _: int(_.rsplit('_', 1)[1][3:]))
    for pretrained_path in pretrained_paths:
        with open(osp.join(pretrained_path, 'checkpoint'), 'w') as f:
            f.write('model_checkpoint_path: "model.ckpt-100000"\n'
                    'all_model_checkpoint_paths: "model.ckpt-100000"\n')

    all_file_metric_dict = {}
    all_pc_error_res = {}
    pre_command = f'export TF_FORCE_GPU_ALLOW_GROWTH=true;' \
                  f'export CUDA_VISIBLE_DEVICES={cuda_device};cd {pcc_geo_color_src_path};'
    for resolution, file_dir in zip(resolutions, file_dirs):
        sub_output_dir = osp.abspath(osp.join(output_dir, file_dir))
        subprocess.run(
            f'{pre_command}'
            f'{sys.executable} partition.py'
            f' {osp.abspath(file_dir)} {osp.join(sub_output_dir, "par")} --block_size=128 --keep_size=0',
            shell=True, check=True, executable=shutil.which('bash')
        )
        for pretrained_path in pretrained_paths:
            pretrained_name = osp.split(pretrained_path)[1]
            sub_sub_output_dir = osp.join(sub_output_dir, pretrained_name)
            pretrained_path = osp.abspath(pretrained_path)
            subprocess.run(
                f'{pre_command}'
                f'{sys.executable} compress.py'
                f' {osp.join(sub_output_dir, "par")} "*.ply"'
                f' {osp.join(sub_sub_output_dir, "cmp")}'
                f' {pretrained_path} --resolution=128 --task=geometry+color',
                shell=True, check=True, executable=shutil.which('bash')
            )
            subprocess.run(
                f'{pre_command}'
                f'{sys.executable} decompress.py'
                f' {osp.join(sub_output_dir, "par")} "*.ply"'
                f' {osp.join(sub_sub_output_dir, "cmp")} "*.ply.bin"'
                f' {osp.join(sub_sub_output_dir, "recon")}'
                f' {pretrained_path} --resolution=128 --task=geometry+color',
                shell=True, check=True, executable=shutil.which('bash')
            )
            subprocess.run(
                f'{pre_command}'
                f'{sys.executable} merge.py'
                f' {osp.abspath(file_dir)}'
                f' {osp.join(sub_sub_output_dir, "recon")}'
                f' {osp.join(sub_sub_output_dir, "merge")} --resolution=128 --task=geometry+color',
                shell=True, check=True, executable=shutil.which('bash')
            )
            for org_file_path, recon_file_path in \
                    zip(sorted(glob(osp.join(file_dir, '*.ply'))),
                        sorted(glob(osp.join(sub_sub_output_dir, "merge", '*.ply')))):
                cmp_size = 0
                for cmp_file in glob(osp.join(sub_sub_output_dir, "cmp", osp.split(org_file_path)[1][:-4] + '*')):
                    cmp_size += osp.getsize(cmp_file)
                if org_file_path not in all_pc_error_res: all_pc_error_res[org_file_path] = []
                all_pc_error_res[org_file_path].append((pool.apply_async(
                    mpeg_pc_error,
                    (org_file_path, recon_file_path, resolution, '',
                     False, True, 1, pc_error_path)), cmp_size))
        shutil.rmtree(osp.join(sub_output_dir, "par"))
    pool.close()

    for k, v in all_pc_error_res.items():
        metric_dict = v[0][0].get()
        metric_dict['bpp'] = v[0][1] / metric_dict['org points num']
        for v_ in v[1:]:
            metric_dict_ = v_[0].get()
            metric_dict_['bpp'] = v_[1] / metric_dict_['org points num']
            metric_dict = concat_values_for_dict(metric_dict, metric_dict_)
        all_file_metric_dict[k] = metric_dict

    with open(osp.join(output_dir, metric_dict_filename), 'w') as f:
        f.write(json.dumps(all_file_metric_dict, indent=2, sort_keys=False))
    print('All Done')


if __name__ == '__main__':
    test()
