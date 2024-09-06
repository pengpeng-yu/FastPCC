"""
This script is based on https://github.com/pengpeng-yu/pcc-geo-color
Original repo: https://github.com/mmspg/pcc-geo-color
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
file_lists = (
    'datasets/Owlii/list_basketball_player_dancer.txt',
    'datasets/8iVFBv2/list_loot_redandblack.txt',
)
resolutions = (2048, 1024,)
output_dir = f'{test_dir}/pcc-geo-color'
output_dir = osp.abspath(output_dir)
processes_num = mp.cpu_count() // 2  # only for mpeg_pc_error


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
    for resolution, file_list in zip(resolutions, file_lists):
        sub_output_dir = osp.abspath(osp.join(output_dir, osp.dirname(file_list)))
        subprocess.run(
            f'{pre_command}'
            f'{sys.executable} partition.py'
            f' {osp.abspath(osp.dirname(file_list))}'
            f' --input_list {osp.abspath(file_list)}'
            f' {osp.join(sub_output_dir, "par")} --block_size=128 --keep_size=0',
            shell=True, check=True, executable=shutil.which('bash')
        )
        for pretrained_path in pretrained_paths:
            pretrained_name = osp.split(pretrained_path)[1]
            sub_sub_output_dir = osp.join(sub_output_dir, pretrained_name)
            pretrained_path = osp.abspath(pretrained_path)
            subprocess.run(
                f'{pre_command}'
                f'{sys.executable} compress.py'
                f' {osp.join(sub_output_dir, "par")} "**/*.ply"'
                f' {osp.join(sub_sub_output_dir, "cmp")}'
                f' {pretrained_path} --resolution=128 --task=geometry+color',
                shell=True, check=True, executable=shutil.which('bash')
            )
            subprocess.run(
                f'{pre_command}'
                f'{sys.executable} decompress.py'
                f' {osp.join(sub_output_dir, "par")} "**/*.ply"'
                f' {osp.join(sub_sub_output_dir, "cmp")} "**/*.ply.bin"'
                f' {osp.join(sub_sub_output_dir, "recon")}'
                f' {pretrained_path} --resolution=128 --task=geometry+color',
                shell=True, check=True, executable=shutil.which('bash')
            )
            subprocess.run(
                f'{pre_command}'
                f'{sys.executable} merge.py'
                f' {osp.abspath(osp.dirname(file_list))}'
                f' {osp.join(sub_sub_output_dir, "recon")}'
                f' {osp.join(sub_sub_output_dir, "merge")}'
                f' --input_list {osp.abspath(file_list)}'
                f' --resolution=128 --task=geometry+color',
                shell=True, check=True, executable=shutil.which('bash')
            )
            with open(file_list) as f:
                f_lines = [_.strip() for _ in f.readlines()]
            org_file_paths = [osp.join(osp.dirname(file_list), _) for _ in f_lines]
            recon_file_paths = [osp.join(sub_sub_output_dir, "merge", osp.splitext(_)[0] + '_dec.ply') for _ in f_lines]
            cmp_file_paths = [osp.join(sub_sub_output_dir, "cmp", osp.splitext(_)[0] + '*') for _ in f_lines]
            for org_file_path, recon_file_path, cmp_file_path in zip(org_file_paths, recon_file_paths, cmp_file_paths):
                cmp_size = 0
                for cmp_file in glob(cmp_file_path):
                    cmp_size += osp.getsize(cmp_file)
                if org_file_path not in all_pc_error_res: all_pc_error_res[org_file_path] = []
                all_pc_error_res[org_file_path].append((pool.apply_async(
                    mpeg_pc_error,
                    (org_file_path, recon_file_path, resolution, '',
                     False, True)), cmp_size))
        shutil.rmtree(osp.join(sub_output_dir, "par"))
        shutil.rmtree(osp.join(sub_output_dir, "recon"))
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
