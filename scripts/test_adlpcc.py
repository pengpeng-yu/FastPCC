"""
This script is based on the commit cac62382472adbbf23fb11ec70c43636c7622e48 of ADLPCC.
"""
import os
import os.path as osp
import shutil
import sys
from glob import glob
import subprocess
import json

sys.path.append(osp.dirname(osp.dirname(__file__)))
from lib.metrics.pc_error_wrapper import mpeg_pc_error
from scripts.log_extract_utils import *
from scripts.script_config import pc_error_path, metric_dict_filename, cuda_device, test_dir


adl_pcc_dir = '../ADLPCC'

file_lists = (
    "datasets/MPEG_GPCC_CTC/Solid/Solid_4096.txt",
    "datasets/MPEG_GPCC_CTC/Solid/Solid_2048.txt",
    "datasets/MPEG_GPCC_CTC/Solid/Solid_1024.txt",
    "datasets/MVUB/list.txt",
)

resolutions = (4096, 2048, 1024, 512,)
assert len(file_lists) == len(resolutions)

output_dir = f'{test_dir}/ADLPCC'
# Only the metric file is here. The compressed and reconstructed points clouds are in ../ADLPCC/results.


class ADLPCCLogExtractor(LogExtractor):
    default_enc_log_mappings: log_mappings_type = {
        'compress time': ('encode time', lambda l: float(l.split()[-1])),
        'compress tf.contrib.memory_stats.MaxBytesInUse()':
            ('encode memory', lambda l: float(l.split()[-1]) / 1024),  # B -> KB
    }
    default_dec_log_mappings: log_mappings_type = {
        'decompress time': ('decode time', lambda l: float(l.split()[-1])),
        'decompress tf.contrib.memory_stats.MaxBytesInUse()':
            ('decode memory', lambda l: float(l.split()[-1]) / 1024),
    }

    def __init__(self):
        self.enc_log_mappings = self.default_enc_log_mappings
        self.dec_log_mappings = self.default_dec_log_mappings
        super(ADLPCCLogExtractor, self).__init__()

    def extract_enc_log(self, log: str):
        return self.extract_log(log, self.enc_log_mappings)

    def extract_dec_log(self, log: str):
        return self.extract_log(log, self.dec_log_mappings)


def test():
    print('Test ADLPCC')

    log_extractor = ADLPCCLogExtractor()
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    python_pre_command = f'export TF_FORCE_GPU_ALLOW_GROWTH=true;export CUDA_VISIBLE_DEVICES={cuda_device};'
    all_file_metric_dict: all_file_metric_dict_type = {}

    for resolution, file_list in zip(resolutions, file_lists):
        file_paths = read_file_list_with_rel_path(file_list)
        for file_path in file_paths:
            sub_metric_dict: one_file_metric_dict_type = {'bpp': []}
            file_basename = osp.splitext(osp.split(file_path)[1])[0]
            for weight_dir_path in glob(osp.join(adl_pcc_dir, 'models', "*/")):
                rate_flag = osp.split(weight_dir_path[:-1])[1]
                print(f'\nTest file {file_path}, res {resolution}, rate flag {rate_flag}\n')
                command_enc = \
                    f'cd {osp.join(adl_pcc_dir, "src")};' \
                    f'{python_pre_command}' \
                    f'{sys.executable} ADLPCC.py compress' \
                    f' "{osp.abspath(file_path)}"' \
                    f' "../models/{rate_flag}/*"'
                subp_enc = subprocess.run(
                    command_enc, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    shell=True, check=True, text=True, executable=shutil.which('bash')
                )
                print(subp_enc.stdout)
                # sub_metric_dict = concat_values_for_dict(
                #     sub_metric_dict,
                #     log_extractor.extract_enc_log(subp_enc.stdout), False
                # )
                encoded_path = osp.join(
                    adl_pcc_dir, 'results', rate_flag, file_basename,
                    file_basename + '.pkl.gz'
                )
                command_dec = \
                    f'cd {osp.join(adl_pcc_dir, "src")};' \
                    f'{python_pre_command}' \
                    f'python ADLPCC.py decompress' \
                    f" ../results/{rate_flag}/{file_basename}/{file_basename}.pkl.gz" \
                    f' "../models/{rate_flag}/*"'
                subp_dec = subprocess.run(
                    command_dec, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    shell=True, check=True, text=True, executable=shutil.which('bash')
                )
                print(subp_dec.stdout)
                # sub_metric_dict = concat_values_for_dict(
                #     sub_metric_dict,
                #     log_extractor.extract_dec_log(subp_dec.stdout), False
                # )
                recon_path = osp.join(
                    adl_pcc_dir, 'results', rate_flag, file_basename,
                    file_basename + '.pkl.gz.dec.ply'
                )
                sub_metric_dict = concat_values_for_dict(
                    sub_metric_dict,
                    mpeg_pc_error(
                        file_path, osp.abspath(recon_path), resolution, command=pc_error_path
                    ), False
                )
                sub_metric_dict['bpp'].append(
                    osp.getsize(encoded_path) * 8 / sub_metric_dict['org points num'][0]
                )
            del sub_metric_dict['org points num']
            all_file_metric_dict[file_path] = sub_metric_dict

    print('Done')
    with open(osp.join(output_dir, metric_dict_filename), 'w') as f:
        f.write(json.dumps(all_file_metric_dict, indent=2, sort_keys=False))


if __name__ == '__main__':
    test()
