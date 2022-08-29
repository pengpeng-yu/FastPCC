"""
This script is based on commit c3c9798a0f63970bd17ce191900ded478a8aa0f6 of mpeg-pcc-tmc13.
"""

from glob import glob
import os
import shutil
import subprocess
import json

from lib.metrics.pc_error_wapper import mpeg_pc_error
from scripts.log_extract_utils import *
from scripts.shared_config import pc_error_path, metric_dict_filename


tmc3_path = '../mpeg-pcc-tmc13/build/tmc3/tmc3'

file_lists = (
    'datasets/8iVFBv2/list.txt',
    'datasets/Owlii/list.txt',
)
resolutions = (1024, 2048)
assert len(file_lists) == len(resolutions)

config_dirs = (
    '../mpeg-pcc-tmc13/cfg/octree-predlift/lossy-geom-lossy-attrs',
    '../mpeg-pcc-tmc13/cfg/trisoup-predlift/lossy-geom-lossy-attrs',
)
output_dirs = (
    'runs/tests/tmc3_geo/octree',
    'runs/tests/tmc3_geo/trisoup',
)
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


def test_geo_single_frame():
    print('Test tmc3 geo coding')

    log_extractor = TMC3LogExtractor()
    for config_dir, output_dir in zip(config_dirs, output_dirs):
        print(f'Test config: "{config_dir}"')
        print(f'Output to "{output_dir}"')
        if osp.exists(output_dir):
            shutil.rmtree(output_dir)
        all_file_metric_dict: all_file_metric_dict_type = {}

        for resolution, file_list in zip(resolutions, file_lists):
            file_paths = read_file_list_with_rel_path(file_list)
            for file_path in file_paths:
                sub_metric_dict: one_file_metric_dict_type = {}
                file_basename = osp.splitext(osp.split(file_path)[1])[0]
                config_paths = glob(osp.join(config_dir, file_basename, '*', 'encoder.cfg'))
                sub_output_dir = osp.join(output_dir, file_basename)
                os.makedirs(sub_output_dir, exist_ok=True)
                for config_path in config_paths:
                    rate_flag = osp.split(osp.split(config_path)[0])[1]
                    print(f'    Test file {file_path}, res {resolution}, {rate_flag}')
                    command_enc = \
                        f'{tmc3_path}' \
                        f' --config={config_path}' \
                        f' --disableAttributeCoding=1' \
                        f' --uncompressedDataPath={file_path}' \
                        f' --compressedStreamPath={osp.join(sub_output_dir, f"{rate_flag}.bin")}'
                    subp_enc = subprocess.run(
                        command_enc, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        shell=True, check=True, text=True
                    )
                    with open(osp.join(sub_output_dir, f'log_{rate_flag}_enc.txt'), 'w') as f:
                        f.write(subp_enc.stdout)
                    sub_metric_dict = concat_values_for_dict(
                        sub_metric_dict,
                        log_extractor.extract_enc_log(subp_enc.stdout)
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
                        log_extractor.extract_dec_log(subp_dec.stdout)
                    )
                    sub_metric_dict = concat_values_for_dict(
                        sub_metric_dict,
                        mpeg_pc_error(
                            file_path,
                            osp.join(sub_output_dir, f'{rate_flag}_recon.ply'),
                            resolution, threads=16, command=pc_error_path,
                            hooks=(hook_for_org_points_num,)
                        )
                    )
                sub_metric_dict['bpp'] = [bits / org_points_num for bits, org_points_num in zip(
                    sub_metric_dict['bits'], sub_metric_dict['org points num'])]
                del sub_metric_dict['bits'], sub_metric_dict['org points num']
                all_file_metric_dict[file_path] = sub_metric_dict

        print(f'{config_dir} Done')
        with open(osp.join(output_dir, metric_dict_filename), 'w') as f:
            f.write(json.dumps(all_file_metric_dict, indent=2, sort_keys=False))
    print('All Done')


if __name__ == '__main__':
    test_geo_single_frame()