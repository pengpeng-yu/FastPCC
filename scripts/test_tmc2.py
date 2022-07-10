import os
import shutil
import subprocess
import json

from lib.metrics.pc_error_wapper import mpeg_pc_error
from scripts.log_extract_utils import *


tmc2_path = ('../mpeg-pcc-tmc2/bin/PccAppEncoder', '../mpeg-pcc-tmc2/bin/PccAppDecoder')
pc_error_path = 'pc_error_d'

file_lists = (
    'datasets/8iVFBv2/list.txt',
    'datasets/Owlii/list.txt',
)
resolutions = (1024, 2048)

config_dir = '../mpeg-pcc-tmc2/cfg'
output_dir = 'runs/tests/tmc2_geo'


class TMC2LogExtractor(LogExtractor):
    default_enc_log_mappings: log_mappings_type = {
        '  TotalMetadata': ('meta bits', lambda l: int(l.split()[-2])),
        '  TotalGeometry': ('geo bits', lambda l: int(l.strip().split()[-2])),
        'Processing time (user.self)': ('encode time', lambda l: float(l.split()[-2])),
        'Peak memory': ('encode memory', lambda l: int(l.split()[-2]))
    }

    default_dec_log_mappings: log_mappings_type = {
        'Processing time (user.self)': ('decode time', lambda l: float(l.split()[-2])),
        'Peak memory': ('decode memory', lambda l: int(l.split()[-2]))
    }

    def __init__(self):
        self.enc_log_mappings = self.default_enc_log_mappings
        self.dec_log_mappings = self.default_dec_log_mappings
        super(TMC2LogExtractor, self).__init__()

    def extract_enc_log(self, log: str):
        return self.extract_log(log, self.enc_log_mappings)

    def extract_dec_log(self, log: str):
        return self.extract_log(log, self.dec_log_mappings)


def test_geo_single_frame():
    print('Test tmc3 geo coding')

    def hook_for_org_points_num(line):
        if line.startswith('Point cloud sizes for org version, dec version, and the scaling ratio'):
            return 'org points num', int(line.rstrip().rsplit(' ', 3)[1][:-1])

    log_extractor = TMC2LogExtractor()
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    all_file_metric_dict: all_file_metric_dict_type = {}

    print(f'Output to "{output_dir}"')
    for resolution, file_list in zip(resolutions, file_lists):
        file_paths = read_file_list_with_rel_path(file_list)
        for file_path in file_paths:
            sub_metric_dict: one_file_metric_dict_type = {}
            file_basename = osp.splitext(osp.split(file_path)[1])[0]
            cfg_name = file_basename.rsplit('_', 1)[0]
            sub_output_dir = osp.join(output_dir, file_basename)
            os.makedirs(sub_output_dir, exist_ok=True)
            for rate in range(1, 6):
                print(f'    Test file {file_path}, res {resolution}, r{rate}')
                command_enc = \
                    f'{tmc2_path[0]}' \
                    f' --configurationFolder={config_dir}/' \
                    f' --config={config_dir}/common/ctc-common.cfg' \
                    f' --config={config_dir}/condition/ctc-all-intra.cfg' \
                    f' --config={config_dir}/sequence/{cfg_name}.cfg' \
                    f' --config={config_dir}/rate/ctc-r{rate}.cfg' \
                    f' --uncompressedDataPath={file_path}' \
                    f' --frameCount=1' \
                    f' --compressedStreamPath={sub_output_dir}/r{rate}.bin' \
                    f' --reconstructedDataPath={sub_output_dir}/r{rate}_enc_recon.ply' \
                    f' --noAttributes=1' \
                    f' --computeMetrics=0'
                subp_enc = subprocess.run(
                    command_enc, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    shell=True, check=True, text=True
                )
                with open(osp.join(sub_output_dir, f'log_r{rate}_enc.txt'), 'w') as f:
                    f.write(subp_enc.stdout)
                sub_metric_dict = concat_values_for_dict(
                    sub_metric_dict,
                    log_extractor.extract_enc_log(subp_enc.stdout)
                )
                command_dec = \
                    f'{tmc2_path[1]}' \
                    f' --compressedStreamPath={sub_output_dir}/r{rate}.bin' \
                    f' --reconstructedDataPath={sub_output_dir}/r{rate}_dec_recon.ply' \
                    f' --inverseColorSpaceConversionConfig={config_dir}/hdrconvert/yuv420torgb444.cfg' \
                    f' --resolution={resolution - 1}' \
                    f' --computeMetrics=0'
                subp_dec = subprocess.run(
                    command_dec, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    shell=True, check=True, text=True
                )
                with open(osp.join(sub_output_dir, f'log_r{rate}_dec.txt'), 'w') as f:
                    f.write(subp_enc.stdout)
                sub_metric_dict = concat_values_for_dict(
                    sub_metric_dict,
                    log_extractor.extract_dec_log(subp_dec.stdout)
                )
                sub_metric_dict = concat_values_for_dict(
                    sub_metric_dict,
                    mpeg_pc_error(
                        file_path,
                        osp.join(sub_output_dir, f'r{rate}_dec_recon.ply'),
                        resolution, threads=16, command=pc_error_path,
                        hooks=(hook_for_org_points_num,)
                    )
                )
            sub_metric_dict['bpp'] = [(meta_bits + geo_bits) / org_points_num for
                                      meta_bits, geo_bits, org_points_num in zip(
                    sub_metric_dict['meta bits'], sub_metric_dict['geo bits'], sub_metric_dict['org points num'])]
            del sub_metric_dict['meta bits'], sub_metric_dict['geo bits'], sub_metric_dict['org points num']
            all_file_metric_dict[file_path] = sub_metric_dict

    print('Done')
    with open(osp.join(output_dir, f'metric_dict.json'), 'w') as f:
        f.write(json.dumps(all_file_metric_dict, indent=2, sort_keys=False))


if __name__ == '__main__':
    test_geo_single_frame()
