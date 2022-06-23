import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Union, Tuple
import pickle

import numpy as np
import matplotlib.pyplot as plt

from lib.metrics.bjontegaard import bdrate


INFO_DICT_TYPE = Dict[str, List[Union[int, float]]]
INFO_DICTS_TYPE = Dict[str, Dict[str, List[Union[int, float]]]]

sample_names = [
        "longdress_vox10_1051", "loot_vox10_1000", "redandblack_vox10_1450", "soldier_vox10_0536",
        "basketball_player_vox11_00000001", "dancer_vox11_00000001"
    ]


def float_num_in_line_end(line):
    return float(line.split()[-1])


vpcc_enc_log_info_mappings = {
    '  TotalMetadata': ('meta_bits', lambda l: int(l.split()[-2])),
    '  TotalGeometry': ('geo_bits', lambda l: int(l.strip().split()[-2])),
    'Point cloud sizes for org version': ('org_points_num', lambda l: int(l.split()[-3][:-1])),
    '   mse1      (p2point)': ('mse1      (p2point)', float_num_in_line_end),
    '   mse1,PSNR (p2point)': ('mse1,PSNR (p2point)', float_num_in_line_end),
    '   mse2      (p2point)': ('mse2      (p2point)', float_num_in_line_end),
    '   mse2,PSNR (p2point)': ('mse2,PSNR (p2point)', float_num_in_line_end),
    '   mseF      (p2point)': ('mseF      (p2point)', float_num_in_line_end),
    '   mseF,PSNR (p2point)': ('mseF,PSNR (p2point)', float_num_in_line_end),
    'Processing time (user.self)': ('Encoder Processing time (user.self)', lambda l: float(l.split()[-2])),
    'Peak memory': ('Encoder Peak memory B', lambda l: int(l.split()[-2]) * 1024)
}


vpcc_dec_log_info_mappings = {
    'Processing time (user.self)': ('Decoder Processing time (user.self)', lambda l: float(l.split()[-2])),
    'Peak memory': ('Decoder Peak memory B', lambda l: int(l.split()[-2]) * 1024)
}


def read_vpcc_log(log_info_mappings, path_pattern, idx_range=range(1, 6, 1)) -> INFO_DICT_TYPE:
    info_dict = defaultdict(list)
    for idx in idx_range:
        path = path_pattern.format(idx)
        with open(path) as f:
            for key, (target_key, map_fn) in log_info_mappings.items():
                for line in f:
                    if line.startswith(key):
                        info_dict[target_key].append(map_fn(line))
                        break
    return info_dict


def read_all_vpcc_logs() -> INFO_DICTS_TYPE:
    log_root_path = '../../code/mpeg-pcc-tmc2'
    enc_path_pattern = os.path.join(log_root_path, 'test_enc/{sample_name}/log_r{rank}.txt')
    dec_path_pattern = os.path.join(log_root_path, 'test_dec/{sample_name}/log_r{rank}.txt')
    info_concat = {}
    for sample_name in sample_names:
        enc_info_dict = read_vpcc_log(
            vpcc_enc_log_info_mappings,
            enc_path_pattern.format(sample_name=sample_name, rank='{}')
        )
        dec_info_dict = read_vpcc_log(
            vpcc_dec_log_info_mappings,
            dec_path_pattern.format(sample_name=sample_name, rank='{}')
        )
        info_concat[sample_name] = {}
        info_concat[sample_name]['bpp'] = [
            (i + j) / k for i, j, k
            in zip(
                enc_info_dict['meta_bits'],
                enc_info_dict['geo_bits'],
                enc_info_dict['org_points_num']
            )
        ]
        info_concat[sample_name].update(enc_info_dict)
        info_concat[sample_name].update(dec_info_dict)
    return info_concat


default_json_log_key_mappings = {
    'bpp': None,
    "mse1      (p2point)": None,
    "mse1,PSNR (p2point)": None,
    "mse2      (p2point)": None,
    "mse2,PSNR (p2point)": None,
    'mseF      (p2point)': None,
    'mseF,PSNR (p2point)': None,
    "encoder_elapsed_time": None,
    "encoder_max_cuda_memory_allocated": None,
    "decoder_elapsed_time": None,
    "decoder_max_cuda_memory_allocated": None
}


def read_json_results(info_paths, info_keys=None) -> INFO_DICTS_TYPE:
    if info_keys is None:
        info_keys = default_json_log_key_mappings
    info_concat = {}

    for file_path in info_paths:
        with open(file_path, 'r') as f:
            info = json.load(f)
        assert isinstance(info, dict)
        info = {os.path.splitext(os.path.split(k)[1])[0]: v for k, v in info.items()}
        assert all([k in info for k in sample_names])

        for sample_name in sample_names:
            sample_dict = info_concat.setdefault(sample_name, {})
            for info_ori_key, info_target_key in info_keys.items():
                sample_dict.setdefault(
                    info_target_key if info_target_key is not None else info_ori_key, []
                ).append(
                    info[sample_name][info_ori_key]
                )

    for value in info_concat.values():
        for v in value.values():
            assert isinstance(v, list)
            assert len(v) == len(info_paths)

    return info_concat


def compute_bd_rate(info_dicts_a, info_dicts_b) -> Dict[str, float]:
    def sort_key_func(bpp_psnr): return bpp_psnr[0]
    bd_rate_results = {}
    for key in info_dicts_a:
        if key in info_dicts_b:
            bpp_psnr_a = list(zip(info_dicts_a[key]['bpp'], info_dicts_a[key]['mseF,PSNR (p2point)']))
            bpp_psnr_b = list(zip(info_dicts_b[key]['bpp'], info_dicts_b[key]['mseF,PSNR (p2point)']))
            bd_rate_results[key] = bdrate(
                sorted(bpp_psnr_a, key=sort_key_func),
                sorted(bpp_psnr_b, key=sort_key_func)
            )
    return bd_rate_results


def compare_with_vpcc(json_path_pattern) -> Tuple[INFO_DICTS_TYPE, INFO_DICTS_TYPE, Dict[str, float]]:
    vpcc_info_dicts = read_all_vpcc_logs()

    json_info_paths = glob.glob(json_path_pattern)
    json_info_dicts = read_json_results(json_info_paths)

    bd_rate_results = compute_bd_rate(
        vpcc_info_dicts, json_info_dicts
    )
    print(json.dumps(bd_rate_results, indent=2, sort_keys=False))
    return vpcc_info_dicts, json_info_dicts, bd_rate_results


def compare_with_jsons(
        json_path_pattern_a, json_path_pattern_b
) -> Tuple[INFO_DICTS_TYPE, INFO_DICTS_TYPE, Dict[str, float]]:
    json_info_paths_a = glob.glob(json_path_pattern_a)
    json_info_dicts_a = read_json_results(json_info_paths_a)

    json_info_paths_b = glob.glob(json_path_pattern_b)
    json_info_dicts_b = read_json_results(json_info_paths_b)

    bd_rate_results = compute_bd_rate(
        json_info_dicts_a, json_info_dicts_b
    )
    print(json.dumps(bd_rate_results, indent=2, sort_keys=False))
    return json_info_dicts_a, json_info_dicts_b, bd_rate_results


def main_output_script(info_paths_dict, vs_list, output_dir):  # TODO: TYPING
    def mean(x: List): return sum(x) / len(x)

    def add_new_key_to_dict(d, k, v):
        if k in d:
            assert d[k] == v
        else:
            d[k] = v

    res_info_dict = {}
    vs_res_dict = {}
    for info_name_a, info_name_b in vs_list:
        if info_name_a == 'vpcc':
            info_dict_a, info_dict_b, a_vs_b = compare_with_vpcc(
                info_paths_dict[info_name_b]
            )
        else:
            info_dict_a, info_dict_b, a_vs_b = compare_with_jsons(
                info_paths_dict[info_name_a],
                info_paths_dict[info_name_b]
            )
        add_new_key_to_dict(res_info_dict, info_name_a, info_dict_a)
        add_new_key_to_dict(res_info_dict, info_name_b, info_dict_b)
        a_vs_b_name = f'{info_name_a}_vs_{info_name_b}'
        vs_res_dict[a_vs_b_name] = a_vs_b

    os.makedirs(output_dir, exist_ok=True)
    for key, value in res_info_dict.items():
        with open(os.path.join(output_dir, key + '.json'), 'w') as f:
            f.write(json.dumps(value, indent=2, sort_keys=False))

    sample_wise_complexity_dict = {}
    for key, value in res_info_dict.items():
        for k, v in value.items():
            if k not in sample_wise_complexity_dict:
                sample_wise_complexity_dict[k] = []
            if key != 'vpcc':
                sample_wise_complexity_dict[k].extend([
                    mean(v["encoder_elapsed_time"]),
                    mean(v["encoder_max_cuda_memory_allocated"]) / 1024 ** 2,
                    mean(v["decoder_elapsed_time"]),
                    mean(v["decoder_max_cuda_memory_allocated"]) / 1024 ** 2
                ])
            else:
                sample_wise_complexity_dict[k].extend([
                    mean(v["Encoder Processing time (user.self)"]),
                    mean(v["Encoder Peak memory B"]) / 1024 ** 2,
                    mean(v["Decoder Processing time (user.self)"]),
                    mean(v["Decoder Peak memory B"]) / 1024 ** 2
                ])
    f = open(os.path.join(output_dir, 'complexity.csv'), 'w')
    f.write(' ,')
    for key, value in res_info_dict.items():
        for k in ('enc time', 'enc peak mem', 'dec time', 'dec peak mem'):
            f.write(key + ' ' + k + ',')
    f.write('\n')
    for key, value in sample_wise_complexity_dict.items():
        f.write(key + ',')
        f.write(','.join(map(str, value)) + ',\n')
    f.close()

    sample_wise_vs_dict = {}
    for key, value in vs_res_dict.items():
        for k, v in value.items():
            if k in sample_wise_vs_dict:
                sample_wise_vs_dict[k].append(v)
            else:
                sample_wise_vs_dict[k] = [v]
    f = open(os.path.join(output_dir, 'vs.csv'), 'w')
    f.write(' ,')
    for key in vs_res_dict:
        f.write(key + ',')
    f.write('\n')
    for key, value in sample_wise_vs_dict.items():
        f.write(key + ',')
        f.write(','.join(map(str, value)) + ',\n')
    f.close()

    with open(os.path.join(output_dir, 'pickle'), 'wb') as f:
        pickle.dump(res_info_dict, f)


if __name__ == '__main__':
    cfg_pattern = '[0-9]*'
    all_res_folder = 'runs/test_all'
    info_paths_dict = {
        'baseline': f'{all_res_folder}/baseline/{cfg_pattern}/results/metric.txt',
        'baseline_4x': f'{all_res_folder}/baseline_4x/{cfg_pattern}/results/metric.txt',
        'scale_normal': f'{all_res_folder}/hyperprior_scale_normal/{cfg_pattern}/results/metric.txt',
        'factorized': f'{all_res_folder}/hyperprior_factorized/{cfg_pattern}/results/metric.txt',
        'lossl_based': f'{all_res_folder}/lossl_based/{cfg_pattern}/results/metric.txt'
    }
    vs_list = [
        ('vpcc', 'baseline'),
        ('vpcc', 'baseline_4x'),
        ('vpcc', 'scale_normal'),
        ('vpcc', 'factorized'),
        ('vpcc', 'lossl_based')
    ]
    output_dir = 'runs/comparison'
    main_output_script(info_paths_dict, vs_list, output_dir)

    with open(os.path.join(output_dir, 'pickle'), 'rb') as f:
        res_info_dict = pickle.load(f)
    # sample-wise line graph
    figs = []
    for sample_name in sample_names:
        fig = plt.figure().add_subplot(111)
        figs.append(fig)
        for cfg_name in ['baseline', 'scale_normal', 'factorized', 'lossl_based', 'vpcc']:
            tmp_x_axis = res_info_dict[cfg_name][sample_name]["bpp"]
            tmp_y_axis = res_info_dict[cfg_name][sample_name]["mseF,PSNR (p2point)"]
            fig.plot(tmp_x_axis, tmp_y_axis, label=cfg_name.replace('_', ' '))
            fig.set_xlabel('bpp')
            fig.set_ylabel('D1')
            fig.set_title(sample_name)
            fig.legend()
        # plt.show()

    print('Done')
