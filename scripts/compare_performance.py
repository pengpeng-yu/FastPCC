import json
import re
import os
import os.path as osp
import shutil
from typing import Dict, List

import matplotlib.pyplot as plt

from scripts.log_extract_utils import all_file_metric_dict_type, concat_values_for_dict
from scripts.shared_config import metric_dict_filename
from lib.metrics.bjontegaard import bdrate, bdsnr


figure_title_re_pattern = re.compile('[a-z,_]+_')


def sort_key_func(bpp_psnr): return bpp_psnr[0]


def compute_bd(info_dicts_a: all_file_metric_dict_type,
               info_dicts_b: all_file_metric_dict_type,
               rate=True, d1=True) -> Dict[str, float]:
    bd_rate_results = {}
    distortion_key = 'mseF,PSNR (p2point)' if d1 else 'mseF,PSNR (p2plane)'
    bd_fn = bdrate if rate else bdsnr
    for key in info_dicts_a:
        if key in info_dicts_b:
            bpp_psnr_a = list(zip(info_dicts_a[key]['bpp'], info_dicts_a[key][distortion_key]))
            bpp_psnr_b = list(zip(info_dicts_b[key]['bpp'], info_dicts_b[key][distortion_key]))
            bd_rate_results[key] = bd_fn(
                sorted(bpp_psnr_a, key=sort_key_func),
                sorted(bpp_psnr_b, key=sort_key_func)
            )
    return bd_rate_results


def write_metric_to_csv(method_names, sample_wise_metric, output_file):
    with open(output_file, 'w') as f:
        f.write(f' , {",".join(method_names)}\n')
        for key, value in sample_wise_metric.items():
            f.write(f'{key}, {",".join(map(str, value))}\n')


def plot_bpp_psnr(method_to_json: Dict[str, all_file_metric_dict_type],
                  output_dir, d1=True, hook=None):
    distortion_key = 'mseF,PSNR (p2point)' if d1 else 'mseF,PSNR (p2plane)'
    y_label = 'D1 PSNR' if d1 else 'D2 PSNR'
    output_dir = osp.join(output_dir, f'sample-wise {y_label}')
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    sample_names = next(iter(method_to_json.values())).keys()
    for sample_name in sample_names:
        fig = plt.figure(figsize=(4, 2.5)).add_subplot(111)
        fig.grid()
        fig.set_xlabel('bpp')
        fig.set_ylabel(y_label)
        fig.set_title(
            re.match(
                figure_title_re_pattern,
                osp.split(sample_name)[1]
            ).group()[:-1].replace('_', ' ')
        )
        for method_name, method_json in method_to_json.items():
            tmp_x_axis = method_json[sample_name]['bpp']
            tmp_y_axis = method_json[sample_name][distortion_key]
            xy_tuple = zip(tmp_x_axis, tmp_y_axis)
            tmp_x_axis, tmp_y_axis = zip(*sorted(xy_tuple, key=sort_key_func))
            if hook is not None:
                tmp_x_axis, tmp_y_axis = hook(tmp_x_axis, tmp_y_axis, method_name)
            fig.plot(tmp_x_axis, tmp_y_axis, '.-', label=method_name)
        fig.legend(loc='lower right')
        fig.figure.savefig(osp.join(
            output_dir, f'{y_label} {osp.splitext(osp.split(sample_name)[1])[0]}.pdf'
        ))
    print(f'Plot "{y_label}" Done')


def list_mean(ls: List):
    return sum(ls) / len(ls)


def compute_multiple_bdrate():
    anchor_name = 'V-PCC'
    anchor_secondly = False
    output_dir = 'runs/comparison'
    json_path_pattern = osp.join('runs', 'tests', '{}', metric_dict_filename)
    method_to_json_path: Dict[str, str] = {
        'G-PCC octree': 'tmc3_geo/octree',
        'G-PCC trisoup': 'tmc3_geo/trisoup',
        'V-PCC': 'tmc2_geo',
        'ADLPCC': 'ADLPCC',
        'PCGCv2': 'convolutional_all_no_par/baseline',
        'PCGCv2 4x': 'convolutional_all_no_par/baseline_4x',
        'Scale Normal': 'convolutional_all_no_par/hyperprior_scale_normal',
        'Learnable': 'convolutional_all_no_par/hyperprior_factorized',
        'Ours': 'convolutional_all_no_par/lossl_based',
        'PCGCv2*': 'convolutional_all_par6e5/baseline',
        'PCGCv2 4x*': 'convolutional_all_par6e5/baseline_4x',
        'Scale Normal*': 'convolutional_all_par6e5/hyperprior_scale_normal',
        'Learnable*': 'convolutional_all_par6e5/hyperprior_factorized',
        'Ours*': 'convolutional_all_par6e5/lossl_based',
        'PCGCv2**': 'convolutional_all_par15e4/baseline',
        'PCGCv2 4x**': 'convolutional_all_par15e4/baseline_4x',
        'Scale Normal**': 'convolutional_all_par15e4/hyperprior_scale_normal',
        'Learnable**': 'convolutional_all_par15e4/hyperprior_factorized',
        'Ours**': 'convolutional_all_par15e4/lossl_based',
    }
    method_to_json_path = {k: json_path_pattern.format(v)
                           for k, v in method_to_json_path.items()}

    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    method_to_json: Dict[str, all_file_metric_dict_type] = {}
    for method_name, json_path in method_to_json_path.items():
        with open(json_path, 'rb') as f:
            method_to_json[method_name] = json.load(f)

    method_names_to_compare = [key for key in method_to_json if key != anchor_name]
    sample_names = list(method_to_json[anchor_name].keys())
    sample_wise_metric_type = Dict[str, List[float]]
    sample_wise_bdrate_d1: sample_wise_metric_type = {}
    sample_wise_bdrate_d2: sample_wise_metric_type = {}
    sample_wise_bdpsnr_d1: sample_wise_metric_type = {}
    sample_wise_bdpsnr_d2: sample_wise_metric_type = {}
    sample_wise_complexity: sample_wise_metric_type = {}

    for method_name, method_json in method_to_json.items():
        single_sample_complexity = {}
        for sample_name in sample_names:
            single_sample_json = method_json[sample_name]
            single_sample_complexity[sample_name] = [
                list_mean(single_sample_json['encode time'])
                if 'encode time' in single_sample_json else -1,
                list_mean(single_sample_json['decode time'])
                if 'decode time' in single_sample_json else -1,
                list_mean(single_sample_json['encode memory']) / 1024 ** 2  # KB -> GB
                if 'encode memory' in single_sample_json else -1,
                list_mean(method_json[sample_name]['decode memory']) / 1024 ** 2
                if 'decode memory' in single_sample_json else -1,
            ]
        concat_values_for_dict(sample_wise_complexity, single_sample_complexity)

        if method_name != anchor_name:
            comparison_tuple = (method_to_json[anchor_name], method_json)
            if anchor_secondly: comparison_tuple = comparison_tuple[::-1]
            sample_wise_bdrate_d1 = concat_values_for_dict(
                sample_wise_bdrate_d1,
                compute_bd(*comparison_tuple)
            )
            sample_wise_bdrate_d2 = concat_values_for_dict(
                sample_wise_bdrate_d2,
                compute_bd(*comparison_tuple, d1=False)
            )
            sample_wise_bdpsnr_d1 = concat_values_for_dict(
                sample_wise_bdpsnr_d1,
                compute_bd(*comparison_tuple, rate=False)
            )
            sample_wise_bdpsnr_d2 = concat_values_for_dict(
                sample_wise_bdpsnr_d2,
                compute_bd(*comparison_tuple, rate=False, d1=False)
            )

    write_metric_to_csv(
        sum((tuple(f'{key} {_}' for _ in ('enc t', 'dec t', 'enc m', 'dec m'))
             for key in method_to_json.keys()), ()),
        sample_wise_complexity, osp.join(output_dir, 'complexity.csv')
    )
    bd_file_name = 'bd{} D{}'
    if anchor_secondly:
        bd_file_name += f' {anchor_name}'
    else:
        bd_file_name = f'{anchor_name} ' + bd_file_name
    bd_file_name += '.csv'
    write_metric_to_csv(
        method_names_to_compare, sample_wise_bdrate_d1,
        osp.join(output_dir, bd_file_name.format('rate', 1))
    )
    write_metric_to_csv(
        method_names_to_compare, sample_wise_bdrate_d2,
        osp.join(output_dir, bd_file_name.format('rate', 2))
    )
    write_metric_to_csv(
        method_names_to_compare, sample_wise_bdpsnr_d1,
        osp.join(output_dir, bd_file_name.format('psnr', 1))
    )
    write_metric_to_csv(
        method_names_to_compare, sample_wise_bdpsnr_d2,
        osp.join(output_dir, bd_file_name.format('psnr', 2))
    )

    def remove_low_psnr_for_vis(tmp_x_axis, tmp_y_axis, method_name):
        if method_name == 'G-PCC octree':
            tmp_x_axis, tmp_y_axis = tmp_x_axis[2:], tmp_y_axis[2:]
        elif method_name == 'G-PCC trisoup':
            tmp_x_axis, tmp_y_axis = tmp_x_axis[1:], tmp_y_axis[1:]
        return tmp_x_axis, tmp_y_axis

    plot_bpp_psnr(method_to_json, output_dir, hook=remove_low_psnr_for_vis)
    plot_bpp_psnr(method_to_json, output_dir, d1=False, hook=remove_low_psnr_for_vis)
    print('All Done')


if __name__ == '__main__':
    compute_multiple_bdrate()
