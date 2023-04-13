import json
import re
import os
import os.path as osp
import shutil
from typing import Dict, List, Union, Tuple

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from scripts.log_extract_utils import all_file_metric_dict_type, concat_values_for_dict, concat_values_for_dict_2
from scripts.shared_config import metric_dict_filename
from lib.metrics.bjontegaard import bdrate, bdsnr


figure_title_re_pattern = re.compile('[a-z,_]+_')  # basketball_player_vox11_00000200 -> basketball_player


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


def write_metric_to_csv(titles: Tuple[Union[List[str], Tuple[str, ...]], ...],
                        sample_wise_metric, output_file):
    titles_lens = (len(_) for _ in titles[1:])
    prefix_products = [1]
    for titles_len in titles_lens:
        prefix_products.append(prefix_products[-1] * titles_len)
    prefix_products.reverse()
    total_cols = len(titles[0]) * prefix_products[0]

    with open(output_file, 'w') as f:
        for sub_titles, sub_titles_len in zip(titles, prefix_products):
            for i in range(total_cols // (len(sub_titles) * sub_titles_len)):
                for t in sub_titles:
                    f.write(f',{t}')
                    for _ in range(1, sub_titles_len):
                        f.write(',')
            if sub_titles_len == 1:
                f.write(',')
            f.write('\n')
        for key, value in sample_wise_metric.items():
            f.write(f'{key}, {",".join(map(str, value))},\n')


def plot_bpp_psnr(method_to_json: Dict[str, all_file_metric_dict_type],
                  output_dir, d1=True, c=-1, hook=None):
    distortion_key = 'mseF,PSNR (p2point)' if d1 else 'mseF,PSNR (p2plane)'
    y_label = 'D1 PSNR' if d1 else 'D2 PSNR'
    if c != -1:
        distortion_key = f'c[{c}],PSNRF'
        y_label = 'Y PSNR' if c == 0 else 'U PSNR' if c == 1 else 'V PSNR'
    output_dir = osp.join(output_dir, f'sample-wise {y_label}')
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    sample_names = next(iter(method_to_json.values())).keys()
    for sample_name in sample_names:
        fig = plt.figure(figsize=(4.5, 2.5)).add_subplot(111)
        fig.yaxis.set_major_locator(MultipleLocator(2))
        fig.grid()
        fig.set_xlabel('bpp')
        fig.set_ylabel(y_label)
        fig.set_title(osp.splitext(osp.split(sample_name)[1])[0])
        #     re.match(
        #         figure_title_re_pattern,
        #         osp.split(sample_name)[1]
        #     ).group()[:-1].replace('_', ' ')
        # )
        for method_name, method_json in method_to_json.items():
            if sample_name not in method_json: continue
            tmp_x_axis = method_json[sample_name]['bpp']
            tmp_y_axis = method_json[sample_name][distortion_key]
            xy_tuple = zip(tmp_x_axis, tmp_y_axis)
            tmp_x_axis, tmp_y_axis = zip(*sorted(xy_tuple, key=sort_key_func))
            if hook is not None:
                tmp_x_axis, tmp_y_axis = hook(tmp_x_axis, tmp_y_axis, method_name)
            if len(tmp_x_axis) and len(tmp_y_axis):
                fig.plot(tmp_x_axis, tmp_y_axis, '.-', label=method_name)
        fig.legend(loc='lower right')
        fig.figure.savefig(osp.join(
            output_dir, f'{y_label} {osp.splitext(osp.split(sample_name)[1])[0]}.pdf'
        ))
    print(f'Plot "{y_label}" Done')


def list_mean(ls: List):
    return sum(ls) / len(ls)


def compute_multiple_bdrate():
    anchor_name = 'Ours'
    anchor_secondly = True
    draw_anchor = True
    output_dir = 'runs/comparison'
    rel_json_path_pattern = osp.join('runs', 'tests', '{}', metric_dict_filename)
    method_to_json_path: Dict[str, Union[str, List[str]]] = {
        'Ours': ['convolutional/no_par/lossl_based',
                 'convolutional/no_par/lossl_based_2x'],
        # 'Ours*': ['convolutional/par6e5/lossl_based',
        #           'convolutional/par6e5/lossl_based_2x'],
        # 'Ours**': ['convolutional/par15e4/lossl_based',
        #            'convolutional/par15e4/lossl_based_2x'],
        # 'Ours w/o integrated lossless\n geometry compression':
        #     ['convolutional/no_par/gpcc_lossl_based',
        #      'convolutional/no_par/gpcc_lossl_based_2x'],
        # 'Ours w/o integrated lossless\n geometry compression*':
        #     ['convolutional/par6e5/gpcc_lossl_based',
        #      'convolutional/par6e5/gpcc_lossl_based_2x'],
        # 'Ours w/o integrated lossless\n geometry compression**':
        #     ['convolutional/par15e4/gpcc_lossl_based',
        #      'convolutional/par15e4/gpcc_lossl_based_2x'],
        'PCGCv2': 'convolutional/no_par/baseline',
        # 'PCGCv2*': 'convolutional/par6e5/baseline',
        # 'PCGCv2**': 'convolutional/par15e4/baseline',
        # 'Deep backbone w/o residual': 'convolutional/no_par/baseline_4x',
        # 'Deep backbone w/o residual*': 'convolutional/par6e5/baseline_4x',
        # 'Deep backbone w/o residual**': 'convolutional/par15e4/baseline_4x',
        'V-PCC': 'tmc2_geo',
        'ADLPCC': 'ADLPCC',
        # 'G-PCC octree-raht': 'tmc3/octree-raht'
        # 'G-PCC octree-predlift': 'tmc3/octree-predlift'
        'G-PCC octree': 'tmc3_geo/octree',
        'G-PCC trisoup': 'tmc3_geo/trisoup'
    }
    method_to_json_path = {
        k: [rel_json_path_pattern.format(v_) if not osp.isabs(v_) else v_
            for v_ in (v if isinstance(v, list) else [v])]
        for k, v in method_to_json_path.items()}

    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    method_to_json: Dict[str, all_file_metric_dict_type] = {}
    for method_name, json_path in method_to_json_path.items():
        if not isinstance(json_path, list):
            json_path = [json_path]
        with open(json_path[0], 'rb') as f:
            method_to_json[method_name] = json.load(f)
        for jp in json_path[1:]:
            with open(jp, 'rb') as f:
                concat_values_for_dict_2(method_to_json[method_name], json.load(f))

    method_names_to_compare = [key for key in method_to_json if key != anchor_name]
    sample_names = list(method_to_json[anchor_name].keys())
    sample_wise_metric_type = Dict[str, List[float]]
    sample_wise_bdrate_d1: sample_wise_metric_type = {}
    sample_wise_bdrate_d2: sample_wise_metric_type = {}
    sample_wise_bdpsnr_d1: sample_wise_metric_type = {}
    sample_wise_bdpsnr_d2: sample_wise_metric_type = {}
    sample_wise_time_complexity: sample_wise_metric_type = {}
    sample_wise_mem_complexity: sample_wise_metric_type = {}

    for method_name, method_json in method_to_json.items():
        single_sample_time_complexity = {}
        single_sample_mem_complexity = {}
        for sample_name in sample_names:
            single_sample_json = method_json[sample_name] if sample_name in method_json else {}
            single_sample_time_complexity[sample_name] = [
                list_mean(single_sample_json['encode time'])
                if 'encode time' in single_sample_json else -1,
                list_mean(single_sample_json['decode time'])
                if 'decode time' in single_sample_json else -1]
            single_sample_mem_complexity[sample_name] = [
                list_mean(single_sample_json['encode memory']) / 1024 ** 2  # KB -> GB
                if 'encode memory' in single_sample_json else -1,
                list_mean(method_json[sample_name]['decode memory']) / 1024 ** 2
                if 'decode memory' in single_sample_json else -1]
        concat_values_for_dict(sample_wise_time_complexity, single_sample_time_complexity)
        concat_values_for_dict(sample_wise_mem_complexity, single_sample_mem_complexity)

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
        (list(method_to_json.keys()), ('Enc', 'Dec')),
        sample_wise_time_complexity, osp.join(output_dir, 'Time_Complexity.csv')
    )
    write_metric_to_csv(
        (list(method_to_json.keys()), ('Enc', 'Dec')),
        sample_wise_mem_complexity, osp.join(output_dir, 'Mem_Complexity.csv')
    )
    sample_wise_bd_metric = {}
    for sample_name in sample_wise_bdrate_d1:
        sample_wise_bd_metric[sample_name] = []
        for bdrate_d1, bdrate_d2, bdpsnr_d1, bdpsnr_d2 in zip(
            sample_wise_bdrate_d1[sample_name],
            sample_wise_bdrate_d2[sample_name],
            sample_wise_bdpsnr_d1[sample_name],
            sample_wise_bdpsnr_d2[sample_name]
        ):
            sample_wise_bd_metric[sample_name].extend(
                [bdrate_d1, bdrate_d2, bdpsnr_d1, bdpsnr_d2]
            )
    bd_filename = f'{anchor_name.replace("*", "^")}'
    if anchor_secondly:
        bd_filename = 'bd gains ' + bd_filename
    else:
        bd_filename += ' bd gains'
    bd_filename += '.csv'
    write_metric_to_csv(
        (method_names_to_compare, ('BD-Rate', 'BD-PSNR'), ('D1', 'D2')),
        sample_wise_bd_metric, osp.join(output_dir, bd_filename)
    )

    def remove_low_psnr_for_vis(tmp_x_axis, tmp_y_axis, method_name):
        if method_name == 'G-PCC octree':
            tmp_x_axis, tmp_y_axis = tmp_x_axis[1:-1], tmp_y_axis[1:-1]
        elif method_name == 'G-PCC trisoup':
            tmp_x_axis, tmp_y_axis = tmp_x_axis[1:], tmp_y_axis[1:]
        elif method_name == 'ADLPCC':
            tmp_x_axis, tmp_y_axis = tmp_x_axis[:-2], tmp_y_axis[:-2]
        elif not draw_anchor and method_name == anchor_name:
            tmp_x_axis, tmp_y_axis = (), ()
        return tmp_x_axis, tmp_y_axis

    plot_bpp_psnr(method_to_json, output_dir, hook=remove_low_psnr_for_vis)
    plot_bpp_psnr(method_to_json, output_dir, d1=False, hook=remove_low_psnr_for_vis)
    # plot_bpp_psnr(method_to_json, output_dir, c=0)
    # plot_bpp_psnr(method_to_json, output_dir, c=1)
    # plot_bpp_psnr(method_to_json, output_dir, c=2)
    print('All Done')


if __name__ == '__main__':
    compute_multiple_bdrate()
