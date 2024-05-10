import json
import sys
import os
import os.path as osp
import shutil
from glob import glob
from typing import Dict, List, Union, Tuple

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

sys.path.append(osp.dirname(osp.dirname(__file__)))
from scripts.log_extract_utils import all_file_metric_dict_type, concat_values_for_dict, concat_values_for_dict_2
from scripts.shared_config import metric_dict_filename, test_dir
from lib.metrics.bjontegaard import bdrate, bdsnr


def compute_bd(info_dicts_a: all_file_metric_dict_type,
               info_dicts_b: all_file_metric_dict_type,
               rate=True, d1=True, c=-1) -> Dict[str, float]:
    bd_rate_results = {}
    distortion_key = 'mseF,PSNR (p2point)' if d1 else 'mseF,PSNR (p2plane)'
    if c != -1:
        distortion_key = f'c[{c}],PSNRF'
    bd_fn = bdrate if rate else bdsnr
    for key in info_dicts_a:
        if key in info_dicts_b:
            try:
                bpp_psnr_a = list(zip(info_dicts_a[key]['bpp'], info_dicts_a[key][distortion_key]))
                bpp_psnr_b = list(zip(info_dicts_b[key]['bpp'], info_dicts_b[key][distortion_key]))
            except KeyError:
                bd_rate_results[key] = -1
            else:
                bd_rate_results[key] = bd_fn(bpp_psnr_a, bpp_psnr_b)
    return bd_rate_results


def write_metric_to_csv(titles: Tuple[Union[List[str], Tuple[str, ...]], ...],
                        sample_wise_metric, output_file):
    titles_lens = [len(_) for _ in titles[1:]]
    titles_lens.reverse()
    prefix_products = [1]
    for titles_len in titles_lens:
        prefix_products.append(prefix_products[-1] * titles_len)
    prefix_products.reverse()
    total_cols = len(titles[0]) * prefix_products[0]

    with open(output_file, 'w') as f:
        for sub_titles, sub_titles_len in zip(titles, prefix_products):
            for i in range(total_cols // (len(sub_titles) * sub_titles_len)):
                for t in sub_titles:
                    f.write(',')
                    f.write(t.replace("\n", ""))
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
    y_label = 'D1 PSNR (dB)' if d1 else 'D2 PSNR (dB)'
    if c != -1:
        distortion_key = f'c[{c}],PSNRF'
        y_label = 'Y PSNR (dB)' if c == 0 else 'U PSNR (dB)' if c == 1 else 'V PSNR (dB)' if c == 2 else 'YUV PSNR (dB)'
    output_dir = osp.join(output_dir, f'sample-wise {y_label}')
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    sample_names = next(iter(method_to_json.values())).keys()
    for sample_name in sample_names:
        fig = plt.figure(figsize=(4.5, 3.5)).add_subplot(111)
        fig.yaxis.set_major_locator(MultipleLocator(2))
        fig.grid()
        fig.tick_params(pad=0.5)
        fig.set_xlabel('bpp (bits per input point)', labelpad=-1)
        fig.set_ylabel(y_label, labelpad=0)
        fig.set_title(osp.splitext(osp.split(sample_name)[1])[0])
        for method_name, method_json in method_to_json.items():
            if sample_name not in method_json: continue
            tmp_x_axis = method_json[sample_name]['bpp']
            if distortion_key not in method_json[sample_name]:
                print(f'Skip plotting "{y_label}" due to the missing of '
                      f'{method_name}: {sample_name}: {distortion_key}')
                shutil.rmtree(output_dir)
                return
            tmp_y_axis = method_json[sample_name][distortion_key]
            if hook is not None:
                tmp_x_axis, tmp_y_axis = hook(tmp_x_axis, tmp_y_axis, method_name)
            if len(tmp_x_axis) and len(tmp_y_axis):
                fig.plot(tmp_x_axis, tmp_y_axis, '.-', label=method_name)
        fig.legend(loc='lower right')
        fig.figure.savefig(osp.join(
            output_dir, f'{y_label} {osp.splitext(osp.split(sample_name)[1])[0]}.pdf'
        ))
        plt.close(fig.figure)
    print(f'Plot "{y_label}" Done')


def list_mean(ls: List):
    return sum(ls) / len(ls)


def remove_low_psnr_point(method_name, sample_name, sorted_indices):
    if method_name == 'G-PCC octree':
        slice_val = (1, -1)
    elif method_name == 'ADLPCC':
        slice_val = (None, -2)
    elif method_name == 'SparsePCGC':
        slice_val = (1, None)
    elif method_name == 'G-PCC octree-raht':
        slice_val = (1, -1)
    elif method_name == 'G-PCC octree-predlift':
        slice_val = (1, -1)
    else:
        slice_val = (None, None)
    return sorted_indices[slice_val[0]: slice_val[1]]


def compute_multiple_bdrate():
    anchor_name = 'Ours'
    anchor_secondly = True
    plot_rd = True
    method_to_json_path: Dict[str, Union[str, List[str]]] = {
        'Ours': 'convolutional/lossy_coord_v2/baseline_r*',
        # 'Ours w/o geometry residual':
        #     'convolutional/lossy_coord_v2/gpcc_based_r*',
        # 'Ours w/o feature residual':
        #     'convolutional/lossy_coord_v2/wo_residual_r*',
        # 'Ours part6e5': 'convolutional/lossy_coord_v2/baseline_part6e5_r*',
        # 'Ours color': 'convolutional/lossy_coord_lossy_color/baseline_r*',
        # 'Ours KITTI': 'convolutional/lossy_coord_v2/baseline_kitti_r*',
        # 'Ours KITTI q1mm': 'convolutional/lossy_coord_v2/baseline_kitti_q1mm_r*',

        'PCGCv2': 'convolutional/lossy_coord/baseline',
        'SparsePCGC': 'SparsePCGC',
        'V-PCC': 'tmc2_geo',
        # 'ADLPCC': 'ADLPCC',
        'G-PCC octree': 'tmc3_geo/octree',
        # 'G-PCC octree-raht': 'tmc3/octree-raht',
        # 'G-PCC octree-predlift': 'tmc3/octree-predlift',
    }
    output_dir = 'runs/comparisons'
    rel_json_path_pattern = osp.join(test_dir, '{}', metric_dict_filename)

    for k, v in method_to_json_path.items():
        if not isinstance(v, list):
            v = (v,)
        method_to_json_path[k] = []
        for v_ in v:
            if not osp.isabs(v_):
                v_ = rel_json_path_pattern.format(v_)
            method_to_json_path[k].extend(sorted(glob(v_)))

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
        for sample_name, metric_dict in method_to_json[method_name].items():
            sorted_indices = sorted(range(len(metric_dict['bpp'])), key=lambda _: metric_dict['bpp'][_])
            sorted_indices = remove_low_psnr_point(method_name, sample_name, sorted_indices)
            for k, v in metric_dict.items():
                metric_dict[k] = [v[_] for _ in sorted_indices]

    method_names_to_compare = [key for key in method_to_json if key != anchor_name]
    sample_names = list(method_to_json[anchor_name].keys())
    metric_dict_t = Dict[str, List[float]]
    sample_wise_bdrate_d1: metric_dict_t = {}
    sample_wise_bdrate_d2: metric_dict_t = {}
    sample_wise_bdpsnr_d1: metric_dict_t = {}
    sample_wise_bdpsnr_d2: metric_dict_t = {}
    sample_wise_bdrate_y: metric_dict_t = {}
    sample_wise_bdpsnr_y: metric_dict_t = {}
    sample_wise_bdrate_u: metric_dict_t = {}
    sample_wise_bdpsnr_u: metric_dict_t = {}
    sample_wise_bdrate_v: metric_dict_t = {}
    sample_wise_bdpsnr_v: metric_dict_t = {}
    sample_wise_bdrate_yuv: metric_dict_t = {}
    sample_wise_bdpsnr_yuv: metric_dict_t = {}
    sample_wise_time_complexity: metric_dict_t = {}
    sample_wise_mem_complexity: metric_dict_t = {}

    for method_name, method_json in method_to_json.items():
        single_sample_time_complexity = {}
        single_sample_mem_complexity = {}
        for sample_name in sample_names:
            single_sample_json = method_json[sample_name] if sample_name in method_json else {}
            single_sample_time_complexity[sample_name] = [
                list_mean(single_sample_json['encode time'])
                if 'encode time' in single_sample_json else None,
                list_mean(single_sample_json['decode time'])
                if 'decode time' in single_sample_json else None]
            single_sample_mem_complexity[sample_name] = [
                list_mean(single_sample_json['encode memory']) / 1024 ** 2  # KB -> GB
                if 'encode memory' in single_sample_json else None,
                list_mean(method_json[sample_name]['decode memory']) / 1024 ** 2
                if 'decode memory' in single_sample_json else None]
        concat_values_for_dict(sample_wise_time_complexity, single_sample_time_complexity)
        concat_values_for_dict(sample_wise_mem_complexity, single_sample_mem_complexity)

        if method_name != anchor_name:
            comparison_tuple = (method_to_json[anchor_name], method_json)
            if anchor_secondly: comparison_tuple = comparison_tuple[::-1]
            concat_values_for_dict(
                sample_wise_bdrate_d1, compute_bd(*comparison_tuple))
            concat_values_for_dict(
                sample_wise_bdrate_d2, compute_bd(*comparison_tuple, d1=False))
            concat_values_for_dict(
                sample_wise_bdpsnr_d1, compute_bd(*comparison_tuple, rate=False))
            concat_values_for_dict(
                sample_wise_bdpsnr_d2, compute_bd(*comparison_tuple, rate=False, d1=False))
            concat_values_for_dict(
                sample_wise_bdrate_y, compute_bd(*comparison_tuple, c=0))
            concat_values_for_dict(
                sample_wise_bdpsnr_y, compute_bd(*comparison_tuple, rate=False, c=0))
            concat_values_for_dict(
                sample_wise_bdrate_u, compute_bd(*comparison_tuple, c=1))
            concat_values_for_dict(
                sample_wise_bdpsnr_u, compute_bd(*comparison_tuple, rate=False, c=1))
            concat_values_for_dict(
                sample_wise_bdrate_v, compute_bd(*comparison_tuple, c=2))
            concat_values_for_dict(
                sample_wise_bdpsnr_v, compute_bd(*comparison_tuple, rate=False, c=2))
            concat_values_for_dict(
                sample_wise_bdrate_yuv, compute_bd(*comparison_tuple, c=3))
            concat_values_for_dict(
                sample_wise_bdpsnr_yuv, compute_bd(*comparison_tuple, rate=False, c=3))

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
        for (bdrate_d1, bdrate_d2, bdpsnr_d1, bdpsnr_d2, bdrate_y, bdpsnr_y,
             bdrate_u, bdpsnr_u, bdrate_v, bdpsnr_v, bdrate_yuv, bdpsnr_yuv) in zip(
            sample_wise_bdrate_d1[sample_name],
            sample_wise_bdrate_d2[sample_name],
            sample_wise_bdpsnr_d1[sample_name],
            sample_wise_bdpsnr_d2[sample_name],
            sample_wise_bdrate_y[sample_name],
            sample_wise_bdpsnr_y[sample_name],
            sample_wise_bdrate_u[sample_name],
            sample_wise_bdpsnr_u[sample_name],
            sample_wise_bdrate_v[sample_name],
            sample_wise_bdpsnr_v[sample_name],
            sample_wise_bdrate_yuv[sample_name],
            sample_wise_bdpsnr_yuv[sample_name],
        ):
            sample_wise_bd_metric[sample_name].extend(
                [bdrate_d1, bdrate_d2, bdrate_y, bdrate_u, bdrate_v, bdrate_yuv,
                 bdpsnr_d1, bdpsnr_d2, bdpsnr_y, bdpsnr_u, bdpsnr_v, bdpsnr_yuv]
            )
    bd_filename = f'{anchor_name.replace("*", "^")}'
    if anchor_secondly:
        bd_filename = 'bd gains ' + bd_filename
    else:
        bd_filename += ' bd gains'
    bd_filename += '.csv'
    write_metric_to_csv(
        (method_names_to_compare, ('BD-Rate (%)', 'BD-PSNR (dB)'), ('D1', 'D2', 'Y', 'U', 'V', 'YUV')),
        sample_wise_bd_metric, osp.join(output_dir, bd_filename)
    )

    if plot_rd:
        plot_bpp_psnr(method_to_json, output_dir)
        plot_bpp_psnr(method_to_json, output_dir, d1=False)
        plot_bpp_psnr(method_to_json, output_dir, c=0)
        plot_bpp_psnr(method_to_json, output_dir, c=1)
        plot_bpp_psnr(method_to_json, output_dir, c=2)
        plot_bpp_psnr(method_to_json, output_dir, c=3)
    print('All Done')


if __name__ == '__main__':
    compute_multiple_bdrate()
