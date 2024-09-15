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
from scripts.script_config import metric_dict_filename, test_dir
from lib.metrics.bjontegaard import bdrate, bdsnr


def compute_bd(info_dicts_a, info_dicts_b, samples_name: List[str],
               rate=True, d1=True, c=-1, pcqm=False, graphsim=False) -> Dict[str, float]:
    bd_rate_results = {}
    distortion_key = 'mseF,PSNR (p2point)' if d1 else 'mseF,PSNR (p2plane)'
    if c != -1:
        distortion_key = f'c[{c}],PSNRF'
    bd_fn = bdrate if rate else bdsnr
    if pcqm is True:
        distortion_key = 'PCQM'
    if graphsim is True:
        distortion_key = 'GraphSIM'
    for key in samples_name:
        try:
            bpp_psnr_a = list(zip(info_dicts_a[key]['bpp'], info_dicts_a[key][distortion_key]))
            bpp_psnr_b = list(zip(info_dicts_b[key]['bpp'], info_dicts_b[key][distortion_key]))
        except KeyError:
            bd_rate_results[key] = None
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


def plot_rd(method_to_json: Dict[str, all_file_metric_dict_type], method_to_plt_cfg,
            output_dir, d1=True, c=-1, pcqm=False, graphsim=False, hook=None, tight_legend=True):
    distortion_key = 'mseF,PSNR (p2point)' if d1 else 'mseF,PSNR (p2plane)'
    y_label = 'D1 PSNR (dB)' if d1 else 'D2 PSNR (dB)'
    if c != -1:
        distortion_key = f'c[{c}],PSNRF'
        y_label = 'Y PSNR (dB)' if c == 0 else 'U PSNR (dB)' if c == 1 else 'V PSNR (dB)' if c == 2 else 'YUV PSNR (dB)'
    if pcqm is True:
        distortion_key = y_label = 'PCQM'
    if graphsim is True:
        distortion_key = y_label = 'GraphSIM'
    output_dir = osp.join(output_dir, f'sample-wise {y_label}')
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    sample_names = next(iter(method_to_json.values())).keys()
    for sample_name in sample_names:
        fig = plt.figure(figsize=(4.5, 3.5))
        ax = fig.add_subplot(111)
        if not pcqm and not graphsim:
            ax.yaxis.set_major_locator(MultipleLocator(2))
        elif graphsim:
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
        elif pcqm:
            ax.yaxis.set_major_locator(MultipleLocator(0.005))
            fig.subplots_adjust(left=0.15)
        ax.grid()
        ax.tick_params(pad=0.5)
        ax.set_xlabel('bpp (bits per input point)', labelpad=-1)
        ax.set_ylabel(y_label, labelpad=0)
        ax.set_title(osp.splitext(osp.split(sample_name)[1])[0])
        for method_name, method_json in method_to_json.items():
            plt_config = method_to_plt_cfg[method_name]
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
                if isinstance(plt_config, str):
                    ax.plot(tmp_x_axis, tmp_y_axis, plt_config, label=method_name)
                else:
                    ax.plot(tmp_x_axis, tmp_y_axis, **plt_config, label=method_name)
        if tight_legend:
            legend_args = dict(
                fontsize=10, labelspacing=0.1, borderpad=0.2,
                handlelength=1.3, handletextpad=0.2, borderaxespad=0.1)
        else:
            legend_args = {}
        if not pcqm:
            ax.legend(loc='lower right', **legend_args)
        else:
            ax.legend(loc='upper right', **legend_args)
        ax.figure.savefig(osp.join(
            output_dir, f'{y_label} {osp.splitext(osp.split(sample_name)[1])[0]}.pdf'
        ))
        plt.close(ax.figure)
    print(f'Plot "{y_label}" Done')


def list_mean(ls: List):
    return sum(ls) / len(ls)


def remove_non_overlapping_points(method_name, sample_name, sorted_indices):
    # Change the numbers below as needed.
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
    elif method_name == 'Ours geo + G-PCC predlift':
        slice_val = (None, -1)
    elif method_name == 'Ours geo + G-PCC raht':
        slice_val = (None, -1)
    elif method_name == 'PCGCv2 + G-PCC predlift':
        slice_val = (None, -1)
    elif method_name == 'PCGCv2 + G-PCC raht':
        slice_val = (None, -1)
    else:
        slice_val = (None, None)
    return sorted_indices[slice_val[0]: slice_val[1]]


def compute_multiple_bdrate():
    # Change the flags and paths below as needed.
    anchor_name = 'Ours'
    anchor_secondly = True
    if_plot_rd = True
    tight_legend = False
    method_configs = {
        'Ours': ('convolutional/lossy_coord_v2/baseline_r*', {'color': '#1f77b4', 'marker': '.'}),
        # 'Ours w/o geometry residual':
        #     ('convolutional/lossy_coord_v2/gpcc_based_r*', {'color': '#ff7f0e', 'marker': '.'}),
        # 'Ours w/o feature residual':
        #     ('convolutional/lossy_coord_v2/wo_residual_r*', {'color': '#2ca02c', 'marker': '.'}),
        # 'Ours w/ expanded channels':
        #     ('convolutional/lossy_coord_v2/expanded_r*', {'color': '#d62728', 'marker': '.'}),
        # 'Ours w/o entropy restriction':
        #     ('convolutional/lossy_coord_v2/wo_bpp_r*', {'color': '#8c564b', 'marker': '.'}),
        # 'Ours part6e5': ('convolutional/lossy_coord_v2/part6e5_r*', {'color': '#9467bd', 'marker': '.'}),
        # 'Ours joint': ('convolutional/lossy_coord_lossy_color/baseline_r*', {'color': '#1f77b4', 'marker': '.'}),
        # 'Ours geo + G-PCC predlift': ('convolutional/lossy_coord_v2_predlift/baseline_r*',
        #                               {'color': '#ff7f0e', 'marker': '.'}),
        # 'Ours geo + G-PCC raht': ('convolutional/lossy_coord_v2_raht/baseline_r*',
        #                           {'color': '#d62728', 'marker': '.'}),
        # 'PCGCv2 + G-PCC predlift': ('convolutional/lossy_coord_predlift/baseline/*',
        #                             {'color': '#2ca02c', 'marker': '.'}),
        # 'PCGCv2 + G-PCC raht': ('convolutional/lossy_coord_raht/baseline/*',
        #                         {'color': '#9467bd', 'marker': '.'}),
        # 'Ours': ('convolutional/lossy_coord_v2/baseline_kitti_r*', {'color': '#1f77b4', 'marker': '.'}),
        # 'Ours': ('convolutional/lossy_coord_v2/baseline_kitti_q1mm_r*', {'color': '#1f77b4', 'marker': '.'}),

        'SparsePCGC': ('SparsePCGC/dense_lossy', {'color': '#ff7f0e', 'marker': '.'}),
        # 'SparsePCGC': ('SparsePCGC/kitti_q1mm', {'color': '#ff7f0e', 'marker': '.'}),
        'PCGCv2': ('convolutional/lossy_coord/baseline/*', {'color': '#2ca02c', 'marker': '.'}),
        'V-PCC': ('tmc2_geo', {'color': '#d62728', 'marker': '.'}),
        'ADLPCC': ('ADLPCC', {'color': '#9467bd', 'marker': '.'}),
        'G-PCC octree': ('tmc3_geo/octree', {'color': '#8c564b', 'marker': '.'}),
        # 'OctAttention': ('OctAttention-lidar', {'color': '#e377c2', 'marker': '.'}),
        # 'EHEM': ('EHEM', {'color': '#bcbd22', 'marker': '.'}),
        # 'Light EHEM': ('Light-EHEM', {'color': '#17becf', 'marker': '.'}),
        # 'pcc-geo-color': ('pcc-geo-color', {'color': '#8c564b', 'marker': '.'}),
        # 'G-PCC octree-predlift': ('tmc3/octree-predlift', {'color': '#bcbd22', 'marker': '.'}),
        # 'G-PCC octree-raht': ('tmc3/octree-raht', {'color': '#17becf', 'marker': '.'}),
    }

    # Change the flags and paths above as needed.
    # Code below should not be frequently changed.
    output_dir = 'runs/comparisons'
    rel_json_path_pattern = osp.join(test_dir, '{}', metric_dict_filename)

    method_to_json_path: Dict[str, Union[str, List[str]]] = {}
    method_to_plt_cfg = {}
    for k, v in method_configs.items():
        method_to_json_path[k] = v[0]
        method_to_plt_cfg[k] = v[1]

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
            sorted_indices = remove_non_overlapping_points(method_name, sample_name, sorted_indices)
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
    sample_wise_bdrate_pcqm: metric_dict_t = {}
    sample_wise_bdrate_graphsim: metric_dict_t = {}
    sample_wise_bd_pcqm: metric_dict_t = {}
    sample_wise_bd_graphsim: metric_dict_t = {}
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
                sample_wise_bdrate_d1, compute_bd(*comparison_tuple, sample_names))
            concat_values_for_dict(
                sample_wise_bdrate_d2, compute_bd(*comparison_tuple, sample_names, d1=False))
            concat_values_for_dict(
                sample_wise_bdpsnr_d1, compute_bd(*comparison_tuple, sample_names, rate=False))
            concat_values_for_dict(
                sample_wise_bdpsnr_d2, compute_bd(*comparison_tuple, sample_names, rate=False, d1=False))
            concat_values_for_dict(
                sample_wise_bdrate_y, compute_bd(*comparison_tuple, sample_names, c=0))
            concat_values_for_dict(
                sample_wise_bdpsnr_y, compute_bd(*comparison_tuple, sample_names, rate=False, c=0))
            concat_values_for_dict(
                sample_wise_bdrate_u, compute_bd(*comparison_tuple, sample_names, c=1))
            concat_values_for_dict(
                sample_wise_bdpsnr_u, compute_bd(*comparison_tuple, sample_names, rate=False, c=1))
            concat_values_for_dict(
                sample_wise_bdrate_v, compute_bd(*comparison_tuple, sample_names, c=2))
            concat_values_for_dict(
                sample_wise_bdpsnr_v, compute_bd(*comparison_tuple, sample_names, rate=False, c=2))
            concat_values_for_dict(
                sample_wise_bdrate_yuv, compute_bd(*comparison_tuple, sample_names, c=3))
            concat_values_for_dict(
                sample_wise_bdpsnr_yuv, compute_bd(*comparison_tuple, sample_names, rate=False, c=3))
            concat_values_for_dict(
                sample_wise_bdrate_pcqm, compute_bd(*comparison_tuple, sample_names, rate=True, pcqm=True))
            concat_values_for_dict(
                sample_wise_bdrate_graphsim, compute_bd(*comparison_tuple, sample_names, rate=True, graphsim=True))
            concat_values_for_dict(
                sample_wise_bd_pcqm, compute_bd(*comparison_tuple, sample_names, rate=False, pcqm=True))
            concat_values_for_dict(
                sample_wise_bd_graphsim, compute_bd(*comparison_tuple, sample_names, rate=False, graphsim=True))

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
             bdrate_u, bdpsnr_u, bdrate_v, bdpsnr_v, bdrate_yuv, bdpsnr_yuv,
             bdrate_pcqm, bdrate_graphsim, bd_pcqm, bd_graphsim) in zip(
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
            sample_wise_bdrate_pcqm[sample_name],
            sample_wise_bdrate_graphsim[sample_name],
            sample_wise_bd_pcqm[sample_name],
            sample_wise_bd_graphsim[sample_name]
        ):
            sample_wise_bd_metric[sample_name].extend(
                [bdrate_d1, bdrate_d2, bdrate_y, bdrate_u, bdrate_v, bdrate_yuv, bdrate_pcqm, bdrate_graphsim,
                 bdpsnr_d1, bdpsnr_d2, bdpsnr_y, bdpsnr_u, bdpsnr_v, bdpsnr_yuv, bd_pcqm, bd_graphsim]
            )
    bd_filename = f'{anchor_name.replace("*", "^")}'
    if anchor_secondly:
        bd_filename = 'bd gains ' + bd_filename
    else:
        bd_filename += ' bd gains'
    bd_filename += '.csv'
    write_metric_to_csv(
        (method_names_to_compare, ('BD-Rate (%)', 'BD-PSNR (dB)'),
         ('D1', 'D2', 'Y', 'U', 'V', 'YUV', 'PCQM', 'GraphSIM')),
        sample_wise_bd_metric, osp.join(output_dir, bd_filename)
    )

    if if_plot_rd:
        plot_rd(method_to_json, method_to_plt_cfg, output_dir, tight_legend=tight_legend)
        plot_rd(method_to_json, method_to_plt_cfg, output_dir, d1=False, tight_legend=tight_legend)
        plot_rd(method_to_json, method_to_plt_cfg, output_dir, c=0, tight_legend=tight_legend)
        plot_rd(method_to_json, method_to_plt_cfg, output_dir, c=1, tight_legend=tight_legend)
        plot_rd(method_to_json, method_to_plt_cfg, output_dir, c=2, tight_legend=tight_legend)
        plot_rd(method_to_json, method_to_plt_cfg, output_dir, c=3, tight_legend=tight_legend)
        plot_rd(method_to_json, method_to_plt_cfg, output_dir, pcqm=True, tight_legend=tight_legend)
        plot_rd(method_to_json, method_to_plt_cfg, output_dir, graphsim=True, tight_legend=tight_legend)
    print('All Done')


if __name__ == '__main__':
    compute_multiple_bdrate()
