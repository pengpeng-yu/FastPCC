import os.path as osp
import pathlib
import re

import open3d as o3d
import numpy as np
import torch
import torch.utils.data

from lib.data_utils import PCData, pc_data_collate_fn, kd_tree_partition_randomly
from lib.morton_code import morton_encode_magicbits
from lib.datasets.PlyVoxel.Base.dataset_config import DatasetConfig


class PlyVoxel(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training: bool, logger):
        super(PlyVoxel, self).__init__()
        self.is_training = is_training

        def get_collections(*items):
            subsets_num = -1
            for x in items:
                if isinstance(x, tuple) or isinstance(x, list):
                    if subsets_num == -1:
                        subsets_num = len(x)
                    else:
                        assert len(x) == subsets_num, \
                            f'Unexpected length ({len(x)}) of a dataset config item,' \
                            f'which is expected to have a length of {subsets_num}.'
            if subsets_num == -1: subsets_num = 1
            ret = []
            for x in items:
                if isinstance(x, tuple) or isinstance(x, list):
                    ret.append(x)
                else:
                    ret.append((x,) * subsets_num)
            return ret

        roots, filelist_paths, file_path_patterns, resolutions, coord_scaler, partition_max_points_num = \
            get_collections(cfg.root, cfg.filelist_path, cfg.file_path_pattern, cfg.resolution,
                            cfg.coord_scaler, cfg.kd_tree_partition_max_points_num)

        # define files list path
        for root, filelist_path, file_path_pattern in zip(roots, filelist_paths, file_path_patterns):
            filelist_abs_path = osp.join(root, filelist_path)
            # generate files list
            if not osp.exists(filelist_abs_path):
                logger.warning(f'no filelist of {root} is given. Trying to generate using {file_path_pattern}...')
                file_list = sorted(pathlib.Path(root).glob(file_path_pattern))
                with open(filelist_abs_path, 'w') as f:
                    f.write('\n'.join([str(_.relative_to(root)) for _ in file_list]))

        # load files list
        self.file_list = []
        self.file_resolutions = []
        self.file_coord_scaler_list = []
        self.file_partition_max_points_num_list = []
        for root, filelist_path, resolution, scaler, par_num in \
                zip(roots, filelist_paths, resolutions, coord_scaler, partition_max_points_num):
            filelist_abs_path = osp.join(root, filelist_path)
            logger.info(f'using filelist: "{filelist_abs_path}"')
            with open(filelist_abs_path) as f:
                for line in f.readlines()[::cfg.list_sampling_interval]:
                    line = line.strip()
                    self.file_list.append(osp.join(root, line))
                    self.file_resolutions.append(resolution)
                    self.file_coord_scaler_list.append(scaler)
                    self.file_partition_max_points_num_list.append(par_num)

        self.cfg = cfg
        self.random_batch_coord_scaler_log2 = tuple((_, 2 ** _) for _ in self.cfg.random_batch_coord_scaler_log2)
        self.pattern_frame_idx = re.compile(r'(\d+)')

        self.file_ref_list = []
        if self.cfg.ref_frames_num > 0:
            for path in self.file_list:
                self.file_ref_list.append(self.gen_ref_frame_path(path))

    def gen_ref_frame_path(self, cur_path):
        search_res = self.pattern_frame_idx.search(cur_path[::-1])
        start = search_res.start(0)
        end = search_res.end(0)
        cur_idx_str = cur_path[-end:-start]
        cur_idx = int(cur_idx_str)

        ref_paths = []
        for i in range(self.cfg.ref_frames_num, 0, -1):
            ref_idx = cur_idx - i
            ref_path = f'{cur_path[:-end]}{ref_idx:0{len(cur_idx_str)}d}{cur_path[-start:]}'
            if ref_idx < 0 or not osp.isfile(ref_path):
                ref_paths.append(None)
            else:
                ref_paths.append(ref_path)
        return tuple(ref_paths)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return index

    def getitem(self, index, batch_coord_scaler: float = 1.0):
        # load
        file_path = self.file_list[index]
        try:
            point_cloud = o3d.t.io.read_point_cloud(file_path)
            point_cloud_ref = []
            if_load_ref = self.cfg.ref_frames_num > 0 and any([_ is not None for _ in self.file_ref_list[index]])
            if if_load_ref:
                for ref_path in self.file_ref_list[index]:
                    point_cloud_ref.append(o3d.t.io.read_point_cloud(ref_path) if ref_path is not None else None)
        except Exception as e:
            print(f'Error when loading {file_path}')
            raise e

        # xyz
        xyz = point_cloud.point.positions.numpy()
        if not np.issubdtype(xyz.dtype, np.floating):
            xyz = xyz.astype(np.float32)
        org_points_num = xyz.shape[0]
        org_point = xyz.min(0)
        if if_load_ref:
            xyzt_ref = []
            for ref_idx, pc_ref in enumerate(point_cloud_ref):
                if pc_ref is None: continue
                tmp_xyz = pc_ref.point.positions.numpy()
                if not np.issubdtype(tmp_xyz.dtype, np.floating):
                    tmp_xyz = tmp_xyz.astype(np.float32)
                if self.cfg.ref_frames_num > 1:
                    tmp_xyz = np.pad(tmp_xyz, ((0, 0), (0, 1)), constant_values=ref_idx)
                xyzt_ref.append(tmp_xyz)
            xyzt_ref = np.concatenate(xyzt_ref, axis=0)
            org_point = np.minimum(org_point, xyzt_ref[:, :3].min(0))
            xyzt_ref[:, :3] -= org_point
        else:
            xyzt_ref = None
        xyz -= org_point

        coord_scaler = self.file_coord_scaler_list[index] * batch_coord_scaler
        if coord_scaler != 1:  # Assume no attributes
            assert not self.cfg.with_color and not self.cfg.with_reflectance
            xyz = (xyz * coord_scaler).round()
            xyz = np.unique(xyz.astype(np.int32), axis=0)
            if if_load_ref:
                xyzt_ref[:, :3] *= coord_scaler
                xyzt_ref.round(out=xyzt_ref)
                xyzt_ref = np.unique(xyzt_ref.astype(np.int32), axis=0)
        else:  # Assume no duplicated points
            xyz = xyz.astype(np.int32)
            if if_load_ref:
                xyzt_ref = xyzt_ref.astype(np.int32)

        if self.cfg.with_color:
            color = point_cloud.point['colors'].numpy()
            assert color.size != 0
            assert color.shape[0] == xyz.shape[0]
            color = color.astype(np.float32)
            if if_load_ref:
                color_ref = []
                for pc_ref in point_cloud_ref:
                    if pc_ref is None: continue
                    color_ref.append(pc_ref.point['colors'].numpy())
                color_ref = np.concatenate(color_ref, dtype=np.float32)
            else:
                color_ref = None
        else:
            color = None
            color_ref = None

        if self.cfg.with_reflectance:
            refl = point_cloud.point['reflectance'].numpy()
            assert refl.size != 0
            assert refl.shape[0] == xyz.shape[0]
            refl = refl.astype(np.float32)
            if if_load_ref:
                refl_ref = []
                for pc_ref in point_cloud_ref:
                    if pc_ref is None: continue
                    refl_ref.append(pc_ref.point['reflectance'].numpy())
                refl_ref = np.concatenate(refl_ref, dtype=np.float32)
            else:
                refl_ref = None
        else:
            refl = None
            refl_ref = None

        if self.is_training:
            par_num = self.file_partition_max_points_num_list[index]
            if par_num != 0 and xyz.shape[0] > par_num:
                if not if_load_ref:
                    xyz, (color, refl) = kd_tree_partition_randomly(
                        xyz, par_num, (color, refl)
                    )
                    tmp_org_point = xyz.min(0)
                    xyz -= tmp_org_point
                else:
                    xyz, (color, refl), xyzt_ref, (color_ref, refl_ref) = kd_tree_partition_randomly(
                        xyz, par_num, (color, refl),
                        coord_ref=xyzt_ref, attrs_ref=(color_ref, refl_ref)
                    )
                    tmp_org_point = np.minimum(xyz.min(0), xyzt_ref[:, :3].min(0))
                    xyz -= tmp_org_point
                    xyzt_ref[:, :3] -= tmp_org_point
                org_point += tmp_org_point

            if self.cfg.random_flip:
                if not if_load_ref:
                    if np.random.rand() > 0.5:
                        xyz[:, 0] = -xyz[:, 0] + xyz[:, 0].max()
                    if np.random.rand() > 0.5:
                        xyz[:, 1] = -xyz[:, 1] + xyz[:, 1].max()
                else:
                    if np.random.rand() > 0.5:
                        tmp_max = np.maximum(xyz[:, 0].max(), xyzt_ref[:, 0].max())
                        xyz[:, 0] = -xyz[:, 0] + tmp_max
                        xyzt_ref[:, 0] = -xyzt_ref[:, 0] + tmp_max
                    if np.random.rand() > 0.5:
                        tmp_max = np.maximum(xyz[:, 1].max(), xyzt_ref[:, 1].max())
                        xyz[:, 1] = -xyz[:, 1] + tmp_max
                        xyzt_ref[:, 1] = -xyzt_ref[:, 1] + tmp_max

        xyz = torch.from_numpy(xyz)
        color = torch.from_numpy(color) if color is not None else None
        refl = torch.from_numpy(refl) if refl is not None else None
        xyzt_ref = torch.from_numpy(xyzt_ref) if xyzt_ref is not None else None
        color_ref = torch.from_numpy(color_ref) if color_ref is not None else None
        refl_ref = torch.from_numpy(refl_ref) if refl_ref is not None else None
        if self.cfg.morton_sort:
            order = torch.argsort(morton_encode_magicbits(xyz, inverse=self.cfg.morton_sort_inverse))
            xyz = xyz[order]
            if color is not None:
                color = color[order]
            if refl is not None:
                refl = refl[order]

        inv_trans = torch.from_numpy(np.concatenate((org_point.reshape(-1), (1 / coord_scaler,)), 0, dtype=np.float32))
        pc_data = PCData(
            xyz=xyz,
            color=color,
            reflectance=refl,
            xyzt_ref=xyzt_ref,
            color_ref=color_ref,
            reflectance_ref=refl_ref,
            file_path=file_path,
            resolution=None if self.is_training else self.file_resolutions[index],
            org_points_num=org_points_num,
            inv_transform=inv_trans
        )
        return pc_data

    def collate_fn(self, batch):
        batch_coord_scaler_log2, batch_coord_scaler = self.random_batch_coord_scaler_log2[
            np.random.randint(0, len(self.random_batch_coord_scaler_log2))]
        par_num = self.file_partition_max_points_num_list[batch[0]]
        batch = [self.getitem(index, batch_coord_scaler) for index in batch]
        if not self.is_training:
            assert len(batch) == 1
            ret = pc_data_collate_fn(batch, kd_tree_partition_max_points_num=par_num)
        else:
            ret = pc_data_collate_fn(batch)
        ret.batch_coord_scaler_log2 = batch_coord_scaler_log2 or None
        return ret


if __name__ == '__main__':
    config = DatasetConfig()
    config.with_color = True

    from loguru import logger
    dataset = PlyVoxel(config, False, logger)
    dataloader = torch.utils.data.DataLoader(dataset, 4, shuffle=False, collate_fn=dataset.collate_fn)
    dataloader = iter(dataloader)
    sample: PCData = next(dataloader)

    from lib.vis import plt_batched_sparse_xyz
    sample_coords = sample.xyz
    plt_batched_sparse_xyz(sample_coords, 0, False)
    plt_batched_sparse_xyz(sample_coords, 1, False)
    print('Done')
