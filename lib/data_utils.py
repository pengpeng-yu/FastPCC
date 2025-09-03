import os
import os.path as osp
from collections import defaultdict
from typing import Tuple, List, Optional, Union, Dict, Callable

import numpy as np
from plyfile import PlyData, PlyElement
try:
    import open3d as o3d
except ImportError: o3d = None
import torch


def batched_coordinates(coords):
    N = [len(cs) for cs in coords]
    bcoords = torch.zeros((sum(N), coords[0].shape[1] + 1), dtype=torch.int32)
    s = 0
    for b, cs in enumerate(coords):
        cn = len(cs)
        bcoords[s: s + cn, 1:] = cs
        bcoords[s: s + cn, 0] = b
        s += cn
    return bcoords, N


class SampleData:
    def __init__(self):
        self.results_dir: Optional[str] = None
        self.training_step: Optional[int] = None

    def to(self, device, non_blocking=False):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.to(device, non_blocking=non_blocking, memory_format=torch.contiguous_format)

    def pin_memory(self):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.pin_memory()
        return self


class PCData(SampleData):
    tensor_to_tensor_items = ('color', 'reflectance', 'color_ref', 'reflectance_ref')
    ref_items = ('color_ref', 'reflectance_ref')

    def __init__(self,
                 xyz: Union[torch.Tensor, List[torch.Tensor]],
                 color: Union[torch.Tensor, List[torch.Tensor]] = None,
                 reflectance: Union[torch.Tensor, List[torch.Tensor]] = None,
                 resolution: Union[float, List[float]] = None,
                 xyzt_ref: Union[torch.Tensor, List[torch.Tensor]] = None,
                 color_ref: Union[torch.Tensor, List[torch.Tensor]] = None,
                 reflectance_ref: Union[torch.Tensor, List[torch.Tensor]] = None,
                 file_path: Union[str, List[str]] = None,
                 batch_size: int = 0,
                 points_num: List[int] = None,
                 org_points_num: Union[int, List[int]] = None,
                 inv_transform: Union[torch.Tensor, List[torch.Tensor]] = None,
                 batch_coord_scaler_log2: int = None):
        super(PCData, self).__init__()
        self.xyz = xyz
        self.color = color
        self.reflectance = reflectance
        self.xyzt_ref = xyzt_ref
        self.color_ref = color_ref
        self.reflectance_ref = reflectance_ref
        self.resolution = resolution
        self.file_path = file_path
        self.batch_size = batch_size
        self.points_num = points_num
        self.org_points_num = org_points_num
        self.inv_transform = inv_transform
        self.batch_coord_scaler_log2 = batch_coord_scaler_log2

    def to(self, device, non_blocking=False):
        super(PCData, self).to(device, non_blocking)
        for key in ('xyz', *self.tensor_to_tensor_items):
            value = self.__dict__[key]
            if isinstance(value, List):
                # Ignore the first value of partitions lists.
                for idx, v in enumerate(value[1:]):
                    value[idx + 1] = v.to(device, non_blocking=non_blocking)

    def pin_memory(self):
        super(PCData, self).pin_memory()
        for key in ('xyz', *self.tensor_to_tensor_items):
            value = self.__dict__[key]
            if isinstance(value, List):
                for idx, v in enumerate(value[1:]):
                    value[idx + 1] = v.pin_memory()
        return self


def pc_data_collate_fn(data_list: List[PCData],
                       kd_tree_partition_max_points_num: int = 0) -> PCData:
    if kd_tree_partition_max_points_num > 0:
        assert len(data_list) == 1, 'Supports kd-tree partition only when batch size == 1.'
        use_kd_tree_partition = data_list[0].xyz.shape[0] > kd_tree_partition_max_points_num
    else:
        use_kd_tree_partition = False

    if not use_kd_tree_partition:
        data_dict: Dict[str, List] = defaultdict(list)
        for data in data_list:
            for key, value in data.__dict__.items():
                if value is not None:
                    data_dict[key].append(value)
        batched_data_dict = {'batch_size': len(data_list)}
        for key, value in data_dict.items():
            if key == 'xyz':
                batched_data_dict[key], batched_data_dict['points_num'] = batched_coordinates(value)
            elif key == 'xyzt_ref':
                batched_data_dict[key] = batched_coordinates(value)[0]
            elif key in PCData.tensor_to_tensor_items:
                batched_data_dict[key] = torch.cat(value, dim=0)
            elif key != 'batch_size':
                batched_data_dict[key] = value
        return PCData(**batched_data_dict)

    else:
        # Use kd-tree partition.
        pc_data = data_list[0]
        pc_data.batch_size = 1

        attrs_list = []
        attrs_ref_list = []
        for k in PCData.tensor_to_tensor_items:
            tmp_v = pc_data.__dict__[k]
            if k not in PCData.ref_items:
                attrs_list.append(tmp_v)
            else:
                attrs_ref_list.append(tmp_v)
        xyz_list, attrs_list, xyzt_ref_list, attrs_ref_list = kd_tree_partition(
            pc_data.xyz, kd_tree_partition_max_points_num,
            attrs_list, pc_data.xyzt_ref, attrs_ref_list
        )

        pc_data.xyz = [pc_data.xyz, *xyz_list]
        for i in range(len(pc_data.xyz)):
            pc_data.xyz[i] = torch.nn.functional.pad(pc_data.xyz[i], (1, 0, 0, 0), value=0)
        for k in PCData.tensor_to_tensor_items:
            if k not in PCData.ref_items:
                attr = attrs_list.pop(0)
                if attr is not None:
                    pc_data.__dict__[k] = [pc_data.__dict__[k], *attr]

        if pc_data.xyzt_ref is not None:
            pc_data.xyzt_ref = [pc_data.xyzt_ref, *xyzt_ref_list]
            for i in range(len(pc_data.xyzt_ref)):
                pc_data.xyzt_ref[i] = torch.nn.functional.pad(pc_data.xyzt_ref[i], (1, 0, 0, 0), value=0)
            for k in PCData.ref_items:
                attr = attrs_ref_list.pop(0)
                if attr is not None:
                    pc_data.__dict__[k] = [pc_data.__dict__[k], *attr]

        for k in pc_data.__dict__:
            if k != 'xyz' and k != 'xyzt_ref' and k != 'batch_size' \
                    and k not in PCData.tensor_to_tensor_items and pc_data.__dict__[k] is not None:
                pc_data.__dict__[k] = [pc_data.__dict__[k]]
        return pc_data


def kd_tree_partition(coord: Union[np.ndarray, torch.Tensor], max_num: int,
                      attrs: List[Union[np.ndarray, torch.Tensor, None]] = (),
                      coord_ref: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      attrs_ref: List[Union[np.ndarray, torch.Tensor, None]] = ()):
    is_torch_tensor = isinstance(coord, torch.Tensor)
    if is_torch_tensor:
        coord = coord.numpy()
        attrs = [_.numpy() if _ is not None else None for _ in attrs]
        if coord_ref is not None:
            coord_ref = coord_ref.numpy()
        attrs_ref = [_.numpy() if _ is not None else None for _ in attrs_ref]
    coord_list, attrs_list, coord_ref_list, attrs_ref_list = \
        _kd_tree_partition(coord, max_num, attrs, coord_ref, attrs_ref)
    if is_torch_tensor:
        coord_list = [torch.from_numpy(_) for _ in coord_list]
        attrs_list = [[torch.from_numpy(_) for _ in __] if __ is not None else None
                      for __ in attrs_list]
        coord_ref_list = [torch.from_numpy(_) for _ in coord_ref_list]
        attrs_ref_list = [[torch.from_numpy(_) for _ in __] if __ is not None else None
                          for __ in attrs_ref_list]
    return coord_list, attrs_list, coord_ref_list, attrs_ref_list


def _kd_tree_partition(coord: np.ndarray, max_num: int, attrs: List[Optional[np.ndarray]] = (),
                       coord_ref: Optional[np.ndarray] = None, attrs_ref: List[Optional[np.ndarray]] = ()) \
        -> Tuple[List[np.ndarray], List[Optional[List[np.ndarray]]],
                 List[np.ndarray], List[Optional[List[np.ndarray]]]]:
    if len(coord) <= max_num:
        return [coord], [[a] if a is not None else None for a in attrs], \
               [coord_ref] if coord_ref is not None else [], \
               [[a] if a is not None else None for a in attrs_ref]

    dim_index = np.argmax(np.var(coord, 0)).item()
    split_point = len(coord) // 2
    split_value = torch.from_numpy(coord[:, dim_index]).kthvalue(split_point).values.numpy()
    mask = coord[:, dim_index] <= split_value
    if coord_ref is not None:
        mask_ref = coord_ref[:, dim_index] <= split_value
    else:
        mask_ref = None

    if split_point <= max_num:
        return [coord[mask], coord[~mask]],\
               [[a[mask], a[~mask]] if a is not None else None for a in attrs], \
               [coord_ref[mask_ref], coord_ref[~mask_ref]] if coord_ref is not None else [], \
               [[a[mask_ref], a[~mask_ref]] if a is not None else None for a in attrs_ref]
    else:
        left_partitions, left_extra_partitions, left_ref_partitions, left_extra_ref_partitions = _kd_tree_partition(
            coord[mask], max_num,
            [a[mask] if a is not None else None for a in attrs],
            coord_ref[mask_ref] if coord_ref is not None else None,
            [a[mask_ref] if a is not None else None for a in attrs_ref]
        )
        mask = np.logical_not(mask)
        mask_ref = np.logical_not(mask_ref)
        right_partitions, right_extra_partitions, right_ref_partitions, right_extra_ref_partitions = _kd_tree_partition(
            coord[mask], max_num,
            [a[mask] if a is not None else None for a in attrs],
            coord_ref[mask_ref] if coord_ref is not None else None,
            [a[mask_ref] if a is not None else None for a in attrs_ref]
        )
        left_partitions.extend(right_partitions)
        left_ref_partitions.extend(right_ref_partitions)
        for idx, p in enumerate(right_extra_partitions):
            if left_extra_partitions[idx] is not None:
                left_extra_partitions[idx].extend(p)
        for idx, p in enumerate(right_extra_ref_partitions):
            if left_extra_ref_partitions[idx] is not None:
                left_extra_ref_partitions[idx].extend(p)

        return left_partitions, left_extra_partitions, left_ref_partitions, left_extra_ref_partitions


def kd_tree_partition_randomly(
        coord: np.ndarray, target_num: int, attrs: Tuple[Optional[np.ndarray], ...] = (),
        choice_fn: Callable[[np.ndarray], int] = lambda d: np.argmax(np.var(d, 0)).item(),
        coord_ref: Optional[np.ndarray] = None, attrs_ref: Tuple[Optional[np.ndarray], ...] = (),
        cur_target_num_scaler: float = 0.5):
    points_num = len(coord)
    if points_num <= target_num:
        ret = [coord]
        if len(attrs) != 0:
            ret.append(attrs)
        if coord_ref is not None:
            ret.append(coord_ref)
        if len(attrs_ref) != 0:
            ret.append(attrs_ref)
        return tuple(ret) if len(ret) > 1 else ret[0]

    dim_index = choice_fn(coord)
    cur_target_num = round(points_num * cur_target_num_scaler)
    if cur_target_num < target_num:
        cur_target_num = target_num

    start_point = np.random.randint(points_num - cur_target_num + 1)
    end_points = start_point + cur_target_num - 1
    start_value = torch.from_numpy(coord[:, dim_index]).kthvalue(start_point + 1).values.numpy()
    end_value = torch.from_numpy(coord[:, dim_index]).kthvalue(end_points + 1).values.numpy()
    mask = np.logical_and(coord[:, dim_index] >= start_value, coord[:, dim_index] <= end_value)

    coord = coord[mask]
    attrs = tuple(a[mask] if a is not None else None for a in attrs)

    if coord_ref is not None:
        mask = np.logical_and(coord_ref[:, dim_index] >= start_value, coord_ref[:, dim_index] <= end_value)
        coord_ref = coord_ref[mask]
        attrs_ref = tuple(a[mask] if a is not None else None for a in attrs_ref)

    if cur_target_num <= target_num:
        ret = [coord]
        if len(attrs) != 0:
            ret.append(attrs)
        if coord_ref is not None:
            ret.append(coord_ref)
        if len(attrs_ref) != 0:
            ret.append(attrs_ref)
        return tuple(ret) if len(ret) > 1 else ret[0]
    return kd_tree_partition_randomly(
        coord, target_num, attrs, choice_fn, coord_ref, attrs_ref
    )


def write_ply_file(
        xyz: Union[torch.Tensor, np.ndarray],
        file_path: str,
        xyz_dtype: str = '<f4',
        rgb: Union[torch.Tensor, np.ndarray] = None,
        write_ascii: bool = False,
        make_dirs: bool = False,
        estimate_normals: bool = False,
        normals: Union[torch.Tensor, np.ndarray] = None) -> None:
    if make_dirs:
        os.makedirs(osp.dirname(file_path), exist_ok=True)
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    if isinstance(normals, torch.Tensor):
        normals = normals.cpu().numpy()
    assert xyz.shape[1] == 3
    xyz = xyz.astype(xyz_dtype, copy=False)
    rgb_dtype = np.uint8
    if rgb is not None:
        assert rgb.shape[1] == 3 and rgb.shape[0] == xyz.shape[0]
        assert rgb.dtype in (np.float32, rgb_dtype)
        rgb = rgb.astype(rgb_dtype, copy=False)

    el_with_properties_dtype = [('x', xyz_dtype), ('y', xyz_dtype), ('z', xyz_dtype)]
    if estimate_normals or normals is not None:
        el_with_properties_dtype.extend([('nx', np.float32), ('ny', np.float32), ('nz', np.float32)])
    if rgb is not None:
        el_with_properties_dtype.extend([('red', rgb_dtype), ('green', rgb_dtype), ('blue', rgb_dtype)])

    el_with_properties = np.empty(len(xyz), dtype=el_with_properties_dtype)
    el_with_properties['x'] = xyz[:, 0]
    el_with_properties['y'] = xyz[:, 1]
    el_with_properties['z'] = xyz[:, 2]
    if estimate_normals:
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        o3d_pc.estimate_normals()
        normals = np.asarray(o3d_pc.normals)
    if estimate_normals or normals is not None:
        el_with_properties['nx'] = normals[:, 0]
        el_with_properties['ny'] = normals[:, 1]
        el_with_properties['nz'] = normals[:, 2]
    if rgb is not None:
        el_with_properties['red'] = rgb[:, 0]
        el_with_properties['green'] = rgb[:, 1]
        el_with_properties['blue'] = rgb[:, 2]
    el = PlyElement.describe(el_with_properties, 'vertex')
    PlyData([el], text=write_ascii).write(file_path)


def read_xyz_from_ply_file(file_path: str):
    ply_data = PlyData.read(file_path)
    xyz = np.stack([
        ply_data.elements[0].data['x'],
        ply_data.elements[0].data['y'],
        ply_data.elements[0].data['z'],
    ])
    return xyz


def o3d_coords_sampled_from_triangle_mesh(
        triangle_mesh_path: str, points_num: int,
        rotation_matrix: np.ndarray = None,
        sample_method: str = 'uniform'
) -> np.ndarray:
    mesh_object = o3d.io.read_triangle_mesh(triangle_mesh_path)
    if rotation_matrix is not None:
        mesh_object.rotate(rotation_matrix)
    if sample_method == 'poisson_disk':
        point_cloud = mesh_object.sample_points_poisson_disk(points_num)
    elif sample_method == 'uniform':
        point_cloud = mesh_object.sample_points_uniformly(points_num)
    else:
        raise NotImplementedError
    return np.asarray(point_cloud.points)


def normalize_coords(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coord_max = xyz.max(axis=0, keepdims=True)
    coord_min = xyz.min(axis=0, keepdims=True)
    scale = (coord_max - coord_min).max()
    xyz -= coord_min
    np.divide(xyz, scale, out=xyz)
    return coord_min, scale
