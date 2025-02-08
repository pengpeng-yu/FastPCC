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
    N = np.array([len(cs) for cs in coords])
    bcoords = torch.zeros((N.sum(), 4), dtype=torch.int32)
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
                self.__dict__[key] = value.to(device, non_blocking=non_blocking)

    def pin_memory(self):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.pin_memory()
        return self


class IMData(SampleData):
    def __init__(self, im, file_path, valid_range=None):
        super(IMData, self).__init__()
        self.im = im
        self.file_path = file_path
        self.valid_range = valid_range


class PCData(SampleData):
    tensor_to_tensor_items = ('color', 'normal')
    list_to_tensor_items = ('class_idx',)

    def __init__(self,
                 xyz: Union[torch.Tensor, List[torch.Tensor]],
                 color: Union[torch.Tensor, List[torch.Tensor]] = None,
                 normal: Union[torch.Tensor, List[torch.Tensor]] = None,
                 class_idx: Union[int, torch.Tensor] = None,
                 resolution: Union[float, List[float]] = None,
                 file_path: Union[str, List[str]] = None,
                 batch_size: int = 0,
                 points_num: List[int] = None,
                 org_xyz: Union[torch.Tensor, List[torch.Tensor]] = None,
                 inv_transform: Union[torch.Tensor, List[torch.Tensor]] = None):
        super(PCData, self).__init__()
        self.xyz = xyz
        self.color = color
        self.normal = normal
        self.class_idx = class_idx
        self.resolution = resolution
        self.file_path = file_path
        self.batch_size = batch_size
        self.points_num = points_num
        self.org_xyz = org_xyz
        self.inv_transform = inv_transform

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

    data_dict: Dict[str, List] = defaultdict(list)
    for data in data_list:
        for key, value in data.__dict__.items():
            if value is not None:
                data_dict[key].append(value)

    batched_data_dict = {'batch_size': len(data_list)}
    if not use_kd_tree_partition:
        for key, value in data_dict.items():
            if key == 'xyz':
                batched_data_dict[key], batched_data_dict['points_num'] = batched_coordinates(value)
            elif key in PCData.tensor_to_tensor_items:
                batched_data_dict[key] = torch.cat(value, dim=0)
            elif key in PCData.list_to_tensor_items:
                batched_data_dict[key] = torch.tensor(value)
            elif key != 'batch_size':
                batched_data_dict[key] = value

    else:
        # Use kd-tree partition.
        extras_dict = {item: data_dict[item][0]
                       for item in PCData.tensor_to_tensor_items if item in data_dict}
        if extras_dict == {}:
            # Retain original coordinates in the head of list.
            batched_data_dict['xyz'] = data_dict['xyz']
            batched_data_dict['xyz'].extend(
                kd_tree_partition(
                    data_dict['xyz'][0], kd_tree_partition_max_points_num,
                )
            )
        else:
            xyz_partitions, extras = kd_tree_partition(
                data_dict['xyz'][0], kd_tree_partition_max_points_num,
                extras=list(extras_dict.values())
            )
            batched_data_dict['xyz'] = data_dict['xyz']
            batched_data_dict['xyz'].extend(xyz_partitions)
            for idx, key in enumerate(extras_dict):
                batched_data_dict[key] = [extras_dict[key]]
                batched_data_dict[key].extend(extras[idx])
        for key, value in data_dict.items():
            if key == 'xyz':
                # Add batch dimension.
                # The first one is supposed to be the original coordinates.
                tmp_ls = [batched_data_dict[key][0]]
                for tmp_ in batched_data_dict[key][1:]:
                    tmp = torch.zeros((tmp_.shape[0], 4), dtype=torch.int32)
                    tmp[:, 1:] = tmp_
                    tmp_ls.append(tmp)
                batched_data_dict[key] = tmp_ls
            elif key in PCData.tensor_to_tensor_items:
                pass
            elif key in PCData.list_to_tensor_items:
                batched_data_dict[key] = torch.tensor(value)
            elif key != 'batch_size':
                batched_data_dict[key] = value

    return PCData(**batched_data_dict)


def kd_tree_partition(data: Union[np.ndarray, torch.Tensor], max_num: int,
                      extras: Union[List[np.ndarray], List[torch.Tensor]] = None)\
        -> Union[List[np.ndarray], List[torch.Tensor],
                 Tuple[List[np.ndarray], List[List[np.ndarray]]],
                 Tuple[List[torch.Tensor], List[List[torch.Tensor]]]]:
    is_torch_tensor = isinstance(data, torch.Tensor)
    if extras is None or extras == []:
        if is_torch_tensor:
            data = data.numpy()
        data_list = kd_tree_partition_base(data, max_num)
        if is_torch_tensor:
            data_list = [torch.from_numpy(_) for _ in data_list]
        return data_list
    else:
        if is_torch_tensor:
            data = data.numpy()
            extras = [_.numpy() for _ in extras]
        data_list, extras_lists = kd_tree_partition_extended(data, max_num, extras)
        if is_torch_tensor:
            data_list = [torch.from_numpy(_) for _ in data_list]
            extras_lists = [[torch.from_numpy(_) for _ in extras_list]
                            for extras_list in extras_lists]
        return data_list, extras_lists


def kd_tree_partition_base(data: np.ndarray, max_num: int) -> List[np.ndarray]:
    if len(data) <= max_num:
        return [data]

    dim_index = np.argmax(np.var(data, 0)).item()
    split_point = len(data) // 2
    split_value = torch.from_numpy(data[:, dim_index]).kthvalue(split_point).values.numpy()
    mask = data[:, dim_index] <= split_value

    if split_point <= max_num:
        return [data[mask], data[~mask]]
    else:
        left_partitions = kd_tree_partition_base(data[mask], max_num)
        right_partitions = kd_tree_partition_base(data[~mask], max_num)
        left_partitions.extend(right_partitions)

    return left_partitions


def kd_tree_partition_extended(data: np.ndarray, max_num: int, extras: List[np.ndarray]) \
        -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    if len(data) <= max_num:
        return [data], [[extra] for extra in extras]

    dim_index = np.argmax(np.var(data, 0)).item()
    split_point = len(data) // 2
    split_value = torch.from_numpy(data[:, dim_index]).kthvalue(split_point).values.numpy()
    mask = data[:, dim_index] <= split_value

    if split_point <= max_num:
        return [data[mask], data[~mask]], [[extra[mask], extra[~mask]] for extra in extras]
    else:
        left_partitions, left_extra_partitions = kd_tree_partition_extended(
            data[mask], max_num,
            [extra[mask] if extra is not None else extra for extra in extras]
        )
        mask = np.logical_not(mask)
        right_partitions, right_extra_partitions = kd_tree_partition_extended(
            data[mask], max_num,
            [extra[mask] if extra is not None else extra for extra in extras]
        )
        left_partitions.extend(right_partitions)
        for idx, p in enumerate(right_extra_partitions):
            left_extra_partitions[idx].extend(p)

        return left_partitions, left_extra_partitions


def kd_tree_partition_randomly(
        data: np.ndarray, target_num: int, extras: Tuple[Optional[np.ndarray], ...] = (),
        choice_fn: Callable[[np.ndarray], int] = lambda d: np.argmax(np.var(d, 0)).item(),
        cur_target_num_scaler: float = 0.5
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[Optional[np.ndarray], ...]]]:
    len_data = len(data)
    if len_data <= target_num:
        if len(extras) != 0:
            return data, extras
        else:
            return data

    dim_index = choice_fn(data)
    cur_target_num = round(len_data * cur_target_num_scaler)
    if cur_target_num < target_num:
        cur_target_num = target_num

    start_point = np.random.randint(len_data - cur_target_num + 1)
    end_points = start_point + cur_target_num - 1
    start_value = torch.from_numpy(data[:, dim_index]).kthvalue(start_point + 1).values.numpy()
    end_value = torch.from_numpy(data[:, dim_index]).kthvalue(end_points + 1).values.numpy()
    mask = np.logical_and(data[:, dim_index] >= start_value, data[:, dim_index] <= end_value)

    data = data[mask]
    extras = tuple(extra[mask] if isinstance(extra, np.ndarray) else extra for extra in extras)

    if cur_target_num <= target_num:
        if len(extras) != 0:
            return data, extras
        else:
            return data
    return kd_tree_partition_randomly(
        data, target_num, extras, choice_fn
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
