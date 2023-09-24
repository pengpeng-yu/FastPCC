import math
import os
from collections import defaultdict
from typing import Tuple, List, Optional, Union, Dict, Callable

import numpy as np
try:
    import cv2
except ImportError: cv2 = None
from plyfile import PlyData, PlyElement
try:
    import open3d as o3d
except ImportError: o3d = None
import torch
try:
    import MinkowskiEngine as ME
except ImportError: ME = None


class SampleData:
    def __init__(self):
        self.results_dir: Optional[str] = None
        self.training_step: Optional[int] = None

    def to(self, device, non_blocking=False):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.to(device, non_blocking=non_blocking)


class IMData(SampleData):
    def __init__(self, im, file_path, valid_range=None):
        super(IMData, self).__init__()
        self.im = im
        self.file_path = file_path
        self.valid_range = valid_range


def im_data_collate_fn(data_list: List[IMData],
                       target_shapes: Union[Tuple[int, ...], List[int]],
                       resize_strategy: str,
                       channel_last_to_channel_first: bool) -> IMData:
    shape_idx = np.random.randint(0, len(target_shapes) // 2) * 2
    target_shape = (target_shapes[shape_idx],
                    target_shapes[shape_idx + 1])

    data_dict = defaultdict(list)
    for data in data_list:
        for key, value in data.__dict__.items():
            if key == 'im':
                if resize_strategy == 'Expand':
                    im, valid_range = im_resize_with_crop(value, target_shape)
                elif resize_strategy == 'Shrink':
                    im, valid_range = im_resize_with_pad(value, target_shape)
                elif resize_strategy == 'Retain':
                    im, valid_range = im_pad(value, target_shape=target_shape)
                elif resize_strategy == 'Adapt':
                    im, valid_range = im_pad(value, base_length=target_shape)
                else:
                    raise NotImplementedError
                data_dict[key].append(im)
                data_dict['valid_range'].append(valid_range)
            elif value is not None:
                data_dict[key].append(value)

    batched_data_dict = {}
    for key, value in data_dict.items():
        if key == 'im':
            if channel_last_to_channel_first is True:
                batched_data_dict[key] = \
                    torch.from_numpy(np.stack(value)).permute(0, 3, 1, 2).contiguous()
            else:
                batched_data_dict[key] = torch.from_numpy(np.stack(value))
        else:
            batched_data_dict[key] = value
    return IMData(**batched_data_dict)


class PCData(SampleData):
    tensor_to_tensor_items = ('color', 'normal')
    list_to_tensor_items = ('class_idx',)

    def __init__(self,
                 xyz: Union[torch.Tensor, List[torch.Tensor]],
                 color: Union[torch.Tensor, List[torch.Tensor]] = None,
                 normal: Union[torch.Tensor, List[torch.Tensor]] = None,
                 class_idx: Union[int, torch.Tensor] = None,
                 ori_resolution: Union[int, List[int]] = None,
                 resolution: Union[int, List[int]] = None,
                 file_path: Union[str, List[str]] = None,
                 batch_size: int = 0):
        """
        xyz is supposed to be a torch.float32 tensor in cpu
        when initialized by a single sample.
        """
        super(PCData, self).__init__()
        self.xyz = xyz
        self.color = color
        self.normal = normal
        self.class_idx = class_idx
        self.ori_resolution = ori_resolution
        self.resolution = resolution
        self.file_path = file_path
        self.batch_size = batch_size

    def to(self, device, non_blocking=False):
        super(PCData, self).to(device, non_blocking)
        for key in ('xyz', *self.tensor_to_tensor_items):
            value = self.__dict__[key]
            if isinstance(value, List):
                # Ignore the first value of partitions lists.
                for idx, v in enumerate(value[1:]):
                    assert isinstance(v, torch.Tensor)
                    value[idx + 1] = v.to(device, non_blocking=non_blocking)


def pc_data_collate_fn(data_list: List[PCData],
                       sparse_collate: bool,
                       kd_tree_partition_max_points_num: int = 0) -> PCData:
    """
    If sparse_collate is True, PCData.xyz will be batched
    using ME.utils.batched_coordinates, which returns a torch.int32 tensor.
    """
    if kd_tree_partition_max_points_num > 0:
        use_kd_tree_partition = True
        assert len(data_list) == 1, 'Only supports kd-tree partition when batch size == 1.'
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
                if not sparse_collate:
                    batched_data_dict[key] = torch.stack(
                        [_.to(torch.float32) for _ in value], dim=0
                    )
                else:
                    batched_data_dict[key] = ME.utils.batched_coordinates(
                        value, dtype=torch.int32
                    )
            elif key in PCData.tensor_to_tensor_items:
                if not sparse_collate:
                    batched_data_dict[key] = torch.stack(value, dim=0)
                else:
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
                if not sparse_collate:
                    batched_data_dict[key] = [_.to(torch.float32)[None] if idx != 0 else _.to(torch.float32)
                                              for idx, _ in enumerate(batched_data_dict[key])]
                else:
                    batched_data_dict[key] = [ME.utils.batched_coordinates([_], dtype=torch.int32)
                                              if idx != 0 else _.to(torch.int32)
                                              for idx, _ in enumerate(batched_data_dict[key])]
            elif key in PCData.tensor_to_tensor_items:
                # Add batch dimension.
                if not sparse_collate:
                    batched_data_dict[key] = [_[None] for _ in batched_data_dict[key][1:]]
            elif key in PCData.list_to_tensor_items:
                batched_data_dict[key] = torch.tensor(value)
            elif key != 'batch_size':
                batched_data_dict[key] = value

    return PCData(**batched_data_dict)


def im_resize_with_crop(
        im: np.ndarray,
        target_shape: Union[Tuple[int, int], List[int]]
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(target_shape) == 2
    shape_factor = (target_shape[0] / im.shape[0],
                    target_shape[1] / im.shape[1])
    shape_scaler = max(shape_factor)
    im: np.ndarray = cv2.resize(im, (0, 0), fx=shape_scaler, fy=shape_scaler)
    boundary = np.array([im.shape[0] - target_shape[0],
                         im.shape[1] - target_shape[1]])
    origin: np.ndarray = np.random.randint(0, boundary + 1)

    im = im[origin[0]: origin[0] + target_shape[0],
            origin[1]: origin[1] + target_shape[1]]
    valid_range = np.array([[0, im.shape[0]],
                            [0, im.shape[1]]], dtype=np.int)
    return im, valid_range


def im_resize_with_pad(
        im: np.ndarray,
        target_shape: Union[Tuple[int, int], List[int]]
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(target_shape) == 2
    shape_factor = (target_shape[0] / im.shape[0],
                    target_shape[1] / im.shape[1])
    shape_scaler = min(shape_factor)
    im = cv2.resize(im, (0, 0), fx=shape_scaler, fy=shape_scaler)
    holder = np.zeros_like(im, shape=(*target_shape, 3))
    boundary = np.array([target_shape[0] - im.shape[0],
                         target_shape[1] - im.shape[1]])
    origin = np.random.randint(0, boundary + 1)

    valid_range = np.array(
        [[origin[0], origin[0] + im.shape[0]],
         [origin[1], origin[1] + im.shape[1]]],
        dtype=np.int
    )
    holder[valid_range[0][0]: valid_range[0][1],
           valid_range[1][0]: valid_range[1][1]] = im
    return holder, valid_range


def im_pad(
        im: np.ndarray,
        target_shape: Union[Tuple[int, int], List[int]] = None,
        base_length: Union[Tuple[int, int], List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if target_shape is None:
        assert len(base_length) == 2
        target_shape = (math.ceil(im.shape[0] / base_length[0]) * base_length[0],
                        math.ceil(im.shape[1] / base_length[1]) * base_length[1])
    else:
        assert len(target_shape) == 2
        assert target_shape[0] >= im.shape[0] and \
               target_shape[1] >= im.shape[1]

    holder = np.zeros_like(im, shape=(*target_shape, 3))
    holder[: im.shape[0], : im.shape[1]] = im
    valid_range = np.array([[0, im.shape[0]],
                            [0, im.shape[1]]], dtype=np.int)
    return holder, valid_range


class KDNode:
    def __init__(self, point: np.ndarray):
        super(KDNode, self).__init__()
        self.point = point
        self.left: Union[np.ndarray, KDNode, None] = None
        self.right: Union[np.ndarray, KDNode, None] = None


def create_kd_tree(data: np.ndarray, max_num: int = 1) -> Union[KDNode, np.ndarray]:
    if len(data) <= max_num:
        return data

    dim_index = np.argmax(np.var(data, dim=0)).item()
    split_point = len(data) // 2
    data_sorted = data[np.argpartition(data[:, dim_index], split_point)]

    kd_node = KDNode(data_sorted[split_point])
    kd_node.left = create_kd_tree(data_sorted[:split_point], max_num)
    kd_node.right = create_kd_tree(data_sorted[split_point:], max_num)
    return kd_node


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


def kd_tree_partition_randomly_old(
        data: np.ndarray, target_num: int, extras: Tuple[Optional[np.ndarray], ...] = (),
        choice_fn: Callable[[np.ndarray], int] = lambda d: np.argmax(np.var(d, 0)).item()
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[Optional[np.ndarray], ...]]]:
    len_data = len(data)
    if len_data <= target_num:
        if len(extras) != 0:
            return data, extras
        else:
            return data

    dim_index = choice_fn(data)
    is_left = np.random.rand() < 0.5

    if len_data // 2 >= target_num:
        split_point = len_data // 2
        arg_sorted = np.argpartition(data[:, dim_index], split_point)
        if is_left:
            arg_sorted = arg_sorted[:split_point]
        else:
            arg_sorted = arg_sorted[split_point:]
    else:
        if is_left:
            arg_sorted = np.argpartition(data[:, dim_index], target_num)[:target_num]
        else:
            arg_sorted = np.argpartition(data[:, dim_index], -target_num)[-target_num:]

    return kd_tree_partition_randomly_old(
        data[arg_sorted], target_num,
        tuple(extra[arg_sorted] if isinstance(extra, np.ndarray) else extra for extra in extras),
        choice_fn
    )


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
        estimate_normals: bool = False) -> None:
    if make_dirs:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()
    assert xyz.shape[1] == 3 and xyz.dtype in (np.int32, np.int64, np.float32, np.float64)
    xyz = xyz.astype(xyz_dtype)
    rgb_dtype = np.uint8
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    if rgb is not None:
        assert rgb.shape[1] == 3 and rgb.shape[0] == xyz.shape[0]
        assert rgb.dtype in (np.float32, rgb_dtype)
        rgb = rgb.astype(rgb_dtype)
    el_with_properties_dtype = [('x', xyz_dtype), ('y', xyz_dtype), ('z', xyz_dtype)]
    if estimate_normals:
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


def if_ply_has_vertex_normal(file_path: str):
    has = False
    with open(file_path, 'rb') as f:
        while True:
            try:
                line = f.readline()
                if line.strip() == b'end_header': break
                elif line.rsplit(maxsplit=1)[-1] == b'nx':
                    has = True
                    break
            except Exception as e:
                print(file_path)
                raise e
    return has


def o3d_coords_sampled_from_triangle_mesh(
        triangle_mesh_path: str, points_num: int,
        rotation_matrix: np.ndarray = None,
        sample_method: str = 'uniform',
        with_color: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    mesh_object = o3d.io.read_triangle_mesh(triangle_mesh_path)
    if rotation_matrix is not None:
        mesh_object.rotate(rotation_matrix)
    if sample_method == 'barycentric':
        assert with_color is False
        coord = resample_mesh_by_faces(
            mesh_object,
            density=points_num / len(mesh_object.triangles))
        color = None
    else:
        if sample_method == 'poisson_disk':
            point_cloud = mesh_object.sample_points_poisson_disk(points_num)
        elif sample_method == 'uniform':
            point_cloud = mesh_object.sample_points_uniformly(points_num)
        else:
            raise NotImplementedError
        coord = np.asarray(point_cloud.points)
        color = np.asarray(point_cloud.colors)
    return coord, color


def normalize_coords(xyz: np.ndarray, random_crop: bool = False, random_crop_ratio: float = 0.5):
    coord_max = xyz.max(axis=0, keepdims=True)
    coord_min = xyz.min(axis=0, keepdims=True)
    if random_crop is True:
        assert 0 < random_crop_ratio < 1
        box_size = (coord_max - coord_min) * random_crop_ratio
        origin_range = coord_max - box_size - coord_min
        while True:
            coord_min = np.random.rand(3) * origin_range + coord_min
            coord_max = coord_min + box_size
            valid_mask = ((xyz >= coord_min) & (xyz <= coord_max)).sum(1) == 3
            if np.any(valid_mask):
                break
            else:
                print('Warning: bad crop')
        xyz = xyz[valid_mask]
    xyz = (xyz - coord_min) / (coord_max - coord_min).max()
    return xyz


def resample_mesh_by_faces(mesh_cad, density=1.0):
    """
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    param mesh_cad: low-polygon triangle mesh in o3d.geometry.TriangleMesh
    param density: density of the point cloud per unit area
    param return_numpy: return numpy format or open3d pointcloud format
    return resampled point cloud

    Reference :
      [1] Barycentric coordinate system
      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    """
    faces = np.array(mesh_cad.triangles).astype(int)
    vertices = np.array(mesh_cad.vertices)

    vec_cross = np.cross(
        vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
        vertices[faces[:, 1], :] - vertices[faces[:, 2], :],
    )
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))

    n_samples = (np.sum(face_areas) * density).astype(int)
    # face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Bug fix by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(density * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]

    P = (
        (1 - np.sqrt(r[:, 0:1])) * A
        + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B
        + np.sqrt(r[:, 0:1]) * r[:, 1:] * C
    )

    return P
