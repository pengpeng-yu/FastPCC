import math
import os
from collections import defaultdict
from typing import Tuple, List, Optional, Union

import numpy as np
import cv2
import open3d as o3d
import torch
try:
    import MinkowskiEngine as ME
except ImportError: pass


class SampleData:
    def __init__(self):
        self.results_dir = None

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
    def __init__(self, xyz: torch.Tensor,
                 colors: torch.Tensor = None,
                 normals: torch.Tensor = None,
                 class_idx: Union[int, torch.Tensor] = None,
                 ori_resolution: Union[int, List[int]] = None,
                 resolution: Union[int, List[int]] = None,
                 file_path: Union[str, List[str]] = None):
        super(PCData, self).__init__()
        self.xyz = xyz
        self.colors = colors
        self.normals = normals
        self.ori_resolution = ori_resolution
        self.resolution = resolution
        self.file_path = file_path
        self.class_idx = class_idx


def pc_data_collate_fn(data_list: List[PCData],
                       sparse_collate: bool) -> PCData:
    data_dict = defaultdict(list)
    for data in data_list:
        for key, value in data.__dict__.items():
            if value is not None:
                data_dict[key].append(value)

    batched_data_dict = {}
    for key, value in data_dict.items():
        if key in ('xyz', 'colors', 'normals'):
            if not sparse_collate:
                batched_data_dict[key] = torch.stack(value, dim=0)
            else:
                if key == 'xyz':
                    batched_data_dict[key] = ME.utils.batched_coordinates(value)
                else:
                    batched_data_dict[key] = torch.cat(value, dim=0)

        elif key in ('class_idx',):
            batched_data_dict[key] = torch.tensor(value)

        else:
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


class OFFIO:
    @classmethod
    def load_by_np(cls, file_path):
        with open(file_path) as f:
            assert f.readlines(1)[0].strip() == 'OFF'
            vertices_num, faces_num, edges_num = [int(_) for _ in f.readlines(1)[0].split()]
            if edges_num != 0: raise NotImplementedError
            vertices = np.loadtxt(f, dtype=np.float32, max_rows=vertices_num)
            faces = np.loadtxt(f, dtype=np.int32, max_rows=faces_num)
        return vertices, faces

    @classmethod
    def save(cls, file_path, vertices: np.ndarray, faces: np.ndarray):
        if os.path.exists(file_path):
            raise FileExistsError
        with open(file_path, 'w') as f:
            f.write('OFF\n')
            f.write(f'{vertices.shape[0]} {faces.shape[0]} 0\n')
            np.savetxt(f, vertices, fmt='%d' if vertices.dtype == np.int32 else '%.18e')
            np.savetxt(f, faces, fmt='%d')
        return True


def o3d_coords_from_triangle_mesh(triangle_mesh_path: str, points_num: int,
                                  sample_method: str = 'uniform') -> np.ndarray:
    mesh_object = o3d.io.read_triangle_mesh(triangle_mesh_path)

    if sample_method == 'barycentric':
        point_cloud = resample_mesh_by_faces(
            mesh_object,
            density=points_num / len(mesh_object.triangles))
    elif sample_method == 'poisson_disk':
        point_cloud = np.asarray(mesh_object.sample_points_poisson_disk(points_num).points)
    elif sample_method == 'uniform':
        point_cloud = np.asarray(mesh_object.sample_points_uniformly(points_num).points)
    else:
        raise NotImplementedError
    point_cloud = point_cloud.astype(np.float32)
    return point_cloud


def normalize_coords(xyz: np.ndarray):
    coord_max = xyz.max(axis=0, keepdims=True)
    coord_min = xyz.min(axis=0, keepdims=True)
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