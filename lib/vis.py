from typing import Optional, Union, Tuple
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
import torch
import open3d as o3d
try:
    import MinkowskiEngine as ME
except ImportError: pass


def open3d_draw_xyz(pc: Union[torch.Tensor, np.ndarray]) -> None:
    if isinstance(pc, torch.Tensor):
        xyz = pc.detach().cpu().numpy()[..., :3]
    elif isinstance(pc, np.ndarray):
        xyz = pc[:, :3]
    else:
        raise NotImplementedError
    if len(xyz.shape) == 2:
        xyz = [xyz]
    elif len(xyz.shape) == 3:
        xyz = [*xyz]
    else:
        raise NotImplementedError
    o3d_pc = [o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(sub_xyz[:, :3]))
        for sub_xyz in xyz]
    o3d.visualization.draw_geometries(o3d_pc)
    print('Done')


def plt_draw_xyz_with_degree(fig: matplotlib.figure.Figure,
                             pos_arg: int,
                             xyz: np.ndarray,
                             degree: Tuple[Optional[int], Optional[int]],
                             if_draw_voxel: bool):
    ax = fig.add_subplot(pos_arg, projection='3d')
    if not if_draw_voxel:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        pos_lim = np.max(xyz)
        neg_lim = np.min(xyz)
        ax.set_xlim(neg_lim, pos_lim)
        ax.set_ylim(neg_lim, pos_lim)
        ax.set_zlim(neg_lim, pos_lim)
    else:
        ax.voxels(xyz)
        pos_lim = max(xyz.shape)
        neg_lim = 0
        ax.set_xlim(neg_lim, pos_lim)
        ax.set_ylim(neg_lim, pos_lim)
        ax.set_zlim(neg_lim, pos_lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(degree[0], degree[1])


def plt_draw_xyz(xyz: Union[torch.Tensor, np.ndarray],
                 if_draw_voxel: bool = False,
                 figsize: Tuple[int, int] = (8, 8)):
    fig = plt.figure(figsize=figsize)
    if if_draw_voxel:
        if isinstance(xyz, np.ndarray):
            xyz = torch.from_numpy(xyz)
        xyz = ME.utils.sparse_collate([xyz], [torch.full((xyz.shape[0], 1),
                                                         True, dtype=torch.bool)])
        xyz = ME.SparseTensor(xyz[1], xyz[0])
        xyz = \
            xyz.dense(min_coordinate=xyz.C.min().expand(3))[0].squeeze(1).squeeze(0).numpy()
    else:
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.detach().cpu().numpy()
    plt_draw_xyz_with_degree(fig, 221, xyz, (None, None), if_draw_voxel)
    plt_draw_xyz_with_degree(fig, 222, xyz, (0, 0), if_draw_voxel)
    plt_draw_xyz_with_degree(fig, 223, xyz, (0, 90), if_draw_voxel)
    plt_draw_xyz_with_degree(fig, 224, xyz, (90, 0), if_draw_voxel)
    plt.show()
    print('Done')


def plt_batched_sparse_xyz(batched_xyz: torch.Tensor, batch_idx: int, if_draw_voxel: bool = False):
    assert batched_xyz.shape[1] == 4
    xyz = batched_xyz[batched_xyz[:, 0] == batch_idx, 1:]
    plt_draw_xyz(xyz, if_draw_voxel)


if __name__ == '__main__':
    pass
