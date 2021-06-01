import os
import pathlib
import hashlib
from tqdm import tqdm
import open3d as o3d
import numpy as np
import torch
import torch.utils.data
from scipy.spatial.transform import Rotation as R
try:
    import MinkowskiEngine as ME
except ImportError: pass

from lib.datasets.modelnet.dataset_config import DatasetConfig
from lib.data_utils import OFFIO, resample_mesh_by_faces


class ModelNetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig, is_training, logger, allow_cache=True):
        super(ModelNetDataset, self).__init__()
        # define files list path and cache path
        filelist_abs_path = os.path.join(cfg.root,
                                         cfg.train_filelist_path if is_training else cfg.test_filelist_path)
        self.cache_root = os.path.join(cfg.root, 'cache',
                                       hashlib.new('md5', cfg.to_yaml().encode('utf-8')).hexdigest())

        # generate files list
        if not os.path.exists(filelist_abs_path):
            logger.info('no filelist is given. Trying to generate...')
            if is_training:
                file_list = pathlib.Path(cfg.root).glob('*/train/*.off')
            else:
                file_list = pathlib.Path(cfg.root).glob('*/test/*.off')
            with open(filelist_abs_path, 'w') as f:
                f.write('\n'.join([str(_.relative_to(cfg.root)) for _ in file_list]))

        # load files list
        logger.info(f'using filelist: "{filelist_abs_path}"')
        with open(filelist_abs_path) as f:
            self.file_list = [os.path.join(cfg.root, f.readline().strip())]
            self.data_file_format = os.path.splitext(self.file_list[0])[1]
            try:
                assert self.data_file_format in ['.off', '.txt']
                for line in f:
                    line = line.strip()
                    assert os.path.splitext(line)[1] == self.data_file_format
                    self.file_list.append(os.path.join(cfg.root, line))
                if is_training: assert len(self.file_list) == 9843
                else: assert len(self.file_list) == 2468
            except AssertionError as e:
                logger.info('wrong number or format of files.')
                raise e

        # check existence of cache
        if self.data_file_format == '.off' and \
                os.path.isfile(os.path.join(self.cache_root,
                                            'train_all_cached' if is_training else 'test_all_cached')):
            logger.info(f'using cache in {self.cache_root}')
            self.file_list = [_.replace(cfg.root, self.cache_root, 1).replace('.off', '.pt')
                              for _ in self.file_list]
            self.use_cache = True
            self.gen_cache = False
        elif self.data_file_format == '.off':
            os.makedirs(self.cache_root, exist_ok=True)
            # log configuration
            with open(os.path.join(self.cache_root, 'dataset_config.yaml'), 'w') as f:
                f.write(cfg.to_yaml())
            self.use_cache = False
            self.gen_cache = True if allow_cache else False
            if self.gen_cache is True:
                logger.info(f'start caching in {self.cache_root}')
        else:
            self.use_cache = False
            self.gen_cache = False

        # load classes indices
        if cfg.with_classes:
            with open(os.path.join(cfg.root, cfg.classes_names)) as f:
                classes_names = f.readlines()
            self.classes_idx = {l.strip(): cls_idx for cls_idx, l in enumerate(classes_names)}

        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load
        file_path = self.file_list[index]

        # use cache
        if self.use_cache is True:
            if self.cfg.random_rotation: raise NotImplementedError
            return torch.load(file_path)

        # TODO: normals & augmentation

        # for modelnet40_normal_resampled
        if file_path.endswith('.txt'):
            point_cloud = np.loadtxt(file_path, dtype=np.float32, delimiter=',')
            # sample
            assert point_cloud.shape[0] >= self.cfg.input_points_num
            if point_cloud.shape[0] > self.cfg.input_points_num:
                if self.cfg.sample_method == 'uniform':
                    uniform_choice = np.random.choice(point_cloud.shape[0], self.cfg.input_points_num, replace=False)
                    point_cloud = point_cloud[uniform_choice]
                else:
                    raise NotImplementedError

        # for original modelnet dataset
        elif file_path.endswith('.off'):
            if self.cfg.with_normal_channel: raise NotImplementedError

            # mesh -> points
            mesh_object = o3d.io.read_triangle_mesh(file_path)
            vertices = np.asarray(mesh_object.vertices)

            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            vertices = (vertices - vmin) / (vmax - vmin).max()
            mesh_object.vertices = o3d.utility.Vector3dVector(vertices)

            # sample points from mesh
            if self.cfg.mesh_sample_point_method == 'barycentric':
                point_cloud = resample_mesh_by_faces(
                    mesh_object,
                    density=self.cfg.input_points_num / len(mesh_object.triangles))
            elif self.cfg.mesh_sample_point_method == 'poisson_disk':
                point_cloud = np.asarray(mesh_object.sample_points_poisson_disk(self.cfg.input_points_num).points)
            elif self.cfg.mesh_sample_point_method == 'uniform':
                point_cloud = np.asarray(mesh_object.sample_points_uniformly(self.cfg.input_points_num).points)
            else:
                raise NotImplementedError
            point_cloud = point_cloud.astype(np.float32)

            # # mesh -> voxel points
            # # could produce strange result if surface is not closed
            # vertices, faces = OFFIO.load_by_np(file_path)
            #
            # vmax = vertices.max(0, keepdims=True)
            # vmin = vertices.min(0, keepdims=True)
            # vertices = (vertices - vmin) * (self.cfg.resolution / (vmax - vmin).max())
            # mesh_object = pyvista.PolyData(vertices, faces)
            #
            # point_cloud = pyvista.voxelize(mesh_object, density=1, check_surface=False)
            # point_cloud = np.asarray(point_cloud.points.astype(np.int32))
            # voxelized_flag = True

        else:
            raise NotImplementedError

        # xyz
        xyz = point_cloud[:, :3]

        # normals
        if self.cfg.with_normal_channel:
            normals = point_cloud[:, 3:]
        else:
            normals = None

        # random rotation
        if not self.gen_cache and self.cfg.random_rotation:
            if self.cfg.with_normal_channel: raise NotImplementedError
            xyz = R.random().apply(xyz).astype(np.float32)

        # quantize  points: ndarray -> voxel points: torch.Tensor
        if self.cfg.resolution != 0:
            assert self.cfg.resolution > 1
            if self.data_file_format == '.txt':
                # coordinates of modelnet40_normal_resampled are in [-1, 1]
                xyz *= (self.cfg.resolution // 2)
            else:
                xyz *= self.cfg.resolution
            if self.cfg.with_normal_channel:
                xyz, normals = ME.utils.sparse_quantize(xyz, normals)
            else:
                xyz = ME.utils.sparse_quantize(xyz)

        # classes
        if self.cfg.with_classes:
            cls_idx = self.classes_idx[os.path.split(self.file_list[index])[1].rsplit('_', 1)[0]]
        else:
            cls_idx = None

        # cache and return
        return_obj = {'xyz': xyz,
                      'normals': normals,
                      'class_index': cls_idx,
                      'file_path': file_path if self.cfg.with_file_path else None}

        if self.gen_cache is True:
            cache_file_path = file_path.replace(self.cfg.root, self.cache_root, 1).replace('.off', '.pt')
            os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
            if os.path.exists(cache_file_path):
                raise FileExistsError
            torch.save(return_obj, cache_file_path)

        return return_obj

    def collate_fn(self, batch):
        assert isinstance(batch, list)

        has_normals = self.cfg.with_normal_channel
        has_class_index = self.cfg.with_classes
        has_file_path = self.cfg.with_file_path

        xyz_list = []
        normals_list = [] if has_normals else None
        class_index_list = [] if has_class_index else None
        file_path_list = [] if has_file_path else None

        for sample in batch:
            if isinstance(sample['xyz'], torch.Tensor):
                xyz_list.append(sample['xyz'])
            else:
                xyz_list.append(torch.from_numpy(sample['xyz']))
            if has_normals:
                if isinstance(sample['normals'], torch.Tensor):
                    normals_list.append(sample['normals'])
                else:
                    normals_list.append(torch.from_numpy(sample['normals']))
            if has_class_index:
                class_index_list.append(sample['class_index'])
            if has_file_path:
                file_path_list.append(sample['file_path'])

        return_obj = []

        if self.cfg.resolution == 0:
            batch_xyz = torch.stack(xyz_list, dim=0)
            return_obj.append(batch_xyz)
            if has_normals:
                batch_normals = torch.stack(normals_list, dim=0)
                return_obj.append(batch_normals)

        else:
            if has_normals:
                batch_xyz, batch_normals = ME.utils.sparse_collate(xyz_list, normals_list)
                return_obj.extend((batch_xyz, batch_normals))
            else:
                batch_xyz = ME.utils.batched_coordinates(xyz_list)
                return_obj.append(batch_xyz)

        if has_class_index:
            return_obj.append(torch.tensor(class_index_list))

        if has_file_path:
            return_obj.append(file_path_list)

        if len(return_obj) == 1:
            return_obj = return_obj[0]
        else:
            return_obj = tuple(return_obj)
        return return_obj


if __name__ == '__main__':
    config = DatasetConfig()
    config.input_points_num = 200000
    config.with_classes = False
    config.with_normal_channel = False
    config.with_file_path = True
    config.resolution = 128
    config.root = 'datasets/modelnet40_manually_aligned'

    from loguru import logger
    dataset = ModelNetDataset(config, True, logger, allow_cache=False)

    dataloader = torch.utils.data.DataLoader(dataset, 16, shuffle=False, collate_fn=dataset.collate_fn)
    dataloader = iter(dataloader)
    sample = next(dataloader)

    from main_debug import plt_draw, plt_batch_sparse_coord
    if config.with_classes or config.with_normal_channel:
        sample_coords = sample[0]
    else:
        sample_coords = sample
    if config.resolution == 0:
        plt_draw(sample_coords[0])
        plt_draw(sample_coords[1])
    else:
        plt_batch_sparse_coord(sample_coords, 0, True)
        plt_batch_sparse_coord(sample_coords, 1, True)
    print('Done')
