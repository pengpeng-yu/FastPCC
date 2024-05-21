import os
import os.path as osp
import pathlib
from tqdm import tqdm

shapenet_root = pathlib.Path('datasets/ShapeNet/ShapeNetCore.v2')


def rename():
    mtl_paths = shapenet_root.glob('*/*/models/model_normalized.mtl')
    for mtl_path in tqdm(mtl_paths):
        os.rename(mtl_path, str(mtl_path) + '.bak')
        with open(mtl_path, 'w') as f:
            pass


def recover():
    mtl_paths = shapenet_root.glob('*/*/models/model_normalized.mtl.bak')
    for mtl_path in tqdm(mtl_paths):
        assert osp.isfile(mtl_path)
        ori_mtl_path = str(mtl_path)[:-4]
        os.remove(ori_mtl_path)
        os.rename(mtl_path, str(mtl_path)[:-4])


if __name__ == '__main__':
    rename()
    print('Done')
