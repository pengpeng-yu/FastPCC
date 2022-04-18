import os
import pathlib
import imghdr
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
        assert os.path.isfile(mtl_path)
        ori_mtl_path = str(mtl_path)[:-4]
        os.remove(ori_mtl_path)
        os.rename(mtl_path, str(mtl_path)[:-4])


def fix_ext():
    img_paths = shapenet_root.glob('*/*/images/*.*')
    for img_path in tqdm(img_paths):
        img_format = imghdr.what(img_path)
        img_base_path, img_ext = os.path.splitext(img_path)
        img_ext = img_ext[1:]
        if img_ext == 'jpg': img_ext = 'jpeg'
        if not img_ext == img_format:
            img_new_path = f'{img_base_path}.{img_format}'
            print(f'{img_path} -> {img_new_path}')
            os.rename(img_path, img_new_path)


if __name__ == '__main__':
    fix_ext()
    print('Done')
