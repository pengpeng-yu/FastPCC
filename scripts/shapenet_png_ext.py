import os
from tqdm import tqdm
import pathlib

shapenet_root = pathlib.Path('datasets/ShapeNet/ShapeNetCore.v2')

mtl_paths = shapenet_root.glob('*/*/models/model_normalized.mtl')

for mtl_path in tqdm(mtl_paths):
    os.renames(mtl_path, str(mtl_path) + '.bak')
    with open(mtl_path, 'w') as f:
        pass
