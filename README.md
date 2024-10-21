This project contains an implementation of our ICME 2023 paper "Sparse Representation based Deep Residual Geometry Compression Network for Large-scale Point Clouds" [1] and other improvements.


## Highlights
See our detailed experimental results on Intel Xeon Gold 5118 and NVIDIA 2080Ti at [OneDrive](https://mssysueducn-my.sharepoint.com/:u:/g/personal/yupp5_ms_sysu_edu_cn/EbzFDM93okNPmceKE5ZLzhgBZPJ1Cb4L-GeoP3stilFJxQ).


## Models
- `config/convolutional/lossy_coord_v2/baseline_r*.yaml`: Improved geometry lossy compression. 
- `config/convolutional/lossy_coord_lossy_color/baseline_r*.yaml`: Joint lossy compression. 
- `config/convolutional/lossy_coord/lossl_based*.yaml`: The configs of model in [1] (Deprecated).
- `config/convolutional/lossy_coord/baseline.yaml`: A reimplementation of PCGCv2.

Run these models using `python train.py/test.py [model yaml path] config_key_1=value_1 config_key_2=value_2 ...`. 
Please see detailed commands [below](#train--test).   


## Environment
- python >= 3.7
- loguru
- ninja
- open3d
- plyfile
- pytorch >= 1.7
- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) (only for knn_points in joint lossy compression. Conda installation suggested: conda install -c fvcore -c iopath -c conda-forge fvcore iopath; conda install pytorch3d -c pytorch3d)
- [minkowskiengine](https://github.com/NVIDIA/MinkowskiEngine?tab=readme-ov-file#installation) ≈ 0.5.4 (for compilation with CUDA 12.1/12.2, please refer to [this](https://github.com/daizhirui/MinkowskiEngine/tree/fix-for-cuda-12.2))


## Requirements
- [Examples of datasets folder and bin folder (with binary tools compiled on Ubuntu 16.04)](https://mssysueducn-my.sharepoint.com/:u:/g/personal/yupp5_ms_sysu_edu_cn/EWQeTYD3y8dDr1-lvq7YawQB-JoJ2_GQw2jSOgpS2i6xLw)
- [Test set of geometry compression](https://mssysueducn-my.sharepoint.com/:u:/g/personal/yupp5_ms_sysu_edu_cn/Ed9ljLgHJD9Ipd_Sd8nP9NABxJbywu1kT1pPo4vdNIZWjg)
- [Trained model weights](https://mssysueducn-my.sharepoint.com/:u:/g/personal/yupp5_ms_sysu_edu_cn/EUIl3GRmQL5KvKCwwipEOhkBzoNXfyDwMs_BHju7M70ayg)
- [ShapeNetCorev2 of OBJ format](https://huggingface.co/datasets/ShapeNet/ShapeNetCore/tree/main) (for training geometry compression)
- [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (for LiDAR geometry compression)
- [8iVFBv2](https://plenodb.jpeg.org/pc/8ilabs) & [Owlii](https://mpeg-pcc.org/index.php/pcc-content-database/owlii-dynamic-human-textured-mesh-sequence-dataset/) (for joint geometry and color compression)

The first link provides binaries of tmc3 and pc_error built on Ubuntu 16.04 with `-O3 -march=x86-64`.
Place them in the `bin` folder, and run `chmod u+x bin/tmc3 bin/pc_error` to make the files executable.
For detailed compilation instructions, please refer to [BUILD.md](BUILD.md).

The first link also provides the file lists of each dataset, which is **necessary** for data retrieval.
If your datasets are stored in directories other than the `datasets` folder, 
it is recommended to create soft links to point to these datasets.

For training on ShapeNet dataset, you need to run `python scripts/shapenet_mtls.py` to prevent Open3D from loading textures, 
which may cause errors. See [Known issues](#Known-issues).


## Train / Test
Training:
```shell
python train.py config/convolutional/lossy_coord_v2/baseline_r1.yaml \
  train.device='0' train.rundir_name='lossy_coord_v2/baseline_r1'
```
Test:
```shell
python test.py config/convolutional/lossy_coord_v2/baseline_r1.yaml \
  test.from_ckpt='weights/convolutional/lossy_coord_v2/baseline_r1.pt' \
  test.device='0'
```
DDP training: 
```shell
python -m torch.distributed.launch --nproc_per_node 2 \
  train.py config/convolutional/lossy_coord_v2/baseline_r1.yaml \
  train.device='0,1' train.rundir_name='lossy_coord_v2/baseline_r1'
```
Resume training from "runs/lossy_coord_v2/baseline_r1_bak0" to "runs/lossy_coord_v2/baseline_r1":
```shell
python train.py config/convolutional/lossy_coord_v2/baseline_r1.yaml \
  train.from_ckpt='runs/lossy_coord_v2/baseline_r1_bak0/ckpts/epoch_<maxindex>.pt' \
  train.resume_items='[all]' \
  train.rundir_name='lossy_coord_v2/baseline_r1' \
  train.device='0'
```
Training with gradient accumulation: 
```shell
python train.py config/convolutional/lossy_coord_v2/baseline_r1.yaml \
  train.device='0' train.rundir_name='lossy_coord_v2/baseline_r1' \
  train.batch_size=2 train.grad_acc_steps=4  # Approximately equivalent to batch_size 8
```


## Configuration System
The configuration system simplifies setting up the models for training and testing. 
Here’s a breakdown:

#### General Configuration:
Located in [`lib/config.py`](lib/config.py).
This file contains the root configurations used by both [`train.py`](train.py) and [`test.py`](test.py).

#### Dataset-Specific Configuration:
Located in `datasets/xxx/dataset_config.py`.
Contains configurations tailored to the specific dataset you're using.

#### Model-Specific Configuration:
Located in `models/xxx/model_config.py`.
Contains configurations tailored to the specific model you're using.

#### Specify dataset and model:
To specify the model or dataset, set the following keys in [`lib/config.py`](lib/config.py).
These modules will be automatically imported and instantiated according to the paths specified.
- `model_module_path`: Path to a Python module containing a `Model` class and a `Config` class for that model class.
- `train.dataset_module_path`: Path to a Python module containing a `Dataset` class and a `Config` class for the training dataset.
- `test.dataset_module_path`: Path to a Python module containing a `Dataset` class and a `Config` class for the test dataset.

#### Configurations via YAML or Command Line:
You don’t need to modify default values in `*.py` files. 
Instead, you can specify configuration values through YAML files or command line arguments. 
For more details, refer to the training/test commands [above](#train--test) and the YAML files located in the `config` folder.

#### Global Script Configuration:
Located in [`scripts/script_config.py`](scripts/script_config.py).
Contains several global path configurations used by `lib/metrics/xxx_wrapper.py` and `scripts/xxx.py`.
If you are using different paths, 
you need to create soft links or edit these configurations manually.


## Other content
- [`lib/simple_config.py`](lib/simple_config.py) is a simple configuration system supports separate configuration definitions, inheritance of yaml files, basic checks of argument types, and mixed use of yaml files and command line arguments.
- `lib/entropy_models` is a minor PyTorch-based re-implementation of the continuous indexed entropy models in tensorflow_compression.
- `lib/entropy_models/rans_coder` is a minor wrapper of RANS coder for simple python calls.
- `scripts` contains useful scripts for batch testing PCC models, summarizing test results and plotting RD curves (note that you should use `python scripts/xxx.py` instead of `python xxx.py` to run these scripts).


## Known issues
- For your first training on ShapeNetCorev2, the meshes in the dataset will be loaded and cached using Open3D. However, Open3D may complain about the loading of textures. You can avoid this by running `python scripts/shapenet_mtls.py`. Besides, you can manually remove the ShapeNetCorev2 cache in `datasets/ShapeNet/ShapeNetCore.v2/cache` if necessary.
- For your first test on KITTI/8iVFBv2, PLY caches with normals (*_n.ply and *_q1mm_n.ply) of the original point cloud files will be generated for pc_error, requiring approximately 55GB/12GB of storage space. You can delete them by running `find ./datasets/KITTI/ -name "*_n.ply" -delete`.


## Citation
If this work is helpful to your research, please consider citing:
````
@inproceedings{yu2023sparse,
  author={Yu, Pengpeng and Zuo, Dian and Huang, Yueer and Huang, Ruishan and Wang, Hanyun and Guo, Yulan and Liang, Fan},
  booktitle={Proceedings of the IEEE International Conference on Multimedia and Expo (ICME)}, 
  title={Sparse Representation based Deep Residual Geometry Compression Network for Large-scale Point Clouds}, 
  year={2023},
  pages={2555-2560}
}
````

## Contact
me for any reproduction issue. <yupp5@mail2.sysu.edu.cn>