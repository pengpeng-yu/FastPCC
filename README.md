This project contains an implementation of our ICME 2023 paper "Sparse Representation based Deep Residual Geometry Compression Network for Large-scale Point Clouds" [1] and other improvements, 
tested on Ubuntu 18.04.


## Highlights
See our detailed experimental results on Intel E5-2678 v3 and NVIDIA 2080Ti at [OneDrive](https://mssysueducn-my.sharepoint.com/:u:/g/personal/yupp5_ms_sysu_edu_cn/EbzFDM93okNPmceKE5ZLzhgBZPJ1Cb4L-GeoP3stilFJxQ). 


## Models
- `config/convolutional/lossy_coord_v2/baseline_r*.yaml`: Improved geometry lossy compression based on [1]. 
- `config/convolutional/lossy_coord_lossy_color/baseline_r*.yaml`: Joint geometry and color lossy compression. 
- `config/convolutional/lossy_coord/lossl_based*.yaml`: The configs of model in [1].
- `config/convolutional/lossy_coord/baseline.yaml`: A reimplementation of PCGCv2.


## Other content
- `lib/simple_config.py` is a simple configuration system supports separate configuration definitions, inheritance of yaml files, basic checks of argument types, and mixed use of yaml files and command line arguments.
- `lib/entropy_models` is a minor PyTorch-based re-implementation of the continuous indexed entropy models in tensorflow_compression.
- `lib/entropy_models/rans_coder` is a minor wrapper of RANS coder for simple python calls.
- `scripts` contains useful scripts for batch testing PCC models, summarizing test results and plotting RD curves.


## Environment
- python >= 3.7
- loguru
- open3d
- plyfile
- pytorch >= 1.7
- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) (only for knn_points in joint lossy compression. Conda installation suggested: conda install -c fvcore -c iopath -c conda-forge fvcore iopath; conda install pytorch3d -c pytorch3d)
- [minkowskiengine](https://github.com/NVIDIA/MinkowskiEngine?tab=readme-ov-file#pip) ≈ 0.5.4 (for compilation with CUDA >= 12, please refer to [this](https://github.com/NVIDIA/MinkowskiEngine/issues/543#issuecomment-1773458776))


## Requirements
- [Binary pc_error and tmc3 compiled on Ubuntu 18.04. And an example of dataset folder](https://mssysueducn-my.sharepoint.com/:u:/g/personal/yupp5_ms_sysu_edu_cn/EWQeTYD3y8dDr1-lvq7YawQB-JoJ2_GQw2jSOgpS2i6xLw)
- [Test set of geometry compression](https://mssysueducn-my.sharepoint.com/:u:/g/personal/yupp5_ms_sysu_edu_cn/Ed9ljLgHJD9Ipd_Sd8nP9NABxJbywu1kT1pPo4vdNIZWjg)
- [Trained model weights](https://mssysueducn-my.sharepoint.com/:u:/g/personal/yupp5_ms_sysu_edu_cn/EUIl3GRmQL5KvKCwwipEOhkBzoNXfyDwMs_BHju7M70ayg)
- [ShapeNetCorev2 OBJ format](https://huggingface.co/datasets/ShapeNet/ShapeNetCore/tree/main) (for training geometry compression)
- [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (for training and testing LiDAR geometry compression)
- [8iVFBv2](https://plenodb.jpeg.org/pc/8ilabs) & [Owlii](https://mpeg-pcc.org/index.php/pcc-content-database/owlii-dynamic-human-textured-mesh-sequence-dataset/) (for joint geometry and color compression)


## Train / Test
Training:
```shell
python train.py config/convolutional/lossy_coord_v2/baseline_r1.yaml \
  train.device='0' train.rundir_name='lossy_coord_v2/baseline_r1'
```
Test:
```shell
python test.py config/convolutional/lossy_coord_v2/baseline_r1.yaml \
  test.weights_from_ckpt='weights/convolutional/lossy_coord_v2/baseline_r1.pt' \
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
  train.resume_from_ckpt='runs/lossy_coord_v2/baseline_r1_bak0/ckpts/epoch_<maxindex>.pt' \
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
The definition of training and testing configurations is lib/config.py


## Known issues
- For your first training on ShapeNetCorev2, the meshes in the dataset will be loaded and cached using Open3D. However, Open3D may complain about the loading of textures. You can avoid this by running `scripts/shapenet_mtls.py`. Besides, you can manually remove the ShapeNetCorev2 cache in `datasets/ShapeNet/ShapeNetCore.v2/cache` if necessary.
- For your first test on KITTI/8iVFBv2, PLY caches with normals (*_n.ply and *_q1mm_n.ply) of the original point cloud files will be generated for pc_error, requiring approximately 55GB/12GB of storage space. You can delete them by `find ./datasets/KITTI/ -name "*_n.ply" -delete`.
- There are some experimental code snippets in this project. Only the models we mentioned above are recommended for use. 


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