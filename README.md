This project contains an implementation of our ICME 2023 paper "Sparse Representation based Deep Residual Geometry Compression Network for Large-scale Point Clouds" [1], 
tested on Ubuntu 20.04. This project is still under development. 

## Models
- `config/convolutional/lossy_coord_v2` and `config/convolutional/lossy_coord_lossy_color`: coming, but not soon. 
- `config/convolutional/lossy_coord` contains the yaml files of the main model and its variants in this paper [1].


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
- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) (for knn_points. Conda installation suggested: conda install -c fvcore -c iopath -c conda-forge fvcore iopath; conda install pytorch3d -c pytorch3d)
- [minkowskiengine](https://github.com/NVIDIA/MinkowskiEngine?tab=readme-ov-file#pip) ≈ 0.5.4


## Requirements
- [Binary pc_error and tmc3 compiled on Ubuntu 20.04. And an example of dataset folder](https://drive.google.com/file/d/1Z0h7umZ6YKtxmqY6h-lBK5aS6I_5Kjo9/view?usp=sharing)
- [Test set of coordinate compression](https://drive.google.com/file/d/1GT3L33ye70uku-HXI1pqU7diuiL3sRGo/view?usp=drive_link)
- [Trained model weights]() (will be uploaded soon)
- [ShapeNetCorev2 OBJ format](https://huggingface.co/datasets/ShapeNet/ShapeNetCore/tree/main) (for training coordinate compression)
- [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (for training and testing coordinate compression)
- [8iVFBv2](https://plenodb.jpeg.org/pc/8ilabs) & [Owlii](https://mpeg-pcc.org/index.php/pcc-content-database/owlii-dynamic-human-textured-mesh-sequence-dataset/) (for training joint coordinate and color compression)


## Train / Test
Training:
```shell
python train.py config/convolutional/lossy_coord/baseline.yaml \
  train.device='0' train.rundir_name='lossy_coord/baseline'
```
Test:
```shell
python test.py config/convolutional/lossy_coord/baseline.yaml \
  test.weights_from_ckpt='weights/convolutional/lossy_coord/0.1_epoch_39.pt' \
  test.device='0'
```
DDP training: 
```shell
python -m torch.distributed.launch --nproc_per_node 2 \
  train.py config/convolutional/lossy_coord/baseline.yaml \
  train.device='0,1' train.rundir_name='lossy_coord/baseline'
```
Resume training from "runs/lossy_coord/baseline_bak0" to "runs/lossy_coord/baseline":
```shell
python train.py config/convolutional/lossy_coord/baseline.yaml \
  train.resume_from_ckpt='runs/lossy_coord/baseline_bak0/ckpts/epoch_<maxindex>.pt' \
  train.resume_items='[all]' \
  train.rundir_name='lossy_coord/baseline' \
  train.device='0'
```

The definition of training and testing configurations is lib/config.py


## Known issues
- For your first training on ShapeNetCorev2, the meshes in the dataset will be loaded and cached using Open3D. However, Open3D may complain about the loading of textures. You can avoid this by running `scripts/shapenet_mtls.py`. Besides, you can manually remove the ShapeNetCorev2 cache in `datasets/ShapeNet/ShapeNetCore.v2/cache` if necessary.
- For your first test on KITTI/8iVFBv2, PLY caches with normals (*_n.ply) of the original point cloud files will be generated for pc_error, requiring approximately 55GB/12GB of storage space. You can delete them by `find ./datasets/KITTI -name "*_n.ply" -delete`.
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