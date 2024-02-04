This project contains an implementation of our ICME 2023 paper "Sparse Representation based Deep Residual Geometry Compression Network for Large-scale Point Clouds" [1], 
tested on Ubuntu 20.04.

## Contents
- `config/convolutional/lossy_coord_v2` and `config/convolutional/lossy_coord_lossy_color`: coming soon. 
- `config/convolutional/lossy_coord` contains the yaml files of the main model and its variants in this paper [1].
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
- [pytorch3d](https://github.com/facebookresearch/pytorch3d) (for knn_points)
- [minkowskiengine](https://github.com/NVIDIA/MinkowskiEngine) ≈ 0.5.4


## Requirements
- [Binary pc_error and tmc3 (compiled on Ubuntu 20.04). And a dataset folder example](https://drive.google.com/file/d/1RC62ddx_YTp0ZtwUhIXknM614sESg0ca/view?usp=sharing)
- [Test set](https://drive.google.com/file/d/1GT3L33ye70uku-HXI1pqU7diuiL3sRGo/view?usp=sharing)
- [Trained model weights](https://drive.google.com/file/d/1ivYoBtZszP8R-hO5trlulRVwZ5vO9sM9/view?usp=sharing)
- [ShapeNet (for training)](https://shapenet.org/download/shapenetcore)
- [8iVFBv2 (for training)](https://plenodb.jpeg.org/pc/8ilabs)


## Train / Test
Training:
```shell
python train.py config/convolutional/lossy_coord/baseline.yaml \
  train.device='0'
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
  train.device='0,1'
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
