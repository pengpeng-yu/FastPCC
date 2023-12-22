This project contains implementation for our papers 
and 
. 
Code files were tested on Ubuntu 20.04.

## Content
- `config/convolutional/lossy_coord_v2` and `config/convolutional/lossy_coord_lossy_color`  
contain the yaml files of the main model and its variants in paper [1](https://).
- `config/convolutional/lossy_coord` contains the yaml files of the main model and its variants in paper [1](https://).
- `lib/simple_config.py` is a simple configuration system supports separate configuration definitions, inheritance of yaml files, basic checks of argument types, and mixed use of yaml files and command line arguments.
- `lib/entropy_models` is a minor PyTorch-based re-implementation of the continuous indexed entropy models in tensorflow_compression.
- `lib/entropy_models/rans_coder` is a minor wrapper of RANS coder for simple python calls.
- `scripts` contains useful scripts for batch testing PCC models, summarizing test results and plotting RD curves.


## Requirements
- python >= 3.7
- loguru
- open3d
- plyfile
- pytorch >= 1.7
- [pytorch3d](https://github.com/facebookresearch/pytorch3d) (for knn_points)
- [minkowskiengine](https://github.com/NVIDIA/MinkowskiEngine) ≈ 0.5.4


## Resources
- [Binary pc_error and tmc3 (compiled on Ubuntu 20.04). A dataset folder example](https://drive.google.com/file/d/1RC62ddx_YTp0ZtwUhIXknM614sESg0ca/view?usp=sharing)
- [Test datasets](https://drive.google.com/file/d/1GT3L33ye70uku-HXI1pqU7diuiL3sRGo/view?usp=sharing)
- [All trained model weights](https://drive.google.com/file/d/1ivYoBtZszP8R-hO5trlulRVwZ5vO9sM9/view?usp=sharing)
- [ShapeNet (for training)](https://shapenet.org/download/shapenetcore)
- [8iVFBv2 (for training)](https://plenodb.jpeg.org/pc/8ilabs)


## Train / Test
Training:
```shell
python train.py \
  config/convolutional/lossy_coord_v2/baseline_r1.yaml \
  train.device='0'
```
Testing:
```shell
python test.py config/convolutional/lossy_coord_v2/baseline_r1.yaml \
  test.weights_from_ckpt='weights/convolutional/lossy_coord_v2/baseline_r1.pt' \
  test.device='0'
```
DDP training: 
```shell
python -m torch.distributed.launch --nproc_per_node 2 \
  train.py config/convolutional/lossy_coord_v2/baseline_r1.yaml \
  train.device='0,1'
```
Resume training:
```shell
python train.py \
  config/convolutional/lossy_coord_v2/baseline_r1.yaml \
  train.resume_from_ckpt='runs/lossy_coord_v2/baseline_r1_bak0/ckpts/epoch_<maxindex>.pt' \
  train.resume_items='[all]' \
  train.rundir_name='lossy_coord_v2/baseline_r1/' \
  train.device='0'
```

The definition of training and testing configurations is lib/config.py


## Citation
If this work is helpful to your research, please cite:
````
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={2023}
}
@inproceedings{,
  title={},
  author={},
  journal={},
  year={}
}
````
