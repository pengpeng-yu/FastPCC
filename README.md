## Requirements

- python >= 3.7
- loguru
- open3d
- plyfile
- pytorch ≈ 1.7
- [pytorch3d](https://github.com/facebookresearch/pytorch3d) (for knn_points)
- [minkowskiengine](https://github.com/NVIDIA/MinkowskiEngine) ≈ 0.5.4

## Datasets
- [ShapeNet](https://shapenet.org/download/shapenetcore)
- [MVUB](http://plenodb.jpeg.org/pc/microsoft)
- [8iVFBv2](https://plenodb.jpeg.org/pc/8ilabs)
- [Owlii](https://mpeg-pcc.org/index.php/pcc-content-database/owlii-dynamic-human-textured-mesh-sequence-dataset/)

## Resources
- [Binary pc_error, tmc3 (compiled on Ubuntu20.04). And an example of the datasets folder](https://)
- [Test datasets](https://)
- [All trained model weights](https://)

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
