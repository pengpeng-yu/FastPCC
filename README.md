### Requirements: 

- python>=3.7
- loguru
- open3d
- plyfile
- pytorch≈1.7
- pytorch3d
- MinkowskiEngine≈0.5.4 (https://github.com/NVIDIA/MinkowskiEngine)

### Datasets:
- https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
- http://modelnet.cs.princeton.edu
- https://shapenet.org
- http://plenodb.jpeg.org/pc/microsoft

### Train:

```shell
python -m torch.distributed.launch --nproc_per_node 1 train.py configs/train/convolutional/baseline.yaml
```

Definition of configuration: lib/config.py
