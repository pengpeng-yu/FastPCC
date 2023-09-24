### Requirements: 

- python >= 3.7
- loguru
- open3d
- plyfile
- pytorch ≈ 1.7
- pytorch3d (for knn_points)
- minkowskiengine ≈ 0.5.4 (https://github.com/NVIDIA/MinkowskiEngine)

### Datasets:
- https://shapenet.org/download/shapenetcore
- http://plenodb.jpeg.org/pc/microsoft
- https://plenodb.jpeg.org/pc/8ilabs
- https://mpeg-pcc.org/index.php/pcc-content-database/owlii-dynamic-human-textured-mesh-sequence-dataset/

### Train:

```shell
python train.py configs/convolutional/lossy_coord_v2/baseline_r1.yaml
```
or 
```shell
python -m torch.distributed.launch --nproc_per_node 2 train.py configs/convolutional/lossy_coord_v2/baseline_r1.yaml
```

The definition of training and testing configurations: lib/config.py
