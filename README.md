Requirements: 

- python>=3.7
- loguru
- pyvista
- open3d
- pytorch≈1.7
- pytorch3d
- compressai (https://github.com/InterDigitalInc/CompressAI.git)
- MinkowskiEngine≈0.5.4 (https://github.com/NVIDIA/MinkowskiEngine)

Dataset:
- https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
- http://modelnet.cs.princeton.edu
- https://shapenet.org
- http://plenodb.jpeg.org/pc/microsoft

Default config: lib/config.py

Train: `python -m torch.distributed.launch --nproc_per_node 2 train.py config/train/config_config_convolutional_PCGCv2.yaml`

