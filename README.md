Requirements: 

- loguru
- pytorch≈1.7
- compressai (https://github.com/InterDigitalInc/CompressAI.git)
- MinkowskiEngine≈0.5.4 (https://github.com/NVIDIA/MinkowskiEngine)

Dataset: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip

Default config: lib/config.py

Train: `python -m torch.distributed.launch --nproc_per_node 2 train.py config/train/config_exp1.yaml`

