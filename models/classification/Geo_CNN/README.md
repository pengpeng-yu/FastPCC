# 🏔 Geo-CNN-Pytorch-PYG
A Pytorch re-implementation of “Modeling Local Geometric Structure of 3D Point Clouds using Geo-CNN”

- This repository is a reproduction of the **GeoCNN**, which can support multiple GPUs.
- My enviroment:
  - Ubuntu 18.04
  - Anaconda Python 3.7
  - [Pytorch](https://github.com/pytorch/pytorch) 1.5.0
  - [PYG](https://github.com/rusty1s/pytorch_geometric) 1.5.0
  - Cuda 10.2
  - Cudnn 7.6.5
  - GPU Memory >= 8G
- If you like graph neural network, too. Welcome to our 🐧 QQ group: `832405795`

## Accuracy on ModelNet40
|this implementation|original paper|
|---|---|
|93.2|93.4|

## How to Use This Code
- Prepare Data
  - Download [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) data set
  - Move `modelnet40_normal_resampled.zip` into `data/ModelNet40_10000`
  - Unzip `modelnet40_normal_resampled.zip`
  - Rename `modelnet40_normal_resampled` to `raw`
- Train
  - We can change args in the [Configuration part](https://github.com/cy69855522/Geo-CNN-Pytorch-PYG/blob/master/geocnn.py#L25) of the code if you want
  - Then let’s start training: `python geocnn.py`
- Test
  - Uncomment [this line](https://github.com/cy69855522/Geo-CNN-Pytorch-PYG/blob/master/geocnn.py#L248) and replace the weight path
  - Set [only_test](https://github.com/cy69855522/Geo-CNN-Pytorch-PYG/blob/master/geocnn.py#L40) as `True`
  - Then let’s start testing: `python geocnn.py`

## Bibtex
```
@article{DBLP:journals/corr/abs-1811-07782,
  author    = {Shiyi Lan and
              Ruichi Yu and
              Gang Yu and
              Larry S. Davis},
  title     = {Modeling Local Geometric Structure of 3D Point Clouds using Geo-CNN},
  journal   = {CoRR},
  volume    = {abs/1811.07782},
  year      = {2018},
  url       = {http://arxiv.org/abs/1811.07782},
  archivePrefix = {arXiv},
  eprint    = {1811.07782},
  timestamp = {Mon, 26 Nov 2018 12:52:45 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1811-07782},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
