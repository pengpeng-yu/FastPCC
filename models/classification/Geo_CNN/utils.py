# -*- coding: utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_txt_array
import math
import random
from tqdm import tqdm
from itertools import repeat
import os.path as osp

class ShufflePoints(object):
    def __call__(self, data):
        idx = torch.randperm(data.pos.size(0))
        data['pos'] = data.pos[idx]
        if data.norm is not None:
            data.norm = data.norm[idx]
        return data

class MovePoints(object):
    def __init__(self, mrange=(-0.2, 0.2)):
        self.low, self.high = mrange

    def __call__(self, data):
        data['pos'] += torch.ones(3, dtype=data.pos.dtype, device=data.pos.device).uniform_(self.low, self.high)
        return data

class ScalePoints(object):
    def __init__(self, srange=(2./3, 3./2)):
        self.low, self.high = srange

    def __call__(self, data):
        data['pos'] *= torch.ones(3, dtype=data.pos.dtype, device=data.pos.device).uniform_(self.low, self.high)
        return data

class Jitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, data):
        jittered_data = data.pos.new(data.pos.size(0), 3).normal_(
            mean=0.0, std=self.std
        ).clamp_(-self.clip, self.clip)
        data['pos'] += jittered_data
        
        return data

    def __repr__(self):
        return '{}(std: {},  clip: {})'.format(self.__class__.__name__, self.std, self.clip)

class ChunkPoints(object):
    def __init__(self, num_points=1024, random_start=False):
        self.N = num_points
        self.random_start = random_start
        
    def __call__(self, data):
        start = 0
        if self.random_start:
            start = random.randint(0, data.pos.size(0) - self.N)
        data['pos'] = data.pos[start : start + self.N]
        if data.norm is not None:
            data.norm = data.norm[start : start + self.N]
        return data

class UnitSphere(object):
    r"""Centers and normalizes node positions to an unit sphere.
    """
    def __call__(self, data):
        data.pos -= data.pos.mean(dim=-2)
        data.pos /= data.pos.norm(dim=-1).max()

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def Norm(name, c, channels_per_group=16, momentum=0.1, md=1):
    if name == 'bn':
        return eval(f'nn.BatchNorm{md}d')(c, momentum=momentum)
    elif name == 'gn':
        num_group = c // channels_per_group
        if num_group * channels_per_group != c:
            num_group = 1
        return nn.GroupNorm(num_group, c)

class ModelNet40_10000(InMemoryDataset):

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        self.name = '40'
        super(ModelNet40_10000, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
            'night_stand', 'sofa', 'table', 'toilet'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        f = osp.join(self.raw_dir, f'modelnet{self.name}_shape_names.txt')
        with open(f, 'r') as f:
            categories = f.read().split('\n')[:-1]
            cate_id = {cate : i for i, cate in enumerate(categories)}
            
        f = osp.join(self.raw_dir, f'modelnet{self.name}_{dataset}.txt')
        with open(f, 'r') as f:
            file_list = f.read().split('\n')[:-1]
        
        data_list = []
        with tqdm(file_list) as t:
            for file_name in t:
                category = '_'.join(file_name.split('_')[:-1])
                f = osp.join(self.raw_dir, category, f'{file_name}.txt')
                data = read_txt_array(f, sep=',')
                data = Data(pos=data[:, :3], norm=data[:, 3:])
                data.y = torch.tensor([cate_id[category]])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))
    
    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
            
        data.pos = data.pos.clone()
        if data.norm is not None: data.norm = data.norm.clone()
        data['path_id'] = idx
        
        return data

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def knn(x, k, d=1, f=None): # x[B, N, C] f[B*n]
    B, N, _ = x.size()
    n = N if f is None else int(f.size(0) / B)
    dev = x.device
    inner = -2 * torch.matmul(x, x.transpose(2, 1)) # [B, N, N]
    xx = torch.sum(x**2, dim=-1, keepdim=True) # [B, N, 1]
    dis = -xx.transpose(2, 1) - inner - xx # [B, N, N]
    if f is not None: dis = dis.view(B*N, N)[f].view(B, n, N)
    sid = dis.topk(k=k*d-d+1, dim=-1)[1][..., ::d] # (B, n, k)
    sid += torch.arange(B, device=dev).view(B, 1, 1) * N
    sid = sid.reshape(-1) # [B*n*k]
    tid = torch.arange(B * N, device=dev) if f is None else f # [B*n]
    tid = tid.view(-1, 1).repeat(1, k).view(-1) # [B*n*k]
    return sid, tid # [B*n*k]

def sphg(pos, r, batch=None, flow='source_to_target', max_num_neighbors=48, fpsi=None, resetFpsi=False, random_replace=True):
    # Make sure "batch" is ascending
    assert flow in ['source_to_target', 'target_to_source']

    B = batch[-1].item() + 1
    N = int(len(pos) / B)
    n = N if fpsi is None else int(len(fpsi) / B)
    C = pos.size(-1)
    k = max_num_neighbors
    dev = pos.device
    
    with torch.no_grad():
      pos_i = pos.view(B, N, C) if fpsi is None else pos[fpsi].view(B, n, C)
      pos_j = pos.view(B, N, C)
      dis = pos_i.unsqueeze(-2).expand(B, n, N, C) - pos_j.unsqueeze(-2).transpose(1, 2)
      dis = dis.norm(dim=-1) # [B, n, N]
      max_valid_neighbors = max(k, (dis <= r).sum(dim=-1).max())
      dis, sid = dis.topk(max_valid_neighbors, largest=False) # [B, n, max_valid_neighbors]
      sid += torch.arange(B, device=dev, dtype=sid.dtype).view(B, 1, 1) * N
      invalid_mask = dis > r # [B, n, max_valid_neighbors]
      # For those have too many valid neighbors, randomly shuffle and choose without repetition
      shuffle_order = torch.rand(B, n, max_valid_neighbors, device=dev)
      shuffle_order[invalid_mask] = -1
      _, shuffle_order = shuffle_order.topk(k, largest=True)
      sid = sid.gather(-1, shuffle_order)[..., :k] # [B, n, k]
      # Invalid neighbors are clustered at the end, so we can intercept the mask directly
      invalid_mask = invalid_mask[..., :k] # [B, n, k]
      # For those have less valid neighbors, randomly replace all invalid neighbors
      if random_replace:
          replacement = torch.rand(B, n, k, device=dev) * (k - invalid_mask.float().sum(dim=-1, keepdim=True))
          replacement.floor_()
          replacement.clamp_(max=k-1)
          replacement = sid.gather(-1, replacement.long())
          sid[invalid_mask] = replacement[invalid_mask]
      else:
          sid[invalid_mask] = sid[..., 0:1].expand(B, n, k)[invalid_mask]
      sid = sid.view(-1)
      if fpsi is None or resetFpsi: fpsi = torch.arange(B * n, device=dev)
      tid = fpsi.view(-1, 1).repeat(1, k).view(-1)
    
    return (sid, tid) if flow == 'source_to_target' else (tid, sid)
    

if __name__ == "__main__":
    train_dataset = ModelNet40_10000('ModelNet40_10000')
    print(train_dataset[0].pos)