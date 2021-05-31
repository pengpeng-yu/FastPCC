from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.torch_utils import MLPBlock
from lib.pointnet_utils import index_points, index_points_dists, farthest_point_sample, knn_points

"""
Classes for neighborhoods-based point cloud networks.

Conventions:
Module accepts and returns a tuple (xyz, feature, raw_neighbors_feature, neighbors_idx, sample_indexes)

Args:
    B: batch size, N: current points number, C: current channels number, M: max neighbors number, S: samples number
    
    xyz: tensor with shape(B, N, 3), coordinates of points
    cached_feature: list of tensor with shape [..., (B, N1, C1), (B, N, C)], last of which is the current feature, or None
    raw_neighbors_feature: tensor with shape(B, N, M, C), neighboring infomation generated using coordinates, or None
    neighbors_idx: tensor with shape(B, N, M) or None
    cached_sample_indexes: list of tensor with shape [..., (B, S1), (B, S)] or None

    xyz is always needed as input and could not be None.
    feature[-1] is always needed as input except the first layer of a network.
    raw_neighbors_feature and neighbors_idx should be None or tensor at the same time. For a non-sample module requiring 
    neighboring feature, the module should use the two if they are not None. Otherwise, the module should generate them 
    using xyz. For a sample module, the two should be returned as None.
    sample_indexes should only be modified by sample layers. Those indexes could be used in upsampling layers manually.

"""


class TransformerBlock(nn.Module):
    def __init__(self, d_in, d_model, nneighbor, d_out=None, cache_out_feature=False) -> None:
        """
        d_in: input feature channels
        d_model: internal channels
        d_out: output channels (default: d_model)
        """
        super().__init__()
        self.nneighbor = nneighbor
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

        self.fc1 = nn.Linear(d_in, d_model)
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

        if d_out is None: d_out = d_model
        self.fc2 = nn.Linear(d_model, d_out)
        self.shortout_fc = nn.Linear(d_in, d_out)
        self.cache_out_feature = cache_out_feature

    # xyz: b, n, 3, features: b, n, d_in
    def forward(self, x):
        xyz, cached_feature, relative_knn_xyz, knn_idx, cached_sample_indexes = x
        if isinstance(cached_feature, torch.Tensor):
            cached_feature = [cached_feature]
        else:
            assert all(isinstance(_, torch.Tensor) for _ in cached_feature)
        feature = cached_feature[-1]

        if relative_knn_xyz is None or knn_idx is None:
            knn_idx = knn_points(xyz, xyz, k=self.nneighbor,  return_sorted=False).idx
            relative_knn_xyz = xyz[:, :, None, :] - index_points(xyz, knn_idx)  # knn_xyz: b, n, k, 3
        else:
            assert feature.shape[1] == relative_knn_xyz.shape[1] == knn_idx.shape[1]
            assert knn_idx.shape[2] == relative_knn_xyz.shape[2] == self.nneighbor

        pos_enc = self.fc_delta(relative_knn_xyz)  # pos_enc: b, n, k, d_model TODO: pos_encoding

        ori_features = feature
        feature = self.fc1(feature)
        knn_feature = index_points(feature, knn_idx)  # knn_feature: b, n, k, d_model
        query, key, value = self.w_qs(feature), self.w_ks(knn_feature), self.w_vs(knn_feature)
        # query: b, n, d_model   key, value: b, n, k, d_model

        attn = self.fc_gamma(query[:, :, None, :] - key + pos_enc)  # attn: b, n, k, d_model
        attn = F.softmax(attn / np.sqrt(key.size(-1)), dim=-2)

        feature = torch.einsum('bmnf,bmnf->bmf', attn,
                               value + pos_enc)  # (attn * (value + pos_enc)).sum(dim=2) feature: b, n, d_model
        feature = self.fc2(feature) + self.shortout_fc(ori_features)  # feature: b, n, d_out

        cached_feature[-1] = feature
        if self.cache_out_feature: cached_feature.append(feature)

        return xyz, cached_feature, relative_knn_xyz, knn_idx, cached_sample_indexes


class NeighborFeatureGenerator(nn.Module):
    def __init__(self, neighbor_num, channels):
        super(NeighborFeatureGenerator, self).__init__()
        assert neighbor_num > 1
        self.neighbor_num = neighbor_num
        self.channels = channels


class RandLANeighborFea(NeighborFeatureGenerator):
    def __init__(self, neighbor_num):
        super(RandLANeighborFea, self).__init__(neighbor_num, channels=3 + 3 + 3 + 1)

    def forward(self, xyz):
        relative_dists, neighbors_idx, _ = knn_points(xyz, xyz, k=self.neighbor_num, return_sorted=False)
        neighbors_xyz = index_points(xyz, neighbors_idx)

        expanded_xyz = xyz[:, :, None].expand(-1, -1, neighbors_xyz.shape[2], -1)
        relative_xyz = expanded_xyz - neighbors_xyz
        relative_dists = relative_dists[:, :, :, None]
        relative_feature = torch.cat([relative_dists, relative_xyz, expanded_xyz, neighbors_xyz], dim=-1)

        return relative_feature, neighbors_idx


class RotationInvariantDistFea(NeighborFeatureGenerator):
    def __init__(self, neighbor_num: int, anchor_points: int, retain_xyz_dists=False):
        if not anchor_points >= 3: raise NotImplementedError
        self.intra_anchor_dists_chnls = (anchor_points * (anchor_points - 1) // 2)
        self.inter_anchor_dists_chnls = anchor_points ** 2
        super(RotationInvariantDistFea, self).__init__(neighbor_num, self.intra_anchor_dists_chnls * 2 +
                                                       self.inter_anchor_dists_chnls)

        self.anchor_points = anchor_points
        self.retain_xyz_dists = retain_xyz_dists
        self.xyz_dists = None

    def forward(self, xyz, concat_raw_relative_fea=True):
        assert len(xyz.shape) == 3 and xyz.shape[2] == 3
        xyz_dists = torch.cdist(xyz, xyz)
        neighbors_dists, neighbors_idx = xyz_dists.topk(max(self.neighbor_num, self.anchor_points),
                                                        dim=2, largest=False, sorted=False)
        # xyz_dists is still necessary here and can not be replaced by neighbors_dists

        # (B, N, 15 if self.anchor_points == 6)
        intra_anchor_dists = self.gen_intra_anchor_dists(xyz_dists, neighbors_dists[:, :, :self.anchor_points],
                                                         neighbors_idx[:, :, :self.anchor_points])
        # (B, N, self.neighbor_num, self.anchor_points ** 2)
        inter_anchor_dists = self.gen_inter_anchor_dists(xyz_dists, neighbors_idx)

        if self.retain_xyz_dists:
            self.xyz_dists = xyz_dists
        else:
            del xyz_dists

        if concat_raw_relative_fea:
            # (B, N, self.neighbor_num, 15)
            center_intra_anchor_dists = intra_anchor_dists[:, :, None, :].expand(-1, -1, self.neighbor_num, -1)
            # (B, N, self.neighbor_num, 15)
            neighbor_intra_anchor_dists = index_points(intra_anchor_dists, neighbors_idx[:, :, :self.neighbor_num])
            # (B, N, self.neighbor_num, 15 + 15 + self.anchor_points ** 2)
            relative_feature = torch.cat([center_intra_anchor_dists, neighbor_intra_anchor_dists, inter_anchor_dists],
                                         dim=3)
            return relative_feature, neighbors_idx

        else:
            return intra_anchor_dists, inter_anchor_dists, neighbors_idx

    def gen_intra_anchor_dists(self, xyz_dists, neighbors_dists, neighbors_idx):
        if self.anchor_points >= 3:
            sub_anchor_dists = []
            for pi, pj in combinations(range(1, self.anchor_points), 2):
                sub_anchor_dists.append(index_points_dists(xyz_dists,
                                                           neighbors_idx[:, :, pi],
                                                           neighbors_idx[:, :, pj])[:, :, None])
            intra_anchor_dists = torch.cat([neighbors_dists[:, :, 1:], *sub_anchor_dists], dim=2)
            return intra_anchor_dists

        else:
            raise NotImplementedError

    def gen_inter_anchor_dists(self, xyz_dists, neighbors_idx):
        batch_size, points_num, _ = xyz_dists.shape
        anchor_points_idx = neighbors_idx[:, :, :self.anchor_points]  # (B, N, anchor_points)
        neighbors_idx = neighbors_idx[:, :, :self.neighbor_num]

        # (B, N, neighbor_num) -> (batch_size, points_num, neighbor_num, anchor_points)
        neighbors_anchor_points_idx = index_points(anchor_points_idx, neighbors_idx)

        anchor_points_idx = anchor_points_idx[:, :, None, :, None].expand(-1, -1, self.neighbor_num, -1,
                                                                          self.anchor_points)
        neighbors_anchor_points_idx = neighbors_anchor_points_idx[:, :, :, None, :].expand(-1, -1, -1,
                                                                                           self.anchor_points, -1)

        # (B, N, neighbor_num, anchor_points(center points index), anchor_points(neighbor points index))
        relative_dists = index_points_dists(xyz_dists, anchor_points_idx, neighbors_anchor_points_idx)
        relative_dists = relative_dists.reshape(batch_size, points_num, self.neighbor_num, self.anchor_points ** 2)
        return relative_dists


class DeepRotationInvariantDistFea(RotationInvariantDistFea):  # deprecated
    def __init__(self, neighbor_num: int, anchor_points: int, extra_intra_anchor_dists_chnls: int,
                 extra_relative_fea_chnls: int, retain_xyz_dists=False):
        super(DeepRotationInvariantDistFea, self).__init__(neighbor_num, anchor_points, retain_xyz_dists)

        mlp_relative_fea_in_chnls = (self.intra_anchor_dists_chnls + extra_intra_anchor_dists_chnls) * 2 \
                                    + self.inter_anchor_dists_chnls

        self.mlp_intra_dist = MLPBlock(self.intra_anchor_dists_chnls, extra_intra_anchor_dists_chnls,
                                       activation='leaky_relu(0.2)', batchnorm='nn.bn1d', skip_connection='concat')
        self.mlp_relative_fea = MLPBlock(mlp_relative_fea_in_chnls, extra_relative_fea_chnls,
                                         activation='leaky_relu(0.2)', batchnorm='nn.bn1d', skip_connection='concat')

        # assert extra_intra_anchor_dists_chnls >= self.intra_anchor_dists_chnls
        # assert extra_relative_fea_chnls >= self.inter_anchor_dists_chnls

        self.intra_anchor_dists_chnls += extra_intra_anchor_dists_chnls
        self.channels = mlp_relative_fea_in_chnls + extra_relative_fea_chnls

    def forward(self, xyz):
        intra_anchor_dists, inter_anchor_dists, neighbors_idx = super(DeepRotationInvariantDistFea, self).forward(xyz,
                                                                                                                  False)
        intra_anchor_fea = self.mlp_intra_dist(intra_anchor_dists)

        center_intra_anchor_fea = intra_anchor_fea[:, :, None, :].expand(-1, -1, self.neighbor_num, -1)
        neighbor_intra_anchor_fea = index_points(intra_anchor_fea, neighbors_idx[:, :, :self.neighbor_num])
        relative_feature = torch.cat([center_intra_anchor_fea, neighbor_intra_anchor_fea, inter_anchor_dists],
                                     dim=3)

        relative_feature = self.mlp_relative_fea(relative_feature)

        return relative_feature, neighbors_idx


class TransitionDown(nn.Module):
    def __init__(self, sample_method: str = 'uniform', sample_rate: float = None, nsample: int = None,
                 cache_sample_indexes=None):
        super(TransitionDown, self).__init__()
        assert (nsample is None) != (sample_rate is None)
        self.sample_method = sample_method
        self.sample_rate = sample_rate
        self.nsample = nsample
        self.cache_sample_indexes = cache_sample_indexes
        assert cache_sample_indexes in ['upsample', 'downsample', None]

    def forward(self, x):
        xyz, cached_feature, raw_neighbors_feature, neighbors_idx, cached_sample_indexes = x
        assert not xyz.requires_grad
        if raw_neighbors_feature is not None: assert not raw_neighbors_feature.requires_grad
        if neighbors_idx is not None: assert not neighbors_idx.requires_grad
        if self.training: assert all([feature.requires_grad for feature in cached_feature])
        if cached_sample_indexes is None: cached_sample_indexes = []
        del raw_neighbors_feature

        sample_indexes = self.get_indexes(xyz, neighbors_idx)
        sampled_xyz = index_points(xyz, sample_indexes)
        cached_feature[-1] = index_points(cached_feature[-1], sample_indexes)

        if self.cache_sample_indexes == 'downsample':
            cached_sample_indexes.append(sample_indexes)
        elif self.cache_sample_indexes == 'upsample':
            cached_sample_indexes.append(
                knn_points(xyz, sampled_xyz, k=1, return_sorted=False).idx)
        else:
            assert self.cache_sample_indexes is None

        return sampled_xyz, cached_feature, None, None, cached_sample_indexes

    def get_indexes(self, xyz, neighbors_idx):
        batch_size, points_num, _ = xyz.shape

        if self.nsample is not None:
            nsample = self.nsample
        else:
            nsample = int(self.sample_rate * points_num)
            assert nsample / points_num == self.sample_rate

        if self.sample_method == 'uniform':
            sample_indexes = torch.multinomial(torch.ones((1, 1), device=xyz.device).expand(batch_size, points_num),
                                               nsample, replacement=False)

        elif self.sample_method == 'uniform_batch_unaware':  # for debug purpose
            sample_indexes = torch.multinomial(torch.ones((1,), device=xyz.device).expand(points_num),
                                               nsample, replacement=False)[None, :].expand(batch_size, -1)

        elif self.sample_method == 'inverse_knn_density':
            freqs = []  # (batch_size, points_num)
            for ni in neighbors_idx:  # TODO: anchor_points is supposed to be smaller than neighbor_num
                # (points_num, )
                freqs.append(
                    torch.maximum(ni.reshape(-1).bincount(minlength=points_num), torch.tensor([1], device=xyz.device)))
            del neighbors_idx
            # (batch_size, nsample)
            sample_indexes = torch.multinomial(1 / torch.stack(freqs, dim=0), nsample, replacement=False)

        elif self.sample_method == 'fps':
            return farthest_point_sample(xyz, nsample)

        else:
            raise NotImplementedError

        return sample_indexes

    def __repr__(self):
        return f'TransitionDown({self.nsample}, {self.sample_rate}, {self.sample_method})'


class TransitionDownWithDistFea(TransitionDown):
    def __init__(self, neighbor_fea_generator: RotationInvariantDistFea, in_channels, transition_fea_chnls,
                 out_channels,
                 sample_method: str = 'uniform', sample_rate: float = None, nsample: int = None,
                 cache_sample_indexes=None):
        super(TransitionDownWithDistFea, self).__init__(sample_method, sample_rate, nsample, cache_sample_indexes)

        self.neighbor_fea_generator = neighbor_fea_generator
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transition_fea_chnls = transition_fea_chnls

        assert neighbor_fea_generator.retain_xyz_dists is True
        # assert self.in_channels >= self.neighbor_fea_generator.channels

        self.mlp_anchor_transition_fea = nn.Sequential(
            MLPBlock(self.neighbor_fea_generator.channels, self.transition_fea_chnls,
                     activation='leaky_relu(0.2)', batchnorm='nn.bn1d'),
            MLPBlock(self.transition_fea_chnls, self.transition_fea_chnls,
                     activation='leaky_relu(0.2)', batchnorm='nn.bn1d'))

        self.mlp_out = nn.Sequential(MLPBlock(self.in_channels + self.transition_fea_chnls,
                                              self.out_channels,
                                              activation='leaky_relu(0.2)', batchnorm='nn.bn1d'))

    def forward(self, x):
        xyz, cached_feature, raw_neighbors_feature, neighbors_idx_before, cached_sample_indexes = x
        batch_size, points_num, _ = xyz.shape
        feature = cached_feature[-1]
        sample_indexes = self.get_indexes(xyz, neighbors_idx_before)
        nsample = sample_indexes.shape[1]

        intra_anchor_dists_before = raw_neighbors_feature[:, :, 0,
                                    :self.neighbor_fea_generator.intra_anchor_dists_chnls]
        del raw_neighbors_feature

        sampled_xyz = index_points(xyz, sample_indexes)
        feature = index_points(feature, sample_indexes)

        if cached_sample_indexes is None: cached_sample_indexes = []
        if self.cache_sample_indexes == 'downsample':
            cached_sample_indexes.append(sample_indexes)
        elif self.cache_sample_indexes == 'upsample':
            cached_sample_indexes.append(
                knn_points(xyz, sampled_xyz, k=1, return_sorted=False).idx)
        else:
            assert self.cache_sample_indexes is None

        # After sampling, all the anchors have to be redefined due to loss of points, which introduces ambiguity of the
        # relative position and gesture between new and old anchors.
        # We manually introduce the distances information between the two and intra themselves before further
        # aggregating neighborhood anchors information. Both info before and after sampling are needed here
        # because we have to gather distances info based on point-wise distances before sampling while relying on
        # neighborhood-based info after sampling.
        # After gathering intra and inter anchors distances, mlps are performed on the sampled points with those
        # distances concatenated.

        intra_anchor_dists_before = index_points(intra_anchor_dists_before, sample_indexes)

        xyz_dists_before = self.neighbor_fea_generator.xyz_dists
        raw_neighbors_feature_after, neighbors_idx_after = self.neighbor_fea_generator(sampled_xyz)
        intra_anchor_dists_after = raw_neighbors_feature_after[:, :, 0, :self.neighbor_fea_generator.intra_anchor_dists_chnls]

        anchor_points_idx_before = index_points(neighbors_idx_before[:, :, :self.neighbor_fea_generator.anchor_points],
                                                sample_indexes)
        anchor_points_idx_before = anchor_points_idx_before[:, :, :, None].\
            expand(-1, -1, -1, self.neighbor_fea_generator.anchor_points)

        anchor_points_idx_after = neighbors_idx_after[:, :, :self.neighbor_fea_generator.anchor_points]
        # mapping indexes in anchor_points_idx_after back to the version before sampling
        anchor_points_idx_after = sample_indexes[..., None].expand(-1, -1, self.neighbor_fea_generator.anchor_points).\
            gather(dim=1, index=anchor_points_idx_after)
        anchor_points_idx_after = anchor_points_idx_after[:, :, None, :].\
            expand(-1, -1, self.neighbor_fea_generator.anchor_points, -1)

        inter_anchor_dists = index_points_dists(xyz_dists_before,
                                                anchor_points_idx_before,
                                                anchor_points_idx_after).\
            reshape(batch_size, nsample, self.neighbor_fea_generator.anchor_points ** 2)

        anchor_dist_feature = torch.cat([intra_anchor_dists_before, intra_anchor_dists_after, inter_anchor_dists],
                                        dim=2)
        if hasattr(self.neighbor_fea_generator, 'mlp_relative_fea'):
            anchor_dist_feature = self.neighbor_fea_generator.mlp_relative_fea(anchor_dist_feature)

        anchor_transition_fea = self.mlp_anchor_transition_fea(anchor_dist_feature)
        feature = torch.cat([feature, anchor_transition_fea], dim=2)
        feature = self.mlp_out(feature)

        cached_feature[-1] = feature
        return sampled_xyz, cached_feature, raw_neighbors_feature_after, neighbors_idx_after, cached_sample_indexes

    def __repr__(self):
        return f'TransitionDownWithDistFea(' \
               f'neighbor_fea_generator.channels={self.neighbor_fea_generator.channels}, ' \
               f'in_channels={self.in_channels}, ' \
               f'out_channels={self.out_channels}, ' \
               f'nsample={self.nsample}, ' \
               f'sample_rate={self.sample_rate}, ' \
               f'sample_method="{self.sample_method})"'


class LocalFeatureAggregation(nn.Module):
    def __init__(self, in_channels, neighbor_feature_generator: NeighborFeatureGenerator,
                 raw_neighbor_fea_out_chnls, out_channels, cache_out_feature=False):
        super(LocalFeatureAggregation, self).__init__()
        # assert neighbor_fea_out_chnls >= neighbor_feature_generator.channels

        self.mlp_raw_neighbor_fea = nn.Sequential(
            MLPBlock(neighbor_feature_generator.channels, raw_neighbor_fea_out_chnls,
                     activation='leaky_relu(0.2)', batchnorm='nn.bn1d'),
            MLPBlock(raw_neighbor_fea_out_chnls, raw_neighbor_fea_out_chnls,
                     activation='leaky_relu(0.2)', batchnorm='nn.bn1d'))
        if in_channels != 0:
            self.neighbor_fea_chnls = in_channels + raw_neighbor_fea_out_chnls
        else:
            self.neighbor_fea_chnls = neighbor_feature_generator.channels + raw_neighbor_fea_out_chnls
        self.mlp_neighbor_fea = nn.Sequential(MLPBlock(self.neighbor_fea_chnls, self.neighbor_fea_chnls,
                                                       activation='leaky_relu(0.2)', batchnorm='nn.bn1d'))

        self.mlp_attn = nn.Linear(self.neighbor_fea_chnls, self.neighbor_fea_chnls, bias=False)
        self.mlp_out = MLPBlock(self.neighbor_fea_chnls, out_channels, activation=None, batchnorm='nn.bn1d')
        if in_channels != 0:
            self.mlp_shortcut = MLPBlock(in_channels, out_channels, None, 'nn.bn1d')
        else:
            self.mlp_shortcut = None

        self.neighbor_feature_generator = neighbor_feature_generator
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.raw_neighbor_fea_out_chnls = raw_neighbor_fea_out_chnls
        self.cache_out_feature = cache_out_feature

    def forward(self, x):
        # There are three typical situations:
        #   1. this layer is the first layer of the model. All the rest value will be calculated using xyz
        #   and returned.
        #   2. this layer is after a sampling layer, which is supposed to be the same as situation 1.
        #   To do this, a sampling layer should return raw_relative_feature and neighbors_idx as None.
        #   3. this layer is an normal layer in the model. raw_relative_feature and neighbors_idx from last layer
        #   will be directly used.
        # This format of inputs and outputs is aimed to simplify the forward function of top-level module.

        xyz, cached_feature, raw_neighbors_feature, neighbors_idx, cached_sample_indexes = x

        if cached_feature is None:
            assert self.in_channels == 0
            cached_feature = []
        elif isinstance(cached_feature, torch.Tensor):
            assert self.in_channels != 0
            cached_feature = [cached_feature]
        else:
            assert all([isinstance(_, torch.Tensor) for _ in cached_feature])

        # calculate these value in dataloader using cpu?
        if raw_neighbors_feature is None or neighbors_idx is None:
            raw_neighbors_feature, neighbors_idx = self.neighbor_feature_generator(xyz)

        raw_neighbors_feature_mlp = self.mlp_raw_neighbor_fea(raw_neighbors_feature)

        if self.in_channels != 0:
            ori_feature = cached_feature[-1]
            # the slice below is necessary in case that RotationInvariantDistFea is used and
            # anchor_points > neighbor_num
            neighbors_feature = index_points(ori_feature,
                                             neighbors_idx[:, :, :self.neighbor_feature_generator.neighbor_num])
            neighbors_feature = torch.cat([neighbors_feature, raw_neighbors_feature_mlp], dim=3)
            feature = self.mlp_neighbor_fea(neighbors_feature)
            feature = self.attn_pooling(feature)
            feature = F.leaky_relu(self.mlp_shortcut(ori_feature) + self.mlp_out(feature), negative_slope=0.2)

        else:
            assert cached_feature == []
            neighbors_feature = torch.cat([raw_neighbors_feature, raw_neighbors_feature_mlp], dim=3)
            feature = self.mlp_neighbor_fea(neighbors_feature)
            feature = self.attn_pooling(feature)
            feature = F.leaky_relu(self.mlp_out(feature), negative_slope=0.2)

        if self.in_channels != 0:
            cached_feature[-1] = feature
        else:
            cached_feature.append(feature)
        if self.cache_out_feature: cached_feature.append(feature)  # be careful about in-place operations
        return xyz, cached_feature, raw_neighbors_feature, neighbors_idx, cached_sample_indexes

    def attn_pooling(self, feature):
        attn = F.softmax(self.mlp_attn(feature), dim=2)
        feature = attn * feature
        feature = torch.sum(feature, dim=2)
        return feature


def transformer_block_t():
    input_xyz = torch.rand(4, 100, 3)
    input_feature = torch.rand(4, 100, 32)
    transfomer_blocks = nn.Sequential(TransformerBlock(d_in=32, d_model=64, nneighbor=16, cache_out_feature=True),
                                      TransformerBlock(d_in=64, d_model=128, nneighbor=16, cache_out_feature=True),
                                      TransitionDown('uniform', 0.5, cache_sample_indexes=True),
                                      TransformerBlock(d_in=128, d_model=128, nneighbor=16, cache_out_feature=True),
                                      TransformerBlock(d_in=128, d_model=256, nneighbor=16, cache_out_feature=True),
                                      TransitionDown('uniform', 0.5, cache_sample_indexes=True))
    out = transfomer_blocks((input_xyz, [input_feature], None, None, None))
    out[1][-1].sum().backward()
    print('Done')


def lfa_test_1():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            neighbor_fea_generator = RandLANeighborFea(16)
            self.layers = nn.Sequential(LocalFeatureAggregation(3, neighbor_fea_generator, 16, 32),
                                        LocalFeatureAggregation(32, neighbor_fea_generator, 32, 64,
                                                                cache_out_feature=True),
                                        TransitionDown('uniform', 0.5, cache_sample_indexes=True),
                                        LocalFeatureAggregation(64, neighbor_fea_generator, 64, 128),
                                        LocalFeatureAggregation(128, neighbor_fea_generator, 64, 128,
                                                                cache_out_feature=True),
                                        TransitionDown('uniform', 0.5))

        def forward(self, x):
            out = self.layers((x, [x], None, None, None))
            return out[1][-1].sum()

    model = Model()
    model.train()
    xyz = torch.rand(16, 100, 3)
    out = model(xyz)
    out.backward()
    print('Done')


def lfa_test_2():
    class Model2(nn.Module):
        def __init__(self):
            super(Model2, self).__init__()
            neighbor_fea_generator = RotationInvariantDistFea(16, 4, retain_xyz_dists=True)
            self.layers = nn.Sequential(
                LocalFeatureAggregation(0, neighbor_fea_generator, 16, 32),
                LocalFeatureAggregation(32, neighbor_fea_generator, 32, 64, cache_out_feature=True),
                TransitionDownWithDistFea(neighbor_fea_generator, 64, 32, 64, 'uniform', 0.5,
                                          cache_sample_indexes=True),
                LocalFeatureAggregation(64, neighbor_fea_generator, 64, 128),
                LocalFeatureAggregation(128, neighbor_fea_generator, 64, 128, cache_out_feature=True),
                TransitionDownWithDistFea(neighbor_fea_generator, 128, 64, 128, 'uniform', 0.5,
                                          cache_sample_indexes=True))

        def forward(self, x):
            out = self.layers((x, None, None, None, None))
            return out[1][-1].sum()

    model2 = Model2()
    model2.train()
    xyz2 = torch.rand(16, 100, 3)
    out2 = model2(xyz2)
    out2.backward()
    print('Done')


if __name__ == '__main__':
    lfa_test_1()
    lfa_test_2()
    transformer_block_t()
