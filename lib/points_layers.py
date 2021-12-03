from itertools import combinations
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.torch_utils import MLPBlock
from lib.pointnet_utils import index_points, index_points_dists, farthest_point_sample, knn_points


class PointLayerMessage:
    def __init__(self,
                 xyz: torch.Tensor,
                 feature: Optional[torch.Tensor] = None,
                 raw_neighbors_feature: Optional[torch.Tensor] = None,
                 neighbors_idx: Optional[torch.Tensor] = None,
                 cached_xyz: Optional[List[torch.Tensor]] = None,
                 cached_feature: Optional[List[torch.Tensor]] = None,
                 cached_sample_indexes: Optional[List[torch.Tensor]] = None):
        """
        Args:
            xyz: tensor with shape(B, N, 3), coordinates of points
            raw_neighbors_feature: tensor with shape(B, N, M, C),
                neighboring information generated using coordinates.
            neighbors_idx: tensor with shape(B, N, M)

            B batch size, N current points number, C current channels number,
            M max neighbors number, S samples number

            "xyz" is always needed as input.
            "feature" is always needed as input except the first layer of a network.
            "raw_neighbors_feature" and neighbors_idx should be None or tensor at the same time.
            For a non-sample module requiring neighboring feature, the module should
            use the two if they are not None. Otherwise, the module should generate
            them using xyz.
            "sample_indexes" should only be modified by sample layers.
            Those indexes could be used in upsample layers.
        """
        self.xyz = xyz
        self.feature = feature
        self.raw_neighbors_feature = raw_neighbors_feature  # B, N, K, 3
        self.neighbors_idx = neighbors_idx
        self.cached_xyz = cached_xyz or []
        self.cached_feature = cached_feature or []
        self.cached_sample_indexes = cached_sample_indexes or []


class TransformerBlock(nn.Module):
    def __init__(self, d_in, d_model, nneighbor, d_out=None, cache_out_feature=False):
        """
        d_in: input feature channels
        d_model: internal channels
        d_out: output channels (default: d_model)
        """
        super().__init__()
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
        self.nneighbor = nneighbor
        self.fc2 = nn.Linear(d_model, d_out)
        self.shortcut_fc = nn.Linear(d_in, d_out)
        self.cache_out_feature = cache_out_feature

    # xyz: b, n, 3, features: b, n, d_in
    def forward(self, msg: PointLayerMessage):
        if msg.raw_neighbors_feature is None or msg.neighbors_idx is None:
            msg.neighbors_idx = knn_points(
                msg.xyz, msg.xyz,
                k=self.nneighbor, return_sorted=False
            ).idx
            msg.raw_neighbors_feature = \
                msg.xyz[:, :, None, :] - \
                index_points(msg.xyz, msg.neighbors_idx)

        feature = self.fc1(msg.feature)
        knn_feature = index_points(feature, msg.neighbors_idx)  # knn_feature: b, n, k, d_model
        query, key, value = self.w_qs(feature), self.w_ks(knn_feature), self.w_vs(knn_feature)
        # query: b, n, d_model   key, value: b, n, k, d_model

        pos_enc = self.fc_delta(msg.raw_neighbors_feature)  # pos_enc: b, n, k, d_model
        attn = self.fc_gamma(query[:, :, None, :] - key + pos_enc)  # attn: b, n, k, d_model
        attn = F.softmax(attn / np.sqrt(key.size(-1)), dim=-2)

        feature = torch.einsum(
            'bmnf,bmnf->bmf', attn,
            value + pos_enc
        )  # (attn * (value + pos_enc)).sum(dim=2) feature: b, n, d_model
        feature = self.fc2(feature) + self.shortcut_fc(msg.feature)  # feature: b, n, d_out
        msg.feature = feature
        if self.cache_out_feature: msg.cached_feature.append(feature)
        return msg


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
        relative_dists, neighbors_idx, _ = knn_points(
            xyz, xyz,
            k=self.neighbor_num, return_sorted=False
        )
        neighbors_xyz = index_points(xyz, neighbors_idx)

        expanded_xyz = xyz[:, :, None].expand(-1, -1, neighbors_xyz.shape[2], -1)
        relative_xyz = expanded_xyz - neighbors_xyz
        relative_dists = relative_dists[:, :, :, None]

        relative_feature = torch.cat(
            [relative_dists,
             relative_xyz,
             expanded_xyz,
             neighbors_xyz], dim=-1
        )
        return relative_feature, neighbors_idx


class RotationInvariantDistFea(NeighborFeatureGenerator):
    def __init__(self, neighbor_num: int, anchor_points: int, retain_xyz_dists=False):
        if not anchor_points >= 3: raise NotImplementedError
        self.intra_anchor_dists_chnls = (anchor_points * (anchor_points - 1) // 2)
        self.inter_anchor_dists_chnls = anchor_points ** 2
        assert anchor_points < neighbor_num
        super(RotationInvariantDistFea, self).__init__(
            neighbor_num,
            self.intra_anchor_dists_chnls * 2 + self.inter_anchor_dists_chnls
        )
        self.anchor_points = anchor_points
        self.retain_xyz_dists = retain_xyz_dists
        self.xyz_dists = None

    def forward(self, xyz, concat_raw_relative_fea=True):
        assert len(xyz.shape) == 3 and xyz.shape[2] == 3
        xyz_dists = torch.cdist(xyz, xyz)
        neighbors_dists, neighbors_idx = xyz_dists.topk(
            max(self.neighbor_num, self.anchor_points),
            dim=2, largest=False, sorted=False
        )
        # xyz_dists is still necessary here and can not be replaced by neighbors_dists

        # (B, N, 15 if self.anchor_points == 6)
        intra_anchor_dists = self.gen_intra_anchor_dists(
            xyz_dists,
            neighbors_dists[:, :, :self.anchor_points],
            neighbors_idx[:, :, :self.anchor_points]
        )
        # (B, N, self.neighbor_num, self.anchor_points ** 2)
        inter_anchor_dists = self.gen_inter_anchor_dists(
            xyz_dists,
            neighbors_idx
        )

        if self.retain_xyz_dists:
            self.xyz_dists = xyz_dists
        else: del xyz_dists

        if concat_raw_relative_fea:
            # (B, N, self.neighbor_num, 15)
            center_intra_anchor_dists = \
                intra_anchor_dists[:, :, None, :].expand(-1, -1, self.neighbor_num, -1)
            # (B, N, self.neighbor_num, 15)
            neighbor_intra_anchor_dists = \
                index_points(intra_anchor_dists, neighbors_idx[:, :, :self.neighbor_num])
            # (B, N, self.neighbor_num, 15 + 15 + self.anchor_points ** 2)
            relative_feature = torch.cat(
                [center_intra_anchor_dists,
                 neighbor_intra_anchor_dists,
                 inter_anchor_dists], dim=3
            )
            return relative_feature, neighbors_idx
        else:
            return intra_anchor_dists, inter_anchor_dists, neighbors_idx

    def gen_intra_anchor_dists(self, xyz_dists, neighbors_dists, neighbors_idx):
        if self.anchor_points >= 3:
            sub_anchor_dists = []
            for pi, pj in combinations(range(1, self.anchor_points), 2):
                sub_anchor_dists.append(
                    index_points_dists(
                        xyz_dists,
                        neighbors_idx[:, :, pi],
                        neighbors_idx[:, :, pj]
                    )[:, :, None]
                )
            intra_anchor_dists = torch.cat(
                [neighbors_dists[:, :, 1:], *sub_anchor_dists], dim=2)
            return intra_anchor_dists
        else:
            raise NotImplementedError

    def gen_inter_anchor_dists(self, xyz_dists, neighbors_idx):
        batch_size, points_num, _ = xyz_dists.shape
        anchor_points_idx = neighbors_idx[:, :, :self.anchor_points]  # (B, N, anchor_points)
        neighbors_idx = neighbors_idx[:, :, :self.neighbor_num]

        # (B, N, neighbor_num) -> (batch_size, points_num, neighbor_num, anchor_points)
        neighbors_anchor_points_idx = index_points(anchor_points_idx, neighbors_idx)
        anchor_points_idx = \
            anchor_points_idx[:, :, None, :, None].expand(
                -1, -1, self.neighbor_num, -1, self.anchor_points
            )
        neighbors_anchor_points_idx = \
            neighbors_anchor_points_idx[:, :, :, None, :].expand(
                -1, -1, -1, self.anchor_points, -1
            )

        # (B, N, neighbor_num, anchor_points(center points index), anchor_points(neighbor points index))
        relative_dists = index_points_dists(
            xyz_dists, anchor_points_idx, neighbors_anchor_points_idx
        )
        relative_dists = relative_dists.view(
            batch_size, points_num, self.neighbor_num, self.anchor_points ** 2
        )
        return relative_dists


class DeepRotationInvariantDistFea(RotationInvariantDistFea):  # deprecated
    def __init__(self,
                 neighbor_num: int,
                 anchor_points: int,
                 extra_intra_anchor_dists_chnls: int,
                 extra_relative_fea_chnls: int,
                 retain_xyz_dists=False):
        super(DeepRotationInvariantDistFea, self).__init__(
            neighbor_num,
            anchor_points,
            retain_xyz_dists
        )
        mlp_relative_fea_in_chnls = \
            (self.intra_anchor_dists_chnls + extra_intra_anchor_dists_chnls) * 2 + \
            self.inter_anchor_dists_chnls

        self.mlp_intra_dist = MLPBlock(
            self.intra_anchor_dists_chnls, extra_intra_anchor_dists_chnls,
            bn='nn.bn1d', act='leaky_relu(0.2)', skip_connection='concat'
        )
        self.mlp_relative_fea = MLPBlock(
            mlp_relative_fea_in_chnls, extra_relative_fea_chnls,
            bn='nn.bn1d', act='leaky_relu(0.2)', skip_connection='concat'
        )

        # assert extra_intra_anchor_dists_chnls >= self.intra_anchor_dists_chnls
        # assert extra_relative_fea_chnls >= self.inter_anchor_dists_chnls
        self.intra_anchor_dists_chnls += extra_intra_anchor_dists_chnls
        self.channels = mlp_relative_fea_in_chnls + extra_relative_fea_chnls

    def forward(self, xyz):
        intra_anchor_dists, inter_anchor_dists, neighbors_idx = \
            super(DeepRotationInvariantDistFea, self).forward(xyz, False)
        intra_anchor_fea = self.mlp_intra_dist(intra_anchor_dists)

        center_intra_anchor_fea = \
            intra_anchor_fea[:, :, None, :].expand(
                -1, -1, self.neighbor_num, -1
            )
        neighbor_intra_anchor_fea = index_points(
            intra_anchor_fea, neighbors_idx[:, :, :self.neighbor_num]
        )
        relative_feature = torch.cat(
            [center_intra_anchor_fea,
             neighbor_intra_anchor_fea,
             inter_anchor_dists], dim=3)

        relative_feature = self.mlp_relative_fea(relative_feature)
        return relative_feature, neighbors_idx


class TransitionDown(nn.Module):
    def __init__(self,
                 sample_method: str = 'uniform',
                 sample_rate: float = None,
                 nsample: int = None,
                 cache_sample_indexes: Optional[str] = None,
                 cache_sampled_xyz: bool = False,
                 cache_sampled_feature: bool = False):
        super(TransitionDown, self).__init__()
        assert (nsample is None) != (sample_rate is None)
        self.sample_method = sample_method
        self.sample_rate = sample_rate
        self.nsample = nsample
        self.cache_sample_indexes = cache_sample_indexes
        self.cache_sampled_xyz = cache_sampled_xyz
        self.cache_sampled_feature = cache_sampled_feature

    def forward(self, msg: PointLayerMessage):
        msg.raw_neighbors_feature = None
        sample_indexes = self.get_indexes(msg.xyz, msg.neighbors_idx)
        sampled_xyz = index_points(msg.xyz, sample_indexes)
        feature = index_points(msg.feature, sample_indexes)

        if self.cache_sample_indexes == 'downsample':
            msg.cached_sample_indexes.append(sample_indexes)
        elif self.cache_sample_indexes == 'upsample':  # nearest_interpolation
            msg.cached_sample_indexes.append(
                knn_points(msg.xyz, sampled_xyz, k=1, return_sorted=False).idx
            )
        else: assert self.cache_sample_indexes is None

        if self.cache_sampled_xyz:
            msg.cached_xyz.append(sampled_xyz)
        if self.cache_sampled_feature:
            msg.cached_feature.append(feature)
        msg.xyz = sampled_xyz
        msg.feature = feature
        msg.raw_neighbors_feature = None
        msg.neighbors_idx = None
        return msg

    def get_indexes(self, xyz, neighbors_idx):
        batch_size, points_num, _ = xyz.shape
        if self.nsample is not None:
            nsample = self.nsample
        else:
            nsample = int(self.sample_rate * points_num)

        if self.sample_method == 'uniform':
            sample_indexes = torch.multinomial(
                torch.ones((1, 1), device=xyz.device).expand(batch_size, points_num),
                nsample, replacement=False
            )
        elif self.sample_method == 'uniform_batch_unaware':  # for debug purpose
            sample_indexes = torch.multinomial(
                torch.ones((1,), device=xyz.device).expand(points_num),
                nsample, replacement=False
            )[None, :].expand(batch_size, -1)
        elif self.sample_method == 'inverse_knn_density':
            freqs = []  # (batch_size, points_num)
            for ni in neighbors_idx:
                # anchor_points is supposed to be smaller than neighbor_num
                # (points_num, )
                freqs.append(torch.maximum(
                    ni.reshape(-1).bincount(minlength=points_num),
                    torch.tensor([1], device=xyz.device)
                ))
            del neighbors_idx
            # (batch_size, nsample)
            sample_indexes = torch.multinomial(
                1 / torch.stack(freqs, dim=0), nsample, replacement=False
            )
        elif self.sample_method == 'fps':
            return farthest_point_sample(xyz, nsample)
        else:
            raise NotImplementedError
        return sample_indexes

    def __repr__(self):
        return f'TransitionDown({self.nsample}, {self.sample_rate}, {self.sample_method})'


class TransitionDownWithDistFea(TransitionDown):
    def __init__(self,
                 neighbor_fea_generator: RotationInvariantDistFea,
                 in_channels,
                 transition_fea_chnls,
                 out_channels,
                 sample_method: str = 'uniform',
                 sample_rate: float = None,
                 nsample: int = None,
                 cache_sample_indexes: Optional[str] = None,
                 cache_sampled_xyz: bool = False):
        super(TransitionDownWithDistFea, self).__init__(
            sample_method=sample_method,
            sample_rate=sample_rate,
            nsample=nsample,
            cache_sample_indexes=cache_sample_indexes,
            cache_sampled_xyz=cache_sampled_xyz)
        self.neighbor_fea_generator = neighbor_fea_generator
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transition_fea_chnls = transition_fea_chnls
        assert neighbor_fea_generator.retain_xyz_dists is True
        # assert self.in_channels >= self.neighbor_fea_generator.channels
        self.mlp_anchor_transition_fea = nn.Sequential(
            MLPBlock(self.neighbor_fea_generator.channels,
                     self.transition_fea_chnls,
                     bn='nn.bn1d', act='leaky_relu(0.2)'),
            MLPBlock(self.transition_fea_chnls,
                     self.transition_fea_chnls,
                     bn='nn.bn1d', act='leaky_relu(0.2)')
        )
        self.mlp_out = nn.Sequential(
            MLPBlock(self.in_channels + self.transition_fea_chnls,
                     self.out_channels,
                     bn='nn.bn1d', act='leaky_relu(0.2)')
        )

    def forward(self, msg: PointLayerMessage):
        batch_size, points_num, _ = msg.xyz.shape
        sample_indexes = self.get_indexes(msg.xyz, msg.neighbors_idx)
        nsample = sample_indexes.shape[1]

        intra_anchor_dists_before = msg.raw_neighbors_feature[
            :, :, 0, :self.neighbor_fea_generator.intra_anchor_dists_chnls
        ]
        msg.raw_neighbors_feature = None

        sampled_xyz = index_points(msg.xyz, sample_indexes)
        feature = index_points(msg.feature, sample_indexes)
        if self.cache_sample_indexes == 'downsample':
            msg.cached_sample_indexes.append(sample_indexes)
        elif self.cache_sample_indexes == 'upsample':
            msg.cached_sample_indexes.append(
                knn_points(msg.xyz, sampled_xyz, k=1, return_sorted=False).idx
            )
        else:
            assert self.cache_sample_indexes is None
        if self.cache_sampled_xyz:
            msg.cached_xyz.append(sampled_xyz)

        # After sampling, all the anchors have to be redefined due to loss of points,
        # which introduces ambiguity of the relative position and gesture between
        # new and old anchors.
        # We manually introduce the distance information between the two and intra
        # themselves before further aggregating neighborhood anchors information.
        #
        # Both info before and after sampling are needed here, because we have to
        # gather distances info based on point-wise distances before sampling
        # while relying on neighborhood-based info after sampling.
        #
        # After gathering intra and inter anchors distances, mlps are performed on
        # the sampled points with those distances concatenated.
        intra_anchor_dists_before = index_points(intra_anchor_dists_before, sample_indexes)
        xyz_dists_before = self.neighbor_fea_generator.xyz_dists

        raw_neighbors_feature_after, neighbors_idx_after = \
            self.neighbor_fea_generator(sampled_xyz)
        intra_anchor_dists_after = raw_neighbors_feature_after[
            :, :, 0, :self.neighbor_fea_generator.intra_anchor_dists_chnls
        ]

        anchor_points_idx_before = index_points(
            msg.neighbors_idx[:, :, :self.neighbor_fea_generator.anchor_points],
            sample_indexes
        )
        anchor_points_idx_before = anchor_points_idx_before[:, :, :, None].expand(
            -1, -1, -1, self.neighbor_fea_generator.anchor_points
        )

        anchor_points_idx_after = neighbors_idx_after[
            :, :, :self.neighbor_fea_generator.anchor_points
        ]
        # mapping indexes in anchor_points_idx_after back to the version before sampling
        anchor_points_idx_after = sample_indexes[..., None].expand(
            -1, -1, self.neighbor_fea_generator.anchor_points
        ).gather(dim=1, index=anchor_points_idx_after)
        anchor_points_idx_after = anchor_points_idx_after[:, :, None, :].expand(
            -1, -1, self.neighbor_fea_generator.anchor_points, -1
        )

        inter_anchor_dists = index_points_dists(
            xyz_dists_before,
            anchor_points_idx_before,
            anchor_points_idx_after
        ).reshape(batch_size, nsample, self.neighbor_fea_generator.anchor_points ** 2)
        anchor_dist_feature = torch.cat(
            [intra_anchor_dists_before,
             intra_anchor_dists_after,
             inter_anchor_dists], dim=2
        )

        if hasattr(self.neighbor_fea_generator, 'mlp_relative_fea'):
            anchor_dist_feature = self.neighbor_fea_generator.mlp_relative_fea(
                anchor_dist_feature
            )
        anchor_transition_fea = self.mlp_anchor_transition_fea(anchor_dist_feature)
        feature = torch.cat([feature, anchor_transition_fea], dim=2)
        feature = self.mlp_out(feature)
        msg.xyz = sampled_xyz
        msg.feature = feature
        msg.raw_neighbors_feature = raw_neighbors_feature_after
        msg.neighbors_idx = neighbors_idx_after
        return msg

    def __repr__(self):
        return f'TransitionDownWithDistFea(' \
               f'neighbor_fea_generator.channels={self.neighbor_fea_generator.channels}, ' \
               f'in_channels={self.in_channels}, ' \
               f'out_channels={self.out_channels}, ' \
               f'nsample={self.nsample}, ' \
               f'sample_rate={self.sample_rate}, ' \
               f'sample_method="{self.sample_method})"'


class LocalFeatureAggregation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 neighbor_feature_generator: NeighborFeatureGenerator,
                 raw_neighbor_fea_out_chnls: int,
                 out_channels: int,
                 cache_out_feature: bool = False):
        super(LocalFeatureAggregation, self).__init__()
        self.neighbor_fea_chnls = in_channels + raw_neighbor_fea_out_chnls
        self.mlp_raw_neighbor_fea = MLPBlock(
            neighbor_feature_generator.channels,
            raw_neighbor_fea_out_chnls,
            bn='nn.bn1d', act='leaky_relu(0.2)'
        )
        self.mlp_neighbor_fea = MLPBlock(
            self.neighbor_fea_chnls, self.neighbor_fea_chnls,
            bn='nn.bn1d', act='leaky_relu(0.2)'
        )
        self.mlp_attn = nn.Linear(
            self.neighbor_fea_chnls,
            self.neighbor_fea_chnls, bias=False
        )
        self.mlp_out = MLPBlock(
            self.neighbor_fea_chnls, out_channels,
            bn='nn.bn1d', act=None
        )
        self.mlp_shortcut = MLPBlock(
            in_channels, out_channels,
            'nn.bn1d', None) if in_channels != 0 else None
        self.neighbor_feature_generator = neighbor_feature_generator
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.raw_neighbor_fea_out_chnls = raw_neighbor_fea_out_chnls
        self.cache_out_feature = cache_out_feature

    def forward(self, msg: PointLayerMessage):
        # There are three typical situations:
        #   1. this layer is the first layer of the model. All the rest value will be calculated using xyz
        #   and returned.
        #   2. this layer is after a sampling layer, which is supposed to be the same as situation 1.
        #   To do this, a sampling layer should return raw_relative_feature and neighbors_idx as None.
        #   3. this layer is a normal layer in the model. raw_relative_feature and neighbors_idx from last layer
        #   will be directly used.
        if msg.raw_neighbors_feature is None or msg.neighbors_idx is None:
            msg.raw_neighbors_feature, msg.neighbors_idx = self.neighbor_feature_generator(msg.xyz)
        raw_neighbors_feature_mlp = self.mlp_raw_neighbor_fea(msg.raw_neighbors_feature)

        if self.in_channels != 0:
            # the slice below is necessary in case that RotationInvariantDistFea is used
            # and anchor_points > neighbor_num
            neighbors_feature = index_points(
                msg.feature,
                msg.neighbors_idx[:, :, :self.neighbor_feature_generator.neighbor_num]
            )
            neighbors_feature = torch.cat(
                [neighbors_feature,
                 raw_neighbors_feature_mlp], dim=3
            )
            feature = self.mlp_neighbor_fea(neighbors_feature)
            feature = self.attn_pooling(feature)
            feature = F.leaky_relu(
                self.mlp_shortcut(msg.feature) + self.mlp_out(feature),
                negative_slope=0.2
            )
        else:
            assert msg.feature is None and msg.cached_feature == []
            neighbors_feature = raw_neighbors_feature_mlp
            feature = self.mlp_neighbor_fea(neighbors_feature)
            feature = self.attn_pooling(feature)
            feature = F.leaky_relu(
                self.mlp_out(feature), negative_slope=0.2
            )

        msg.feature = feature
        if self.cache_out_feature: msg.cached_feature.append(feature)  # be careful about in-place operations
        return msg

    def attn_pooling(self, feature):
        attn = F.softmax(self.mlp_attn(feature), dim=2)
        feature = attn * feature
        feature = torch.sum(feature, dim=2)
        return feature

    def __repr__(self):
        return f'LFA(in_channels={self.in_channels}, ' \
               f'raw_neighbor_fea_out_chnls={self.raw_neighbor_fea_out_chnls}, ' \
               f'neighbor_fea_chnls={self.neighbor_fea_chnls}, ' \
               f'out_channels={self.out_channels})'


def transformer_block_t():
    input_xyz = torch.rand(4, 100, 3)
    input_feature = torch.rand(4, 100, 32)
    transformer_blocks = nn.Sequential(
        TransformerBlock(d_in=32, d_model=64, nneighbor=16, cache_out_feature=True),
        TransformerBlock(d_in=64, d_model=128, nneighbor=16, cache_out_feature=True),
        TransitionDown('uniform', 0.5, cache_sample_indexes='upsample'),
        TransformerBlock(d_in=128, d_model=128, nneighbor=16, cache_out_feature=True),
        TransformerBlock(d_in=128, d_model=256, nneighbor=16, cache_out_feature=True),
        TransitionDown('uniform', 0.5, cache_sample_indexes='upsample'))
    out = transformer_blocks(PointLayerMessage(xyz=input_xyz, feature=input_feature))
    out.feature.sum().backward()
    print('Done')


def lfa_test_1():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            neighbor_fea_generator = RandLANeighborFea(16)
            self.layers = nn.Sequential(
                LocalFeatureAggregation(3, neighbor_fea_generator, 16, 32),
                LocalFeatureAggregation(32, neighbor_fea_generator, 32, 64,
                                        cache_out_feature=True),
                TransitionDown('uniform', 0.5, cache_sample_indexes='upsample'),
                LocalFeatureAggregation(64, neighbor_fea_generator, 64, 128),
                LocalFeatureAggregation(128, neighbor_fea_generator, 64, 128,
                                        cache_out_feature=True),
                TransitionDown('uniform', 0.5))

        def forward(self, x):
            out = self.layers(PointLayerMessage(xyz=x, feature=x))
            return out.feature.sum()

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
                LocalFeatureAggregation(32, neighbor_fea_generator, 32, 64,
                                        cache_out_feature=True),

                TransitionDownWithDistFea(
                    neighbor_fea_generator, 64, 32, 64, 'uniform', 0.5,
                    cache_sample_indexes='upsample'),

                LocalFeatureAggregation(64, neighbor_fea_generator, 64, 128),
                LocalFeatureAggregation(128, neighbor_fea_generator, 64, 128,
                                        cache_out_feature=True),

                TransitionDownWithDistFea(
                    neighbor_fea_generator, 128, 64, 128, 'uniform', 0.5,
                    cache_sample_indexes='upsample')
            )

        def forward(self, x):
            out = self.layers(PointLayerMessage(xyz=x))
            return out.feature.sum()

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
