import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointnet_utils import index_points, index_points_dists
from lib.torch_utils import MLPBlock
from models.classification.baseline import Config


class LFA(nn.Module):
    def __init__(self, in_channels, neighbors_num, relative_fea_chnls, out_channels, return_neighbor_based_fea=True,
                 anchor_points=4):
        super(LFA, self).__init__()
        if anchor_points == 4:
            ori_relative_fea_chnls = 6 + 6 + 4
        else:
            raise NotImplementedError

        self.mlp_relative_fea = MLPBlock(ori_relative_fea_chnls, relative_fea_chnls, 'leaky_relu(0.2)', 'nn.bn1d')
        self.mlp_attn = nn.Linear(in_channels + relative_fea_chnls, in_channels + relative_fea_chnls, bias=False)
        self.mlp_out = MLPBlock(in_channels + relative_fea_chnls, out_channels, None, 'nn.bn1d')
        self.mlp_shortcut = MLPBlock(in_channels, out_channels, None, 'nn.bn1d')

        self.in_channels = in_channels
        self.neighbors_num = neighbors_num
        self.out_channels = out_channels
        self.relative_fea_chnls = relative_fea_chnls
        # return_neighbor_based_fea should be True if next layer is not a sampling layer
        self.return_neighbor_based_fea = return_neighbor_based_fea
        self.anchor_points = anchor_points

    def forward(self, x):
        # There are three typical situations:
        #   1. this layer is the first layer of the model. All the rest value will be calculated using xyz
        #   and returned if self.return_neighbor_based_fea is True.
        #   2. this layer is after a sampling layer, which is supposed to be the same as 1.
        #   To do this, a sampling layer should return raw_relative_feature and neighbors_idx as None,
        #   or, the layer before sampling should has return_neighbor_based_fea == False.
        #   3. this layer is an normal layer in the model. raw_relative_feature and neighbors_idx from last layer
        #   will be directly used.
        # This format of inputs and outputs is aimed to simplify the forward function of top-level module.

        xyz, feature, raw_relative_feature, neighbors_idx = x
        batch_size, n_points, _ = feature.shape
        ori_feature = feature

        if raw_relative_feature is None or neighbors_idx is None:
            xyz_dists = torch.cdist(xyz, xyz, compute_mode='donot_use_mm_for_euclid_dist')
            neighbors_dists, neighbors_idx = xyz_dists.topk(max(self.neighbors_num, self.anchor_points),
                                                            dim=2, largest=False, sorted=True)
            feature, raw_relative_feature = self.gather_neighbors(xyz_dists, feature, neighbors_dists, neighbors_idx)
            del xyz_dists
            torch.cuda.empty_cache()
        else:
            feature = index_points(feature, neighbors_idx)

        relative_feature = raw_relative_feature.reshape(batch_size, n_points * self.neighbors_num, -1)
        relative_feature = self.mlp_relative_fea(relative_feature)
        relative_feature = relative_feature.reshape(batch_size, n_points, self.neighbors_num, -1)
        feature = torch.cat([feature, relative_feature], dim=3)
        feature = self.attn_pooling(feature)
        feature = F.leaky_relu(self.mlp_shortcut(ori_feature) + self.mlp_out(feature), negative_slope=0.2)

        if not self.return_neighbor_based_fea:
            raw_relative_feature = neighbors_idx = None
            torch.cuda.empty_cache()
        return xyz, feature, raw_relative_feature, neighbors_idx

    def gather_neighbors(self, xyz_dists, feature, neighbors_dists, neighbors_idx):
        # xyz_dists is still necessary here and can not be replaced by neighbors_dists
        # (B, N, self.neighbor_num, self.channels)
        feature = index_points(feature, neighbors_idx[:, :, :self.neighbors_num])

        # (B, N, 6 if self.anchor_points == 4)  TODO: anchors will change if sampling is performed. Is this rational?
        intra_anchor_dists = self.gen_intra_anchor_dists(xyz_dists, neighbors_dists, neighbors_idx[:, :, :self.anchor_points])
        # (B, N, self.neighbor_num, self.anchor_points)
        inter_anchor_dists = self.gen_inter_anchor_dists(xyz_dists, neighbors_idx[:, :, :self.neighbors_num])
        # (B, N, self.neighbor_num, 6)
        center_intra_anchor_dists = intra_anchor_dists[:, :, None, :].expand(-1, -1, self.neighbors_num, -1)
        # (B, N, self.neighbor_num, 6)
        nerigbor_intra_anchor_dists = index_points(intra_anchor_dists, neighbors_idx[:, :, :self.neighbors_num])
        # (B, N, self.neighbor_num, 6 + 6 + self.anchor_points)
        relative_feature = torch.cat([center_intra_anchor_dists, nerigbor_intra_anchor_dists, inter_anchor_dists], dim=3)

        return feature, relative_feature

    def gen_intra_anchor_dists(self, xyz_dists, neighbors_dists, neighbors_idx):
        if self.anchor_points == 4:
            sub_anchor_dists1 = index_points_dists(xyz_dists, neighbors_idx[:, :, 1:2], neighbors_idx[:, :, 2:3])
            sub_anchor_dists2 = index_points_dists(xyz_dists, neighbors_idx[:, :, 2:3], neighbors_idx[:, :, 3:4])
            sub_anchor_dists3 = index_points_dists(xyz_dists, neighbors_idx[:, :, 3:4], neighbors_idx[:, :, 1:2])

            intra_anchor_dists = torch.cat([neighbors_dists[:, :, 1:],
                                            sub_anchor_dists1, sub_anchor_dists2, sub_anchor_dists3], dim=2)
            return intra_anchor_dists

        else:
            raise NotImplementedError

    def gen_inter_anchor_dists(self, xyz_dists, neighbors_idx):
        anchor_points_idx = neighbors_idx[:, :, :self.anchor_points]
        neighbors_anchor_points_idx = index_points(anchor_points_idx, neighbors_idx)

        anchor_points_idx = anchor_points_idx[:, :, None, :].expand(-1, -1, self.neighbors_num, -1)
        relative_dists = index_points_dists(xyz_dists, anchor_points_idx, neighbors_anchor_points_idx)
        return relative_dists

    def attn_pooling(self, feature):
        attn = F.softmax(self.mlp_attn(feature), dim=2)
        feature = attn * feature
        feature = torch.sum(feature, dim=2)
        return feature


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super(Model, self).__init__()
        self.cfg = cfg
        self.layers = nn.Sequential(LFA(3, cfg.neighbor_num, 32, 64),
                                    LFA(64, cfg.neighbor_num, 64, 128),
                                    LFA(128, cfg.neighbor_num, 64, 256),
                                    LFA(256, cfg.neighbor_num, 128, 512),
                                    LFA(512, cfg.neighbor_num, 256, 1024))
        self.head = nn.Linear(1024, cfg.classes_num, bias=True)
        self.log_pred_res('init')

    def log_pred_res(self, mode, pred=None, target=None):
        if mode == 'init':
            # 0: samples_num, 1: correct_num, 2: wrong_num, 3: correct_rate
            pred_res = torch.zeros((self.cfg.classes_num, 4), dtype=torch.float32)
            self.register_buffer('pred_res', pred_res)

        elif mode == 'reset':
            self.pred_res[...] = 0

        elif mode == 'log':
            assert not self.training
            assert pred is not None and target is not None
            self.pred_res[:, 0] += torch.bincount(target, minlength=self.cfg.classes_num)
            self.pred_res[:, 1] += torch.bincount(target[pred == target], minlength=self.cfg.classes_num)

        elif mode == 'show':
            self.pred_res[:, 2] = self.pred_res[:, 0] - self.pred_res[:, 1]
            self.pred_res[:, 3] = self.pred_res[:, 1] / self.pred_res[:, 0]
            samples_num = self.pred_res[:, 0].sum().cpu().item()
            correct_num = self.pred_res[:, 1].sum().cpu().item()
            return {'samples_num': samples_num,
                    'correct_num': correct_num,
                    'accuracy': correct_num / samples_num,
                    'mean_accuracy': self.pred_res[:, 3].mean().cpu().item(),
                    'class_info': self.pred_res.clone().cpu()}
        else:
            raise NotImplementedError

    def forward(self, x):
        xyz, target = x
        feature = self.layers((xyz, xyz, None, None))[1]
        feature = torch.max(feature, dim=1).values
        feature = self.head(feature)

        if self.training:
            loss = nn.functional.cross_entropy(feature, target)
            return {'loss': loss,
                    'ce_loss': loss.detach().cpu().item()}

        else:
            pred = torch.argmax(feature, dim=1)
            if target is not None:
                self.log_pred_res('log', pred, target)
            return {'pred': pred.detach().cpu()}


def main_t():
    cfg = Config()
    cfg.input_points_num = 8192
    model = Model(cfg).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])
    xyz = torch.rand(4, cfg.input_points_num, 3).cuda()
    target = torch.randint(0, 40, (4,)).cuda()

    y = model((xyz, target))
    y['loss'].backward()

    model.eval()
    model.module.log_pred_res('reset')
    _ = model((xyz, target))
    test_res = model.module.log_pred_res('show')
    print('Done')

if __name__ == '__main__':
    main_t()
