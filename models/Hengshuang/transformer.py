from lib.pointnet_utils import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, nneighbor) -> None:
        """
        d_points: input channels num
        d_model: block channels num
        """
        super().__init__()
        self.k = nneighbor
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

        self.fc1 = nn.Linear(d_points, d_model)
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

        self.fc2 = nn.Linear(d_model, d_points)
        
    # xyz: b, n, 3, features: b, n, f
    def forward(self, xyz, features, require_attn=False):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # knn_idx b, n, k
        knn_xyz = index_points(xyz, knn_idx)  # knn_xyz: b, n, k, 3
        pos_enc = self.fc_delta(xyz[:, :, None, :] - knn_xyz)  # pos_enc: b, n, k, d_model

        shortcut = features
        features = self.fc1(features)  # features: b, n, d_model
        knn_features = index_points(features, knn_idx)  # knn_features: b, n, k, d_model
        query, key, value = self.w_qs(features), self.w_ks(knn_features), self.w_vs(knn_features)
        # query: b, n, d_model   key, value: b, n, k, d_model

        attn = self.fc_gamma(query[:, :, None, :] - key + pos_enc)  # attn: b, n, k, d_model
        attn = F.softmax(attn / np.sqrt(key.size(-1)), dim=-2)  # attn: b, n, k, d_model
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, value + pos_enc)  # (attn * (value + pos_enc)).sum(dim=2) res: b, n, d_model
        res = self.fc2(res) + shortcut  # res: b, n, f
        if require_attn:
            return res, attn  # res: b, n, f  attn: b, n, k, d_model
        else:
            return res


def main_t():
    input_xyz = torch.rand(16, 1024, 3)  # B, N, C
    input_feature = torch.rand(16, 1024, 32)  # B, N, C
    transfomer_block = TransformerBlock(d_points=32, d_model=512, nneighbor=16)
    output = transfomer_block(input_xyz, input_feature)
    points_fea = output[0]
    assert points_fea.shape == input_feature.shape
    print('Done')


if __name__ == '__main__':
    main_t()