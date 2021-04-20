from lib.pointnet_utils import index_points
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, d_in, d_model, nneighbor, return_idx, d_out=None) -> None:
        """
        d_in: input feature channels
        d_model: internal channels
        d_out: output channels (default: d_model)
        """
        super().__init__()
        self.nneighbor = nneighbor
        self.return_idx = return_idx
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
        
    # xyz: b, n, 3, features: b, n, d_in
    def forward(self, x):
        xyz, feature, relative_knn_xyz, knn_idx = x
        if relative_knn_xyz is None or knn_idx is None:
            knn_idx = torch.cdist(xyz, xyz).topk(self.nneighbor, dim=-1, largest=False, sorted=False)[1]
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
        
        feature = torch.einsum('bmnf,bmnf->bmf', attn, value + pos_enc)  # (attn * (value + pos_enc)).sum(dim=2) feature: b, n, d_model
        feature = self.fc2(feature) + self.shortout_fc(ori_features)  # feature: b, n, d_out

        if self.return_idx:
            return xyz, feature, relative_knn_xyz, knn_idx
        else:
            return xyz, feature, None, None


def main_t():
    input_xyz = torch.rand(16, 1024, 3)
    input_feature = torch.rand(16, 1024, 32)
    transfomer_block = TransformerBlock(d_in=32, d_model=512, nneighbor=16)
    output_fea = transfomer_block(input_xyz, input_feature)
    print('Done')


if __name__ == '__main__':
    main_t()