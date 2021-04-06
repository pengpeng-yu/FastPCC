from lib.pointnet_utils import index_points
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, d_in, d_model, nneighbor, d_out=None) -> None:
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

        self.fc1 = nn.Linear(d_in, d_model)
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

        if d_out is None: d_out = d_in
        self.fc2 = nn.Linear(d_model, d_out)
        self.shortout_fc = nn.Linear(d_in, d_out)
        
    # xyz: b, n, 3, features: b, n, d_in
    def forward(self, xyz, features, require_attn=False):
        knn_idx = torch.cdist(xyz, xyz).topk(self.k, dim=-1, largest=False, sorted=False)[1]  # knn_idx b, n, k  # TODO: recompute knn_idx to avoid duplicated computation
        knn_xyz = index_points(xyz, knn_idx)  # knn_xyz: b, n, k, 3
        pos_enc = self.fc_delta(xyz[:, :, None, :] - knn_xyz)  # pos_enc: b, n, k, d_model  # TODO: pos_encoding

        shortcut = features
        features = self.fc1(features)  # features: b, n, d_model
        knn_features = index_points(features, knn_idx)  # knn_features: b, n, k, d_model
        query, key, value = self.w_qs(features), self.w_ks(knn_features), self.w_vs(knn_features)
        # query: b, n, d_model   key, value: b, n, k, d_model

        attn = self.fc_gamma(query[:, :, None, :] - key + pos_enc)  # attn: b, n, k, d_model
        attn = F.softmax(attn / np.sqrt(key.size(-1)), dim=-2)  # attn: b, n, k, d_model
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, value + pos_enc)  # (attn * (value + pos_enc)).sum(dim=2) res: b, n, d_model
        res = self.fc2(res) + self.shortout_fc(shortcut)  # res: b, n, d_out
        if require_attn:
            return res, attn  # res: b, n, d_out  attn: b, n, k, d_model
        else:
            return res


def main_t():
    input_xyz = torch.rand(16, 1024, 3)  # B, N, C
    input_feature = torch.rand(16, 1024, 32)  # B, N, C
    transfomer_block = TransformerBlock(d_in=32, d_model=512, nneighbor=16, d_out=64)
    output_fea = transfomer_block(input_xyz, input_feature)
    print('Done')


if __name__ == '__main__':
    main_t()