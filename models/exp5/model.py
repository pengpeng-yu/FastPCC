import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import loss_function
from lib.pointnet_utils import index_points
from lib.torch_utils import MLPBlock
from models.exp5.model_config import ModelConfig


class LocalFeatureAggregation(nn.Module):
    def __init__(self, in_channels, neighbors_num, relative_fea_channels, out_channels, return_idx):
        super(LocalFeatureAggregation, self).__init__()
        self.mlp_relative_fea = MLPBlock(10, relative_fea_channels, activation='leaky_relu(0.2)', batchnorm='nn.bn1d')
        self.mlp_attn = nn.Linear(in_channels + relative_fea_channels, in_channels + relative_fea_channels, bias=False)
        self.mlp_out = MLPBlock(in_channels + relative_fea_channels, out_channels, activation=None, batchnorm='nn.bn1d')
        self.mlp_shortcut = MLPBlock(in_channels, out_channels, activation=None, batchnorm='nn.bn1d')

        self.in_channels = in_channels
        self.neighbors_num = neighbors_num
        self.out_channels = out_channels
        self.relative_fea_channels = relative_fea_channels
        self.return_idx = return_idx

    def forward(self, x):
        xyz, feature, raw_relative_feature, neighbors_idx = x
        batch_size, n_points, _ = feature.shape
        ori_feature = feature

        if raw_relative_feature is None or neighbors_idx is None:
            feature, raw_relative_feature, neighbors_idx = self.gather_neighbors(xyz, feature)
            torch.cuda.empty_cache()
        else:
            feature = index_points(feature, neighbors_idx)

        relative_feature = self.mlp_relative_fea(raw_relative_feature.reshape(batch_size, -1, 10)).reshape(batch_size, n_points, self.neighbors_num, -1)
        feature = torch.cat([feature, relative_feature], dim=3)  # channels: in_channels + relative_fea_channels
        feature = self.attn_pooling(feature)
        feature = F.leaky_relu(self.mlp_shortcut(ori_feature) + self.mlp_out(feature), negative_slope=0.2)

        if self.return_idx:
            return xyz, feature, raw_relative_feature, neighbors_idx
        else:
            return xyz, feature, None, None

    def gather_neighbors(self, xyz, feature):
        dists = torch.cdist(xyz, xyz, compute_mode='donot_use_mm_for_euclid_dist')
        relative_dists, neighbors_idx = dists.topk(self.neighbors_num, dim=-1, largest=False, sorted=True)
        feature = index_points(feature, neighbors_idx)
        neighbors_xyz = index_points(xyz, neighbors_idx)

        expanded_xyz = xyz[:, :, None, :].expand(-1, -1, neighbors_xyz.shape[2], -1)
        relative_xyz = expanded_xyz - neighbors_xyz
        relative_dists = relative_dists[:, :, :, None]
        relative_feature = torch.cat([relative_dists, relative_xyz, expanded_xyz, neighbors_xyz], dim=-1)

        return feature, relative_feature, neighbors_idx

    def attn_pooling(self, feature):
        attn = F.softmax(self.mlp_attn(feature), dim=2)
        feature = attn * feature
        feature = torch.sum(feature, dim=2)
        return feature


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # TODO: should I use Gumbel-Softmax trick?
        self.encoder_layers = [LocalFeatureAggregation(3, cfg.neighbor_num, 16, 24, True),
                               LocalFeatureAggregation(24, cfg.neighbor_num, 16, 32, True),
                               LocalFeatureAggregation(32, cfg.neighbor_num, 16, 48, True),
                               LocalFeatureAggregation(48, cfg.neighbor_num, 24, 48, True),
                               LocalFeatureAggregation(48, cfg.neighbor_num, 24, 64, True),
                               LocalFeatureAggregation(64, cfg.neighbor_num, 24, 64, True),
                               LocalFeatureAggregation(64, cfg.neighbor_num, 24, 128, True),
                               LocalFeatureAggregation(128, cfg.neighbor_num, 32, 128, True),]
        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        self.encoded_points_num = cfg.input_points_num

        self.mlp_enc_out = nn.Sequential(MLPBlock(128, 128, activation='leaky_relu(0.2)', batchnorm='nn.bn1d'),
                                         MLPBlock(128, 32 * 3, activation=None, batchnorm='nn.bn1d'))

        self.decoder_layers = [LocalFeatureAggregation(32 * 3, cfg.neighbor_num, 128, 128, True),
                               LocalFeatureAggregation(128, cfg.neighbor_num, 128, 128, False)]
        self.decoder_layers = nn.Sequential(*self.decoder_layers)

        self.mlp_dec_out = nn.Sequential(MLPBlock(128, 128, activation='leaky_relu(0.2)', batchnorm='nn.bn1d'),
                                         MLPBlock(128, 3, activation=None, batchnorm='nn.bn1d'))

    def forward(self, fea):
        if self.training: ori_fea = fea
        batch_size = fea.shape[0]
        xyz = fea[..., :3]  # B, N, C

        xyz, fea = self.encoder_layers((xyz, fea, None, None))[:2]
        fea = self.mlp_enc_out(fea)
        fea.sigmoid_()

        if self.training:
            with torch.no_grad():
                label = torch.round(fea)
                quantize_diff = label - fea

            quantize_loss = torch.nn.functional.binary_cross_entropy(fea, label) * self.cfg.quantize_loss_factor
            balance_loss = torch.mean(fea) * self.cfg.balance_loss_factor

            fea = fea + quantize_diff
            fea = self.decoder_layers((xyz, fea, None, None))[1]  # TODO: eliminate xyz here
            fea = self.mlp_dec_out(fea)
            fea = fea.reshape(batch_size, self.cfg.input_points_num, 3)

            reconstruct_loss = loss_function.chamfer_loss(fea, ori_fea)
            loss = reconstruct_loss + quantize_loss + balance_loss

            return {'quantize_loss': quantize_loss.detach().cpu().item(),
                    'balance_loss': balance_loss.detach().cpu().item(),
                    'reconstruct_loss': reconstruct_loss.detach().cpu().item(),
                    'loss': loss}

        else:
            round_fea = torch.round(fea)
            fea = self.decoder_layers((xyz, round_fea, None, None))[1]
            fea = self.mlp_dec_out(fea)
            fea = fea.reshape(batch_size, self.cfg.input_points_num, 3)

            return {'round_fea': round_fea,
                    'decoder_output': fea}


def main_t():
    torch.cuda.set_device('cuda:0')
    model = PointCompressor(ModelConfig()).cuda()
    model.train()
    batch_points = torch.rand(1, 8192, 3).cuda()
    out =  model(batch_points)
    print('Done')

if __name__ == '__main__':
    main_t()