import torch
import torch.nn as nn
import torch.nn.functional as F
import compressai
from compressai.models.utils import update_registered_buffers

from lib import loss_function
from lib.pointnet_utils import index_points
from lib.torch_utils import MLPBlock
from models.exp4.model_config import ModelConfig


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
        xyz, feature, ori_relative_feature, neighbors_idx = x
        batch_size, n_points, _ = feature.shape
        ori_feature = feature

        if ori_relative_feature is None or neighbors_idx is None:
            feature, ori_relative_feature, neighbors_idx = self.gather_neighbors(xyz, feature)
        else:
            feature = index_points(feature, neighbors_idx)

        relative_feature = self.mlp_relative_fea(ori_relative_feature.reshape(batch_size, -1, 10)).reshape(batch_size, n_points, self.neighbors_num, -1)
        feature = torch.cat([feature, relative_feature], dim=-1)  # channels: in_channels + relative_fea_channels
        feature = self.attn_pooling(feature)
        feature = F.leaky_relu(self.mlp_shortcut(ori_feature) + self.mlp_out(feature), negative_slope=0.2)

        if self.return_idx:
            return xyz, feature, ori_relative_feature, neighbors_idx
        else: return xyz, feature, None, None

    def gather_neighbors(self, xyz, feature):
        dists = torch.cdist(xyz, xyz)
        relative_dists, neighbors_idx = dists.topk(self.neighbors_num, dim=-1, largest=False, sorted=False)
        feature = index_points(feature, neighbors_idx)
        neighbors_xyz = index_points(xyz, neighbors_idx)

        expanded_xyz = xyz[:, :, None].expand(-1, -1, neighbors_xyz.shape[2], -1)
        relative_xyz = expanded_xyz - neighbors_xyz
        relative_dists = relative_dists[:, :, :, None]
        relative_feature = torch.cat([relative_dists, relative_xyz, expanded_xyz, neighbors_xyz], dim=-1)

        return feature, relative_feature, neighbors_idx

    def attn_pooling(self, feature):
        attn = F.softmax(self.mlp_attn(feature), dim=2)
        feature = attn * feature
        feature = torch.sum(feature, dim=2)
        return feature


class TransitionDown(nn.Module):
    def __init__(self, nsample, sample_method):
        super(TransitionDown, self).__init__()
        self.nsample = nsample
        self.sample_method = sample_method

    def forward(self, x):
        xyz, points_fea, *args = x
        if self.sample_method == 'uniform':
            return xyz[:, : self.nsample], points_fea[:, : self.nsample], *args
        else:
            raise NotImplementedError

    def __repr__(self):
        return f'TransitionDown({self.nsample}, {self.sample_method})'


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder_layers = [LocalFeatureAggregation(3, cfg.neighbor_num, 16, 24, True),
                               LocalFeatureAggregation(24, cfg.neighbor_num, 16, 32, True),
                               LocalFeatureAggregation(32, cfg.neighbor_num, 16, 48, True),
                               LocalFeatureAggregation(48, cfg.neighbor_num, 24, 48, True),
                               LocalFeatureAggregation(48, cfg.neighbor_num, 24, 64, True),
                               LocalFeatureAggregation(64, cfg.neighbor_num, 24, 64, False),
                               TransitionDown(cfg.input_points_num // 2, 'uniform'),
                               LocalFeatureAggregation(64, cfg.neighbor_num, 24, 128, True),
                               LocalFeatureAggregation(128, cfg.neighbor_num, 32, 128, False),
                               TransitionDown(cfg.input_points_num // 4, 'uniform'),
                               LocalFeatureAggregation(128, cfg.neighbor_num, 32, 256, True),
                               LocalFeatureAggregation(256, cfg.neighbor_num, 32, 256, False),
                               TransitionDown(cfg.input_points_num // 8, 'uniform'),
                               LocalFeatureAggregation(256, cfg.neighbor_num, 32, 512, True),
                               LocalFeatureAggregation(512, cfg.neighbor_num, 32, 512, False),
                               TransitionDown(cfg.input_points_num // 16, 'uniform'),
                               LocalFeatureAggregation(512, cfg.neighbor_num, 64, 1024, True),
                               LocalFeatureAggregation(1024, cfg.neighbor_num, 128, 1024, False),
                               TransitionDown(cfg.input_points_num // 32, 'uniform'),
                               ]
        self.encoded_points_num = cfg.input_points_num // 32
        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        self.mlp_enc_out = nn.Sequential(MLPBlock(1024, 1024, activation='leaky_relu(0.2)', batchnorm='nn.bn1d'),
                                         MLPBlock(1024, 1024, activation=None, batchnorm='nn.bn1d'))

        self.entropy_bottleneck = compressai.entropy_models.EntropyBottleneck(1024)

        self.decoder_layers = [LocalFeatureAggregation(1024, cfg.neighbor_num, 128, 1024, True),
                               LocalFeatureAggregation(1024, cfg.neighbor_num, 128, 1024, False)]
        self.decoder_layers = nn.Sequential(*self.decoder_layers)
        self.mlp_dec_out = nn.Sequential(MLPBlock(1024, 256, activation='leaky_relu(0.2)', batchnorm='nn.bn1d'),
                                         MLPBlock(256, 96, activation=None, batchnorm='nn.bn1d'))

        self.init_weights()

    def forward(self, fea):
        if self.training: ori_fea = fea
        batch_size = fea.shape[0]
        xyz = fea[..., :3]  # B, N, C
        # encode
        xyz, fea = self.encoder_layers((xyz, fea, None, None))[:2]
        fea = self.mlp_enc_out(fea)

        if self.training:
            fea, likelihoods = self.entropy_bottleneck(fea.permute(0, 2, 1).unsqueeze(3).contiguous())
            fea = fea.squeeze(3).permute(0, 2, 1).contiguous()
            likelihoods = likelihoods.squeeze(3).permute(0, 2, 1).contiguous()
            fea = self.decoder_layers((xyz, fea, None, None))[1]
            fea = self.mlp_dec_out(fea)
            fea = fea.reshape(batch_size, self.cfg.input_points_num, 3)

            bpp_loss = torch.log2(likelihoods).sum() * (-self.cfg.bpp_loss_factor / (ori_fea.shape[0] * ori_fea.shape[1]))
            reconstruct_loss = loss_function.chamfer_loss(fea, ori_fea)
            aux_loss = self.entropy_bottleneck.loss() * self.cfg.aux_loss_factor
            loss = reconstruct_loss + bpp_loss + aux_loss
            return {'aux_loss': aux_loss.detach().cpu().item(),
                    'bpp_loss': bpp_loss.detach().cpu().item(),
                    'reconstruct_loss': reconstruct_loss.detach().cpu().item(),
                    'loss': loss}
        else:
            compressed_strings = self.entropy_bottleneck_compress(fea)
            decompressed_tensors = self.entropy_bottleneck_decompress(compressed_strings)
            fea = self.decoder_layers((xyz, decompressed_tensors, None, None))[1]
            fea = self.mlp_dec_out(fea)
            fea = fea.reshape(batch_size, self.cfg.input_points_num, 3)

            return {'compressed_strings': compressed_strings,
                    'decompressed_tensors': decompressed_tensors,
                    'decoder_output': fea}

    def init_weights(self):
        torch.nn.init.uniform_(self.mlp_enc_out[-1].bn.weight, -10, 10)
        torch.nn.init.uniform_(self.mlp_enc_out[-1].bn.bias, -10, 10)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    def entropy_bottleneck_compress(self, encoder_output):
        assert not self.training
        encoder_output = encoder_output.permute(0, 2, 1).unsqueeze(3).contiguous()
        return self.entropy_bottleneck.compress(encoder_output)

    def entropy_bottleneck_decompress(self, compressed_strings):
        assert not self.training
        decompressed_tensors = self.entropy_bottleneck.decompress(compressed_strings, size=(self.encoded_points_num, 1))
        decompressed_tensors = decompressed_tensors.squeeze(3).permute(0, 2, 1)
        return decompressed_tensors

def main_t():
    torch.cuda.set_device('cuda:0')
    model = PointCompressor(ModelConfig()).cuda()
    model.train()
    batch_points = torch.rand(1, 8192, 3).cuda()
    out =  model(batch_points)
    print('Done')

if __name__ == '__main__':
    main_t()