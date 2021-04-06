import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import compressai
from compressai.models.utils import update_registered_buffers

from lib import loss_function
from lib.pointnet_utils import PointNetSetAbstraction
from .transformer import TransformerBlock
from .model_config import ModelConfig


class TransitionDown(nn.Module):
    def __init__(self, nsample, sample_method, nneighbor, nchannels) -> None:
        super().__init__()
        self.sa = PointNetSetAbstraction(nsample, sample_method, 0, nneighbor,
                                         nchannels[0], nchannels[1:], knn=True, attn_xyz=False)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, inchannels, outchannels, scale_factor):
        super(TransitionUp, self).__init__()
        self.scale_factor = scale_factor
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.mlp = nn.Sequential(nn.Linear(inchannels, outchannels * scale_factor),
                                 nn.ReLU(inplace=True))

    def forward(self, points_fea):
        batch_size, points_num, inchannels = points_fea.shape
        assert self.inchannels == inchannels
        points_fea = self.mlp(points_fea)  # b, n, ic -> b, n, scale, oc
        points_fea = points_fea.reshape(batch_size, points_num * self.scale_factor, self.outchannels)
        return points_fea


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoded_points_num = cfg.input_points_num // (cfg.dowansacle_per_block ** cfg.encoder_blocks_num)
        assert self.encoded_points_num * (cfg.dowansacle_per_block ** cfg.encoder_blocks_num) == cfg.input_points_num

        self.mlp0 = nn.Sequential(nn.Linear(cfg.input_points_dim, cfg.first_mlp_dim), nn.ReLU(inplace=True),
                                  nn.Linear(cfg.first_mlp_dim, cfg.first_mlp_dim))
        self.transformer0 = TransformerBlock(cfg.first_mlp_dim, cfg.first_mlp_dim, cfg.neighbor_num)

        self.transition_downs = nn.ModuleList()
        self.encoder_transformers = nn.ModuleList()
        nchannels = cfg.first_mlp_dim
        for i in range(cfg.encoder_blocks_num):
            nchannels = nchannels * cfg.chnl_upscale_per_block
            self.transition_downs.append(TransitionDown(cfg.input_points_num // cfg.dowansacle_per_block ** (i + 1),
                                                        cfg.sample_method, cfg.neighbor_num,
                                                        [nchannels // cfg.chnl_upscale_per_block + 3, nchannels, nchannels]))
            self.encoder_transformers.append(TransformerBlock(nchannels, cfg.transformer_dim, cfg.neighbor_num))
        self.mlp1 = nn.Linear(nchannels, cfg.encoded_points_dim)
        torch.nn.init.uniform_(self.mlp1.weight, -10, 10)
        torch.nn.init.uniform_(self.mlp1.bias, -10, 10)

        self.entropy_bottleneck = compressai.entropy_models.EntropyBottleneck(cfg.encoded_points_dim)

        self.decoder = [nn.Linear(cfg.encoded_points_dim, nchannels)]
        for i in range(cfg.decoder_blocks_num):
            self.decoder.append(TransitionUp(nchannels, nchannels // cfg.chnl_downscale_per_block, cfg.upsacle_per_block))
            nchannels = nchannels // cfg.chnl_downscale_per_block
        self.decoder.append(nn.Linear(nchannels, 3))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        xyz = x[..., :3]  # B, N, C
        feature = self.transformer0(xyz, self.mlp0(x))
        for i in range(self.cfg.encoder_blocks_num):
            xyz, feature = self.transition_downs[i](xyz, feature)
            feature = self.encoder_transformers[i](xyz, feature)
        encoder_output = self.mlp1(feature)

        if self.training:
            encoder_output_hat, likelihoods = self.entropy_bottleneck(encoder_output.permute(0, 2, 1).unsqueeze(3).contiguous())
            encoder_output_hat = encoder_output_hat.squeeze(3).permute(0, 2, 1).contiguous()
            likelihoods = likelihoods.squeeze(3).permute(0, 2, 1).contiguous()
            decoder_output = self.decoder(encoder_output_hat)

            bpp_loss = torch.log2(likelihoods).sum() * (-self.cfg.bpp_loss_factor / (x.shape[0] * x.shape[1]))
            reconstruct_loss = loss_function.chamfer_loss(decoder_output, x)
            aux_loss = self.entropy_bottleneck.loss() * self.cfg.aux_loss_factor
            loss = reconstruct_loss + bpp_loss + aux_loss
            return {'aux_loss': aux_loss.detach().cpu().item(),
                    'bpp_loss': bpp_loss.detach().cpu().item(),
                    'reconstruct_loss': reconstruct_loss.detach().cpu().item(),
                    'loss': loss}
        else:
            compressed_strings = self.entropy_bottleneck_compress(encoder_output)
            decompressed_tensors = self.entropy_bottleneck_decompress(compressed_strings)
            decoder_output = self.decoder(decompressed_tensors)
            return {'encoder_output': encoder_output,
                    'compressed_strings': compressed_strings,
                    'decompressed_tensors': decompressed_tensors,
                    'decoder_output': decoder_output}

    def init_weights(self):
        pass

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
    module = PointCompressor(ModelConfig())
    module = module.cuda()
    point_cloud = torch.rand(4, 1024, 3, device='cuda')
    res = module(point_cloud)  # 4, 16, 256
    torch.save(module.state_dict(), 'weights/t.pth')


if __name__ == '__main__':
    main_t()