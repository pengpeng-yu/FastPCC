import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import compressai
from compressai.models.utils import update_registered_buffers

from lib import loss_function
from models.exp0.transformer import TransformerBlock
from models.exp0.model_config import ModelConfig


class TransitionDown(nn.Module):
    def __init__(self, nsample, sample_method):
        super(TransitionDown, self).__init__()
        self.nsample = nsample
        self.sample_method = sample_method

    def forward(self, xyz, points_fea):
        if self.sample_method == 'uniform':
            return xyz[:, : self.nsample], points_fea[:, : self.nsample]
        else:
            raise NotImplementedError

    def __repr__(self):
        return f'TransitionDown({self.nsample}, {self.sample_method})'


class TransitionUp(nn.Module):
    def __init__(self, inchnls, outchnls, innum, outnum):
        super(TransitionUp, self).__init__()
        self.mlp_num = nn.Sequential(nn.Linear(innum, outnum), nn.ReLU(inplace=True))
        self.mlp_chnl = nn.Sequential(nn.Linear(inchnls, outchnls), nn.ReLU(inplace=True))
        self.inchnls, self.outchnls, self.innum, self.outnum = inchnls, outchnls, innum, outnum

    def forward(self, fea):
        batch_size, points_num, inchnls = fea.shape
        assert self.inchnls == inchnls, self.innum == points_num
        fea = fea.permute(0, 2, 1).contiguous()
        fea = self.mlp_num(fea)
        fea = fea.permute(0, 2, 1).contiguous()
        fea = self.mlp_chnl(fea)
        return fea

    def __repr__(self):
        return f'TransitionUp({self.inchnls}, {self.outchnls}, {self.innum}, {self.outnum})'


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoded_points_num = cfg.input_points_num // (cfg.dowansacle_per_block ** cfg.encoder_blocks_num)
        assert self.encoded_points_num * (cfg.dowansacle_per_block ** cfg.encoder_blocks_num) == cfg.input_points_num

        self.mlp0 = nn.Sequential(nn.Linear(cfg.input_points_dim, cfg.first_mlp_dim), nn.ReLU(inplace=True))

        self.transition_downs = nn.ModuleList()
        self.encoder_transformers = nn.ModuleList()
        nchannels = cfg.first_mlp_dim
        for i in range(cfg.encoder_blocks_num):
            self.encoder_transformers.append(TransformerBlock(nchannels, nchannels * cfg.chnl_upscale_per_block,
                                                              cfg.neighbor_num, nchannels * cfg.chnl_upscale_per_block))
            self.transition_downs.append(TransitionDown(cfg.input_points_num // cfg.dowansacle_per_block ** (i + 1),
                                                        cfg.sample_method))
            nchannels = nchannels * cfg.chnl_upscale_per_block

        self.mlp1 = nn.Sequential(nn.Linear(nchannels, nchannels), nn.ReLU(inplace=True),
                                  nn.Linear(nchannels, cfg.encoded_points_dim))

        self.entropy_bottleneck = compressai.entropy_models.EntropyBottleneck(cfg.encoded_points_dim)

        self.decoder = [nn.Linear(cfg.encoded_points_dim, nchannels), nn.ReLU(inplace=True)]
        for i in range(cfg.decoder_blocks_num):
            self.decoder.append(TransitionUp(nchannels, nchannels // cfg.chnl_downscale_per_block,
                                             self.encoded_points_num * cfg.upsacle_per_block ** i,
                                             self.encoded_points_num * cfg.upsacle_per_block ** (i + 1)))
            nchannels = nchannels // cfg.chnl_downscale_per_block
        self.decoder.append(nn.Linear(nchannels, cfg.input_points_dim))
        self.decoder = nn.Sequential(*self.decoder)

        self.init_weights()

    def forward(self, fea):
        if self.training: ori_fea = fea
        xyz = fea[..., :3]  # B, N, C
        fea = self.mlp0(fea)
        for i in range(self.cfg.encoder_blocks_num):
            fea = self.encoder_transformers[i](xyz, fea)
            xyz, fea = self.transition_downs[i](xyz, fea)
        encoder_output = self.mlp1(fea)

        if self.training:
            encoder_output_hat, likelihoods = self.entropy_bottleneck(encoder_output.permute(0, 2, 1).unsqueeze(3).contiguous())
            encoder_output_hat = encoder_output_hat.squeeze(3).permute(0, 2, 1).contiguous()
            likelihoods = likelihoods.squeeze(3).permute(0, 2, 1).contiguous()
            decoder_output = self.decoder(encoder_output_hat)

            bpp_loss = torch.log2(likelihoods).sum() * (-self.cfg.bpp_loss_factor / (ori_fea.shape[0] * ori_fea.shape[1]))
            reconstruct_loss = loss_function.chamfer_loss(decoder_output, ori_fea)
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
        torch.nn.init.uniform_(self.mlp1[-1].weight, -20, 20)
        torch.nn.init.uniform_(self.mlp1[-1].bias, -20, 20)

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
    point_cloud = torch.rand(2, 8192, 3, device='cuda')
    res = module(point_cloud)  # 4, 16, 256
    torch.save(module.state_dict(), 'weights/t.pth')


if __name__ == '__main__':
    main_t()