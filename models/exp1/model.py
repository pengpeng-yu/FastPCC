import torch
import torch.nn as nn
import torch.nn.functional as F
import compressai
from compressai.models.utils import update_registered_buffers

from lib import loss_function
from lib.torch_utils import MLPBlock
from lib.points_layers import TransformerBlock
from models.exp1.model_config import ModelConfig


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = [TransformerBlock(3, 32, cfg.neighbor_num, True),
                        TransformerBlock(32, 64, cfg.neighbor_num, True),
                        TransformerBlock(64, 128, cfg.neighbor_num, False),]
        self.encoder = nn.Sequential(*self.encoder)
        self.mlp_enc_out = MLPBlock(128, 128,  activation=None, batchnorm='nn.bn1d')
        self.encoded_points_num = cfg.input_points_num
        self.encoded_points_dim = 128

        self.entropy_bottleneck = compressai.entropy_models.EntropyBottleneck(self.encoded_points_dim,
                                                                              init_scale=cfg.bottleneck_scaler * 4)

        self.decoder = [TransformerBlock(128, 64, cfg.neighbor_num, True),
                        TransformerBlock(64, 32, cfg.neighbor_num, False)]
        self.mlp_dec_out = MLPBlock(32, 3, activation=None, batchnorm='nn.bn1d')
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, fea):
        if self.training: ori_fea = fea
        batch_size = fea.shape[0]
        xyz = fea[..., :3]  # B, N, C only coordinate supported
        # encode
        xyz, fea = self.encoder((xyz, fea, None, None, None))[:2]
        fea = self.mlp_enc_out(fea)
        fea *= self.cfg.bottleneck_scaler

        if self.training:
            fea, likelihoods = self.entropy_bottleneck(fea.permute(0, 2, 1).unsqueeze(3).contiguous())
            fea /= self.cfg.bottleneck_scaler
            fea = fea.squeeze(3).permute(0, 2, 1).contiguous()
            likelihoods = likelihoods.squeeze(3).permute(0, 2, 1).contiguous()
            fea = self.decoder((xyz, fea, None, None, None))[1]
            fea = self.mlp_dec_out(fea)
            fea = fea.reshape(batch_size, self.cfg.input_points_num, self.cfg.input_points_dim)

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
            decompressed_tensors /= self.cfg.bottleneck_scaler
            decoder_output = self.decoder((xyz, decompressed_tensors, None, None, None))[1]
            decoder_output = self.mlp_dec_out(decoder_output)
            decoder_output = decoder_output.reshape(batch_size, self.cfg.input_points_num, self.cfg.input_points_dim)

            return {'encoder_output': fea,
                    'compressed_strings': compressed_strings,
                    'decompressed_tensors': decompressed_tensors,
                    'decoder_output': decoder_output}

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
    torch.cuda.set_device('cuda:2')
    cfg = ModelConfig()
    cfg.input_points_num = 1024
    model = PointCompressor(cfg)
    model = model.cuda()
    point_cloud = torch.rand(4, cfg.input_points_num, cfg.input_points_dim, device='cuda')
    out = model(point_cloud)  # 4, 16, 256
    out['loss'].backward()
    model.eval()
    model.entropy_bottleneck.update()
    val_out = model(point_cloud)


if __name__ == '__main__':
    main_t()