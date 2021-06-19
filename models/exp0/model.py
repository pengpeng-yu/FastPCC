import torch
import torch.nn as nn
import torch.nn.functional as F
import compressai
from compressai.models.utils import update_registered_buffers

from lib import loss_function
from lib.torch_utils import MLPBlock
from lib.points_layers import PointLayerMessage, TransitionDown, TransformerBlock
from models.exp0.model_config import ModelConfig


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = [TransformerBlock(3, 24, cfg.neighbor_num),
                        TransformerBlock(24, 64, cfg.neighbor_num),
                        TransitionDown('uniform', 0.5),
                        TransformerBlock(64, 128, cfg.neighbor_num),
                        TransitionDown('uniform', 0.5),
                        TransformerBlock(128, 256, cfg.neighbor_num),
                        TransitionDown('uniform', 0.5),
                        TransformerBlock(256, 512, cfg.neighbor_num),
                        TransitionDown('uniform', 0.5),
                        TransformerBlock(512, 1024, cfg.neighbor_num),
                        TransitionDown('uniform', 0.5),
                        TransformerBlock(1024, 1024, cfg.neighbor_num)]
        self.encoder = nn.Sequential(*self.encoder)
        self.mlp_enc_out = MLPBlock(1024, 1024,  activation=None, batchnorm='nn.bn1d')
        self.encoded_points_num = cfg.input_points_num // 32
        self.encoded_points_dim = 1024

        self.entropy_bottleneck = compressai.entropy_models.EntropyBottleneck(self.encoded_points_dim)

        self.decoder = [TransformerBlock(1024, 512, cfg.neighbor_num),
                        TransformerBlock(512, 256, cfg.neighbor_num),
                        TransformerBlock(256, 96, cfg.neighbor_num)]
        self.mlp_dec_out = MLPBlock(96, 96, activation=None, batchnorm='nn.bn1d')
        self.decoder = nn.Sequential(*self.decoder)
        self.init_weights()

    def forward(self, fea):
        if self.training: ori_fea = fea
        batch_size = fea.shape[0]
        xyz = fea[..., :3]  # B, N, C only coordinate supported
        # encode
        msg = self.encoder(PointLayerMessage(xyz=xyz, feature=fea))  # type: PointLayerMessage
        xyz, fea = msg.xyz, msg.feature
        fea = self.mlp_enc_out(fea)

        if self.training:
            fea, likelihoods = self.entropy_bottleneck(fea.permute(0, 2, 1).unsqueeze(3).contiguous())
            fea = fea.squeeze(3).permute(0, 2, 1).contiguous()
            likelihoods = likelihoods.squeeze(3).permute(0, 2, 1).contiguous()
            fea = self.decoder(PointLayerMessage(xyz=xyz, feature=fea)).feature
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
            decoder_output = self.decoder(PointLayerMessage(xyz=xyz, feature=decompressed_tensors)).feature
            decoder_output = self.mlp_dec_out(decoder_output)
            decoder_output = decoder_output.reshape(batch_size, self.cfg.input_points_num, self.cfg.input_points_dim)

            return {'encoder_output': fea,
                    'compressed_strings': compressed_strings,
                    'decompressed_tensors': decompressed_tensors,
                    'decoder_output': decoder_output}

    def init_weights(self):
        torch.nn.init.uniform_(self.mlp_enc_out.bn.weight, -10, 10)
        torch.nn.init.uniform_(self.mlp_enc_out.bn.bias, -10, 10)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        return super().load_state_dict(state_dict, strict=strict)

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
    print('Done')


if __name__ == '__main__':
    main_t()
