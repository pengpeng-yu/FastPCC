import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import loss_function
from lib.points_layers import RandLANeighborFea, LocalFeatureAggregation as LFA
from lib.torch_utils import MLPBlock
from models.exp5.model_config import ModelConfig


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        neighbor_fea_generator = RandLANeighborFea(cfg.neighbor_num)
        # TODO: should I use Gumbel-Softmax trick?
        self.encoder_layers = [LFA(3, neighbor_fea_generator, 16, 24),
                               LFA(24, neighbor_fea_generator, 16, 32),
                               LFA(32, neighbor_fea_generator, 16, 48),
                               LFA(48, neighbor_fea_generator, 24, 48),
                               LFA(48, neighbor_fea_generator, 24, 64),
                               LFA(64, neighbor_fea_generator, 24, 64),
                               LFA(64, neighbor_fea_generator, 24, 128),
                               LFA(128, neighbor_fea_generator, 32, 128),]
        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        self.encoded_points_num = cfg.input_points_num

        self.mlp_enc_out = nn.Sequential(MLPBlock(128, 128, activation='leaky_relu(0.2)', batchnorm='nn.bn1d'),
                                         MLPBlock(128, 32 * 3, activation=None, batchnorm='nn.bn1d'))

        self.decoder_layers = [LFA(32 * 3, neighbor_fea_generator, 128, 128),
                               LFA(128, neighbor_fea_generator, 128, 128)]
        self.decoder_layers = nn.Sequential(*self.decoder_layers)

        self.mlp_dec_out = nn.Sequential(MLPBlock(128, 128, activation='leaky_relu(0.2)', batchnorm='nn.bn1d'),
                                         MLPBlock(128, 3, activation=None, batchnorm='nn.bn1d'))

    def forward(self, fea):
        if self.training: ori_fea = fea
        batch_size = fea.shape[0]
        xyz = fea[..., :3]  # B, N, C

        xyz, fea = self.encoder_layers((xyz, fea, None, None, None))[:2]
        fea = self.mlp_enc_out(fea)
        fea.sigmoid_()

        if self.training:
            with torch.no_grad():
                label = torch.round(fea)
                quantize_diff = label - fea

            quantize_loss = torch.nn.functional.binary_cross_entropy(fea, label) * self.cfg.quantize_loss_factor
            balance_loss = torch.mean(fea) * self.cfg.balance_loss_factor

            fea = fea + quantize_diff  # binary
            fea = self.decoder_layers((xyz, fea, None, None, None))[1]
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
            fea = self.decoder_layers((xyz, round_fea, None, None, None))[1]
            fea = self.mlp_dec_out(fea)
            fea = fea.reshape(batch_size, self.cfg.input_points_num, 3)

            return {'round_fea': round_fea,
                    'decoder_output': fea}


def main_t():
    torch.cuda.set_device('cuda:1')
    model = PointCompressor(ModelConfig()).cuda()
    model.train()
    batch_points = torch.rand(1, 8192, 3).cuda()
    out =  model(batch_points)
    print('Done')

if __name__ == '__main__':
    main_t()