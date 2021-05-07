import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from lib import loss_function
from lib.points_layers import TransitionDown, RandLANeighborFea, LocalFeatureAggregation as LFA
from lib.torch_utils import MLPBlock
from models.exp6.model_config import ModelConfig


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        neighbor_fea_generator = RandLANeighborFea(cfg.neighbor_num)

        self.encoder_layers = [LFA(3, neighbor_fea_generator, 16, 32),
                               LFA(32, neighbor_fea_generator, 32, 64),
                               LFA(64, neighbor_fea_generator, 64, 128),
                               LFA(128, neighbor_fea_generator, 128, 256),
                               LFA(256, neighbor_fea_generator, 256, 512),
                               LFA(512, neighbor_fea_generator, 256, 1024),
                               LFA(1024, neighbor_fea_generator, 512, 2048),
                               LFA(2048, neighbor_fea_generator, 1024, 4096)]

        sample_rate = reduce(lambda x, y: x * y, [t.sample_rate for t in self.encoder_layers if isinstance(t, TransitionDown)] + [1])
        self.encoded_points_num = int(cfg.input_points_num * sample_rate)
        self.encoder_layers = nn.Sequential(*self.encoder_layers)

        # conv kernel moves in channels dimension
        self.conv_enc_out = nn.Sequential(nn.ConvTranspose1d(1, 1, 3, 2, padding=1, output_padding=1),
                                          nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),
                                          nn.Conv1d(1, 1, 3, padding=1),
                                          nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),

                                          nn.ConvTranspose1d(1, 1, 3, 2, padding=1, output_padding=1),
                                          nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),
                                          nn.Conv1d(1, 1, 3, padding=1),
                                          nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),

                                          nn.ConvTranspose1d(1, 1, 3, 2, padding=1, output_padding=1),
                                          nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),
                                          nn.Conv1d(1, 1, 3, padding=1),
                                          nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),

                                          nn.ConvTranspose1d(1, 1, 3, 2, padding=1, output_padding=1),
                                          nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),
                                          nn.Conv1d(1, 1, 3, padding=1),
                                          nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),

                                          nn.ConvTranspose1d(1, 1, 3, 2, padding=1, output_padding=1),
                                          nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),
                                          nn.Conv1d(1, 1, 3, padding=1),
                                          nn.BatchNorm1d(1))
        # 1024 * 32 * 3 / 4096 = 24, 2 ** 5 = 32, 4096 * 32 = 2 ** 17

        self.decoder_layers = [nn.Conv1d(1, 1, 3, stride=2, padding=1),
                               nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),

                               nn.Conv1d(1, 1, 3, stride=2, padding=1),
                               nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),

                               nn.Conv1d(1, 1, 3, stride=2, padding=1),
                               nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),

                               nn.Conv1d(1, 1, 3, stride=2, padding=1),
                               nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),

                               nn.Conv1d(1, 1, 3, stride=2, padding=1),
                               nn.BatchNorm1d(1), nn.LeakyReLU(0.2, True),]

        self.decoder_layers = nn.Sequential(*self.decoder_layers)
        self.mlp_dec_out = nn.Sequential(MLPBlock(4096, 1024 * 3, activation='leaky_relu(0.2)', batchnorm='nn.bn1d'),
                                         MLPBlock(1024 * 3, 1024 * 3, activation=None, batchnorm='nn.bn1d'),)

    def forward(self, fea):
        if self.training: ori_fea = fea
        batch_size = fea.shape[0]
        xyz = fea[..., :3]  # B, N, C
        # encode
        fea = self.encoder_layers((xyz, fea, None, None))[1]
        fea = torch.max(fea, dim=1, keepdim=True).values
        fea = self.conv_enc_out(fea)
        fea = fea.sigmoid_()

        if self.training:
            with torch.no_grad():
                label = torch.round(fea)
                quantize_diff = label - fea

            quantize_loss = self.quantize_loss(fea) * self.cfg.quantize_loss_factor

            fea = fea + quantize_diff  # binary
            fea = self.decoder_layers(fea)
            fea = self.mlp_dec_out(fea)
            fea = fea.reshape(batch_size, self.cfg.input_points_num, 3)

            reconstruct_loss = loss_function.chamfer_loss(fea, ori_fea)
            loss = reconstruct_loss + quantize_loss
            return {'quantize_loss': quantize_loss.detach().cpu().item(),
                    'reconstruct_loss': reconstruct_loss.detach().cpu().item(),
                    'loss': loss}
        else:
            round_fea = torch.round(fea)
            fea = self.decoder_layers(fea)
            fea = self.mlp_dec_out(fea)
            fea = fea.reshape(batch_size, self.cfg.input_points_num, 3)

            return {'round_fea': round_fea,
                    'decoder_output': fea}

    @staticmethod
    def quantize_loss(fea):
        loss = torch.where(fea < 0.5, 2 * fea, -fea + 1.5)
        return torch.mean(loss)


def main_t():
    from thop import profile
    from thop import clever_format

    torch.cuda.set_device('cuda:1')
    cfg = ModelConfig()
    cfg.input_points_num = 1024

    xyz = torch.rand(2, cfg.input_points_num, 3)
    model = PointCompressor(cfg)
    model.train()

    macs, params = profile(model, inputs=(xyz,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f'macs: {macs}, params: {params}')  # macs: 475.831G, params: 62.875M

    xyz = xyz.cuda()
    model = model.cuda()
    out =  model(xyz)

    print('Done')

if __name__ == '__main__':
    main_t()