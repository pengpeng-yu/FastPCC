from typing import Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.data_utils import IMData
from lib.torch_utils import MLPBlock
from lib.evaluators import ImageCompressionEvaluator
from lib.entropy_models.hyperprior.noisy_deep_factorized.basic import \
    NoisyDeepFactorizedEntropyModel as HyperPriorNoisyDeepFactorizedEntropyModel

from models.image_compression.baseline.image_compressor_config import ModelConfig


class Conv2dBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, groups=1,
                 bn: bool = False, act: Union[str, nn.Module, None] = 'relu'):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation, groups, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = nn.ReLU(inplace=True) if act == 'relu' else act

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvTrans2dBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 dilation=1, groups=1,
                 bn: bool = False, act: Union[str, nn.Module, None] = 'relu'):
        super(ConvTrans2dBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding,
            groups, bias=not bn, dilation=dilation
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = nn.ReLU(inplace=True) if act == 'relu' else act

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


def channel_first_permutation(x):
    return x.permute(0, 3, 1, 2)


def channel_last_permutation(x):
    return x.permute(0, 2, 3, 1)


class ChannelFirstPermutation(nn.Module):
    def __int__(self):
        super(ChannelFirstPermutation, self).__int__()

    def forward(self, x):
        return channel_first_permutation(x)


class ChannelLastPermutation(nn.Module):
    def __int__(self):
        super(ChannelLastPermutation, self).__int__()

    def forward(self, x):
        return channel_last_permutation(x)


class ImageCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(ImageCompressor, self).__init__()
        self.encoder = nn.Sequential(
            *[Conv2dBlock(
                3 if idx == 0 else cfg.encoder_channels[idx - 1], ch,
                5, 2, 2,
                bn=cfg.use_batch_norm,
                act=cfg.activation if idx != len(cfg.encoder_channels) - 1 else None
            ) for idx, ch in enumerate(cfg.encoder_channels)]
        )
        self.decoder = nn.Sequential(
            *[ConvTrans2dBlock(
                cfg.encoder_channels[-1] if idx == 0
                else cfg.decoder_channels[idx - 1], ch,
                5, 2, 2, 1,
                bn=cfg.use_batch_norm,
                act=cfg.activation
            ) for idx, ch in enumerate(cfg.decoder_channels)],
            Conv2dBlock(cfg.decoder_channels[-1], 3, 3, 1, 1, bn=False, act=cfg.activation)
        )
        assert cfg.hyper_decoder_channels[-1] == cfg.encoder_channels[-1] * len(cfg.prior_indexes_range)
        hyper_encoder = nn.Sequential(
            ChannelFirstPermutation(),
            *[Conv2dBlock(
                cfg.encoder_channels[-1] if idx == 0
                else cfg.hyper_encoder_channels[idx - 1], ch,
                5, 2, 2,
                bn=cfg.use_batch_norm,
                act=cfg.activation if idx != len(cfg.hyper_encoder_channels) - 1 else None
            ) for idx, ch in enumerate(cfg.hyper_encoder_channels)],
            ChannelLastPermutation()
        )
        hyper_decoder = nn.Sequential(
            ChannelFirstPermutation(),
            *[ConvTrans2dBlock(
                cfg.hyper_encoder_channels[-1] if idx == 0
                else cfg.hyper_decoder_channels[idx - 1], ch,
                5, 2, 2, 1,
                bn=cfg.use_batch_norm,
                act=cfg.activation
            ) for idx, ch in enumerate(cfg.hyper_decoder_channels)],
            ChannelLastPermutation()
        )

        def parameter_fns_factory(in_channels, out_channels):
            return nn.Sequential(
                MLPBlock(in_channels, out_channels,
                         bn=None, act=cfg.activation),
                MLPBlock(out_channels, out_channels,
                         bn=None, act=cfg.activation),
                nn.Linear(out_channels, out_channels,
                          bias=True)
            )
        self.em = \
            HyperPriorNoisyDeepFactorizedEntropyModel(
                hyper_encoder=hyper_encoder,
                hyper_decoder=hyper_decoder,
                hyperprior_batch_shape=torch.Size([cfg.hyper_encoder_channels[-1]]),
                coding_ndim=3,
                hyperprior_broadcast_shape_bytes=(2, 2),
                prior_bytes_num_bytes=4,
                index_ranges=cfg.prior_indexes_range,
                parameter_fns_type='transform',
                parameter_fns_factory=parameter_fns_factory,
                num_filters=(1, 3, 3, 3, 1),
                quantize_indexes=True
            )
        self.evaluator = ImageCompressionEvaluator()
        self.cfg = cfg

    def forward(self, im_data: IMData):
        batch_im = im_data.im
        pixels_num = batch_im.shape[0] * batch_im.shape[2] * batch_im.shape[3]
        feature = self.encoder(batch_im)
        feature = channel_last_permutation(feature)

        if self.training:
            fea_tilde, loss_dict = self.em(feature)
            fea_tilde = channel_first_permutation(fea_tilde)
            im_recon = self.decoder(fea_tilde)

            for key in loss_dict:
                if key.endswith('bits_loss'):
                    loss_dict[key] = loss_dict[key] * (
                            self.cfg.bpp_loss_factor / pixels_num
                    )
            loss_dict['reconstruct_loss'] = F.mse_loss(
                im_recon, batch_im
            ) * self.cfg.reconstruct_loss_factor
            loss_dict['loss'] = sum(loss_dict.values())
            for key in loss_dict:
                if key != 'loss':
                    loss_dict[key] = loss_dict[key].item()
            loss_dict['mean_psnr'] = -10 * math.log10(
                loss_dict['reconstruct_loss'] / self.cfg.reconstruct_loss_factor
            )
            return loss_dict

        elif not self.training:
            fea_recon, compressed_strings, coding_batch_shape = self.em(feature)
            fea_recon = channel_first_permutation(fea_recon)
            im_recon = self.decoder(fea_recon)
            im_recon = (im_recon * 255).round_().clip(None, 255)
            batch_im = (batch_im * 255).round_()

            ret = self.evaluator.log_batch(
                im_recon, batch_im,
                compressed_strings,
                im_data.file_path,
                im_data.valid_range,
                im_data.results_dir
            )
            return ret

    def train(self, mode: bool = True):
        """
        Use model.train() to reset evaluator.
        """
        if mode is True:
            self.evaluator.reset()
        return super(ImageCompressor, self).train(mode=mode)
