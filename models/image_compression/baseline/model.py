import io
from typing import Union, Tuple, List, Optional
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.data_utils import IMData
from lib.torch_utils import MLPBlock
from lib.evaluators import ImageCompressionEvaluator
from lib.entropy_models.rans_coder import IndexedRansCoder
from lib.entropy_models.continuous_batched import NoisyDeepFactorizedEntropyModel
from lib.torch_utils import concat_loss_dicts

from models.image_compression.baseline.model_config import ModelConfig


MLPBlock = partial(MLPBlock, version='conv')


def get_act_module(act: Union[str, nn.Module, None]) -> Optional[nn.Module]:
    if isinstance(act, nn.Module):
        act_module = act
    elif act is None or act == 'None':
        act_module = None
    elif act == 'relu':
        act_module = nn.ReLU(inplace=True)
    elif act.startswith('leaky_relu'):
        act_module = nn.LeakyReLU(
            negative_slope=float(act.split('(', 1)[1].split(')', 1)[0]),
            inplace=True)
    elif act == 'sigmoid':
        act_module = nn.Sigmoid()
    elif act == 'prelu':
        act_module = nn.PReLU()
    else:
        raise NotImplementedError(act)
    return act_module


class Conv2dBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, groups=1,
                 bn: bool = False, act: Union[str, nn.Module, None] = 'relu'):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation, groups, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = get_act_module(act)

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
        self.act = get_act_module(act)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(Model, self).__init__()
        self.cfg = cfg
        self.encoder = nn.ModuleList()
        self.residual_m = nn.ModuleList()
        self.fusion_m = nn.ModuleList()
        self.decoder = nn.ModuleList()
        last_ch = 3
        for ch in cfg.channels:
            self.encoder.append(nn.Sequential(
                Conv2dBlock(last_ch, ch, 3, 1, 1, bn=cfg.use_batch_norm, act=cfg.activation),
                Conv2dBlock(ch, ch, 2, 2, 0, bn=cfg.use_batch_norm, act=cfg.activation)
            ))
            self.decoder.append(nn.Sequential(
                ConvTrans2dBlock(ch, ch, 2, 2, 0, 0, bn=cfg.use_batch_norm, act=cfg.activation),
                Conv2dBlock(ch, last_ch, 3, 1, 1, bn=cfg.use_batch_norm, act=cfg.activation)
            ))
            self.residual_m.append(nn.Sequential(
                MLPBlock(ch * 2, ch, bn=cfg.use_batch_norm, act=cfg.activation),
                MLPBlock(ch, 1, bn=cfg.use_batch_norm, act=None)
            ))
            self.fusion_m.append(nn.ModuleList([
                MLPBlock(1, ch, bn=cfg.use_batch_norm, act=cfg.activation),
                MLPBlock(ch * 2, ch, bn=cfg.use_batch_norm, act=cfg.activation)
            ]))
            last_ch = ch
        self.encoder.append(nn.Sequential(
            Conv2dBlock(cfg.channels[-1], cfg.channels[-1], 3, 1, 1, bn=cfg.use_batch_norm, act=cfg.activation),
            Conv2dBlock(cfg.channels[-1], 1, 2, 2, 0, bn=cfg.use_batch_norm, act=cfg.activation)
        ))
        self.decoder.append(nn.Sequential(
            ConvTrans2dBlock(1, cfg.channels[-1], 2, 2, 0, 0, bn=cfg.use_batch_norm, act=cfg.activation),
            Conv2dBlock(cfg.channels[-1], cfg.channels[-1], 3, 1, 1, bn=cfg.use_batch_norm, act=None)
        ))

        self.rans_coder = IndexedRansCoder(False, 1)
        self.em = NoisyDeepFactorizedEntropyModel(
            batch_shape=torch.Size([1]),
            coding_ndim=3,
            bottleneck_scaler=cfg.bottleneck_scaler,
            init_scale=10,
            broadcast_shape_bytes=(3,),
        )
        self.broadcast_shape_bytes = 3
        self.evaluator = ImageCompressionEvaluator()
        self.cfg = cfg

    def forward(self, *arg, **kwargs):
        if self.training:
            return self.train_forward(*arg, **kwargs)
        else:
            return self.test_forward(*arg, **kwargs)

    def train_forward(self, im_data: IMData):
        im_data.im /= 255
        x = im_data.im
        pixels_num = x.shape[0] * x.shape[2] * x.shape[3]

        cached_x = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            cached_x.append(x)
        x = self.decoder[-1](x)
        loss_dict = {}
        for i in range(len(self.decoder) - 2, -1, -1):
            res = self.residual_m[i](torch.cat((cached_x[i], x), 1))
            res_tilde, fea_loss_dict = self.em(res.squeeze(1).unsqueeze(3))
            concat_loss_dicts(loss_dict, fea_loss_dict, lambda k: f'fea_{i}_' + k)
            res = self.fusion_m[i][0](res_tilde.squeeze(3).unsqueeze(1))
            x = self.fusion_m[i][1](torch.cat((res, x), 1))
            x = self.decoder[i](x)

        for key in loss_dict:
            if key.endswith('bits_loss'):
                loss_dict[key] = loss_dict[key] * (self.cfg.bpp_loss_factor / pixels_num)
        loss_dict['recon_loss'] = F.mse_loss(x, im_data.im) * self.cfg.recon_loss_factor
        loss_dict['loss'] = sum(loss_dict.values())
        for key in loss_dict:
            if key != 'loss':
                loss_dict[key] = loss_dict[key].item()
        loss_dict['mean_psnr'] = -10 * math.log10(loss_dict['recon_loss'] / self.cfg.recon_loss_factor)
        return loss_dict

    def test_forward(self, im_data: IMData):
        assert im_data.im.shape[0] == 1
        compressed_bytes = self.compress(im_data.im / 255)
        im_recon = self.decompress(compressed_bytes)

        ret = self.evaluator.log(
            im_recon[0], im_data.im[0],
            compressed_bytes,
            im_data.file_path[0],
            im_data.results_dir
        )
        return ret

    def compress(self, x):
        cached_x = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            cached_x.append(x)
        x *= self.cfg.bottleneck_scaler
        x.round_()
        res_list = [x.cpu().to(torch.int32).numpy().reshape(-1)]
        x /= self.cfg.bottleneck_scaler
        x = self.decoder[-1](x)
        for i in range(len(self.decoder) - 2, -1, -1):
            res = self.residual_m[i](torch.cat((cached_x[i], x), 1))
            res *= self.cfg.bottleneck_scaler
            res.round_()
            res_list.append(res.cpu().to(torch.int32).numpy().reshape(-1))
            res /= self.cfg.bottleneck_scaler
            res = self.fusion_m[i][0](res)
            x = self.fusion_m[i][1](torch.cat((res, x), 1))
            x = self.decoder[i](x)

        with io.BytesIO() as bs:
            bs.write(int_to_bytes(cached_x[-1].shape[2], 2))
            bs.write(int_to_bytes(cached_x[-1].shape[3], 2))
            res_concat = np.concatenate(res_list, axis=0)
            del res_list
            self.rans_encode_with_cdf(res_concat, bs)
            concat_bytes = bs.getvalue()
        return concat_bytes

    def decompress(self, concat_bytes: bytes):
        with io.BytesIO(concat_bytes) as bs:
            bottom_height, bottom_width = bytes_to_int(bs.read(2)), bytes_to_int(bs.read(2))
            res_concat, res_cdf = self.rans_decode_with_cdf(bs)
            res_concat /= self.cfg.bottleneck_scaler

            x, res_concat = torch.split(
                res_concat, [bottom_height * bottom_width, res_concat.shape[0] - bottom_height * bottom_width], dim=0)

        x = self.decoder[-1](x.reshape(1, 1, bottom_height, bottom_width))
        for i in range(len(self.decoder) - 2, -1, -1):
            res, res_concat = torch.split(
                res_concat, [x.shape[2] * x.shape[3], res_concat.shape[0] - x.shape[2] * x.shape[3]], dim=0)
            res = self.fusion_m[i][0](res.reshape(1, 1, x.shape[2], x.shape[3]))
            x = self.fusion_m[i][1](torch.cat((res, x), 1))
            x = self.decoder[i](x)

        im_recon = (x * 255).round_().clip(0, 255)
        return im_recon

    def rans_encode_with_cdf(self, target: np.ndarray, bs: io.BytesIO, offset: int = None):
        bs.write(int_to_bytes(target.shape[0], self.broadcast_shape_bytes))
        if offset is None:
            offset = target.min().item()
            bs.write(int_to_bytes(-offset, 1))

        pmf = np.bincount((target - offset).reshape(-1))
        self.rans_coder.init_with_pmfs(pmf[None], np.array([offset], dtype=np.int32))
        cdf = self.rans_coder.get_cdfs()[0]
        bs.write(int_to_bytes(len(cdf) - 2, 1))
        for cd in cdf[1:-1]:
            bs.write(int_to_bytes(cd, 2))

        target_bytes = self.rans_coder.encode(target.reshape(1, -1))[0]
        bs.write(int_to_bytes(len(target_bytes), 3))
        bs.write(target_bytes)

    def rans_decode_with_cdf(self, bs: io.BytesIO, dtype=torch.float, offset: int = None, channels: int = None)\
            -> Tuple[torch.Tensor, List[int]]:
        shape_sum = bytes_to_int(bs.read(self.broadcast_shape_bytes))
        if offset is None:
            offset = -bytes_to_int(bs.read(1))
        cdf = [0, *(bytes_to_int(bs.read(2))
                    for _ in range(bytes_to_int(bs.read(1)))), 1 << 16]
        self.rans_coder.init_with_quantized_cdfs([cdf], np.array([offset], dtype=np.int32))

        bytes_len = bytes_to_int(bs.read(3))
        target_bytes = bs.read(bytes_len)
        target = np.empty((1, shape_sum * (channels or 1)), np.int32)
        self.rans_coder.decode([target_bytes], target)

        target = torch.from_numpy(
            target.reshape(shape_sum, (channels or 1))
        ).to(dtype=dtype, device=next(self.parameters()).device)
        return target, cdf


def int_to_bytes(x, length, byteorder='little', signed=False):
    assert isinstance(x, int)
    return x.to_bytes(length, byteorder=byteorder, signed=signed)


def bytes_to_int(s, byteorder='little', signed=False):
    assert isinstance(s, bytes)
    return int.from_bytes(s, byteorder=byteorder, signed=signed)
