import numpy as np

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from lib.sparse_conv_layers import AbstractGenerativeUpsample

MConv = ME.MinkowskiConvolution
MReLU = ME.MinkowskiReLU
MGenConvTranspose = ME.MinkowskiGenerativeConvolutionTranspose


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv0 = MConv(channels, channels, 3, 1, bias=True, dimension=3)
        self.conv1 = MConv(channels, channels, 3, 1, bias=True, dimension=3)
        self.relu = MReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.conv0(x)))
        out += x
        return out


class InceptionResBlock(nn.Module):
    def __init__(self, channels):
        super(InceptionResBlock, self).__init__()
        self.path_0 = nn.Sequential(MConv(channels, channels // 4, 3, 1, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    MConv(channels // 4, channels // 2, 3, 1, bias=True, dimension=3))

        self.path_1 = nn.Sequential(MConv(channels, channels // 4, 1, 1, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    MConv(channels // 4, channels // 4, 3, 1, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    MConv(channels // 4, channels // 2, 1, 1, bias=True, dimension=3))

    def forward(self, x):
        out0 = self.path_0(x)
        out1 = self.path_1(x)
        out = ME.cat(out0, out1) + x
        return out


class Encoder(nn.Module):
    def __init__(self, out_channels, res_blocks_num, res_block_type):
        super(Encoder, self).__init__()
        in_channels = 1
        ch = [16, 32, 64, 32, out_channels]
        if res_block_type == 'ResNet':
            self.basic_block = ResBlock
        elif res_block_type == 'InceptionResNet':
            self.basic_block = InceptionResBlock

        self.block0 = nn.Sequential(MConv(in_channels, ch[0], 3, 1, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    MConv(ch[0], ch[1], 2, 2, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    *[self.basic_block(ch[1]) for _ in range(res_blocks_num)])

        self.block1 = nn.Sequential(MConv(ch[1], ch[1], 3, 1, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    MConv(ch[1], ch[2], 2, 2, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    *[self.basic_block(ch[2]) for _ in range(res_blocks_num)])

        self.block2 = nn.Sequential(MConv(ch[2], ch[2], 3, 1, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    MConv(ch[2], ch[3], 2, 2, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    *[self.basic_block(ch[3]) for _ in range(res_blocks_num)],
                                    MConv(ch[3], ch[4], 3, 1, bias=True, dimension=3))

    def forward(self, x):
        return self.block2(self.block1(self.block0(x)))


class GenerativeUpsample(AbstractGenerativeUpsample):
    def __init__(self, in_channels, out_channels, res_blocks_num, res_block_type,
                 mapping_target_kernel_size=1, is_last_layer=False):
        super(GenerativeUpsample, self).__init__(mapping_target_kernel_size, is_last_layer)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_blocks_num = res_blocks_num
        if res_block_type == 'ResNet':
            self.basic_block = ResBlock
        elif res_block_type == 'InceptionResNet':
            self.basic_block = InceptionResBlock
        self.upsample_block = nn.Sequential(MGenConvTranspose(self.in_channels, self.out_channels,
                                                              2, 2, bias=True, dimension=3),
                                            MReLU(inplace=True),
                                            MConv(self.out_channels, self.out_channels, 3, 1, bias=True, dimension=3),
                                            MReLU(inplace=True),
                                            *[self.basic_block(self.out_channels) for _ in range(self.res_blocks_num)])
        self.classify_block = MConv(self.out_channels, 1, 3, 1, bias=True, dimension=3)



