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
    def __init__(self, in_channels, out_channels, res_blocks_num, res_block_type, mapping_target_kernel_size=1):
        super(GenerativeUpsample, self).__init__(mapping_target_kernel_size)
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


# deprecated
class Decoder(nn.Module):
    def __init__(self, in_channels, res_blocks_num, res_block_type):
        nn.Module.__init__(self)
        ch = [in_channels, 64, 32, 16]

        self.layers = nn.Sequential(GenerativeUpsample(ch[0], ch[1], res_blocks_num, res_block_type),
                                    GenerativeUpsample(ch[1], ch[2], res_blocks_num, res_block_type),
                                    GenerativeUpsample(ch[2], ch[3], res_blocks_num, res_block_type))

        self.block0 = nn.Sequential(MGenConvTranspose(ch[0], ch[1], 2, 2, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    MConv(ch[1], ch[1], 3, 1, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    *[self.basic_block(ch[1]) for _ in range(res_blocks_num)])
        self.conv0_cls = MConv(ch[1], out_channels, 3, 1, bias=True, dimension=3)

        self.block1 = nn.Sequential(MGenConvTranspose(ch[1], ch[2], 2, 2, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    MConv(ch[2], ch[2], 3, 1, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    *[self.basic_block(ch[2]) for _ in range(res_blocks_num)])
        self.conv1_cls = MConv(ch[2], out_channels, 3, 1, bias=True, dimension=3)

        self.block2 = nn.Sequential(MGenConvTranspose(ch[2], ch[3], 2, 2, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    MConv(ch[3], ch[3], 3, 1, bias=True, dimension=3),
                                    MReLU(inplace=True),
                                    *[self.basic_block(ch[3]) for _ in range(res_blocks_num)])
        self.conv2_cls = MConv(ch[3], out_channels, 3, 1, bias=True, dimension=3)

        self.pruning = ME.MinkowskiPruning()

    @staticmethod
    @torch.no_grad()
    def get_target_by_key(out, target_key):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(
            target_key, out.tensor_stride[0], force_creation=True)

        kernel_map = cm.kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=1,
            region_type=1,
        )
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    @staticmethod
    @torch.no_grad()
    def get_coords_nums_by_key(out, target_key):
        cm = out.coords_man
        strided_target_key = cm.stride(target_key, out.tensor_stride[0], force_creation=True)

        ins, outs = cm.get_kernel_map(
            out.coords_key,
            strided_target_key,
            kernel_size=1,
            region_type=1)

        row_indices_per_batch = cm.get_row_indices_per_batch(out.coords_key)

        coords_nums = [len(np.in1d(row_indices, ins[0]).nonzero()[0]) for _, row_indices in
                       enumerate(row_indices_per_batch)]
        # coords_nums = [len(np.intersect1d(row_indices,ins[0])) for _, row_indices in enumerate(row_indices_per_batch)]
        return coords_nums

    @staticmethod
    @torch.no_grad()
    def keep_adaptive(out, coords_nums, rho=1.0):
        keep = torch.zeros(len(out), dtype=torch.bool)
        #  get row indices per batch.
        # row_indices_per_batch = out.coords_man.get_row_indices_per_batch(out.coords_key)
        row_indices_per_batch = out._batchwise_row_indices

        for row_indices, ori_coords_num in zip(row_indices_per_batch, coords_nums):
            coords_num = min(len(row_indices), ori_coords_num * rho)  # select top k points.
            values, indices = torch.topk(out.F[row_indices].squeeze(), int(coords_num))
            keep[row_indices[indices]] = True

        return keep

    def forward(self, x, target_key, adaptive=False, rhos=(1.0, 1.0, 1.0)):
        target_format = 'key'
        assert adaptive is False

        targets = []
        out_cls = []
        keeps = []

        # Decode 0.
        out0 = self.block0(x)
        out0_cls = self.conv0_cls(out0)

        # get target 0.
        target0 = self.get_target_by_key(out0, target_key)

        targets.append(target0)
        out_cls.append(out0_cls)

        # get keep 0.
        if adaptive:
            coords_nums0 = self.get_coords_nums_by_key(out0, target_key)
            keep0 = self.keep_adaptive(out0_cls, coords_nums0, rho=rhos[0])
        else:
            keep0 = (out0_cls.F > 0).cpu().squeeze()
            if out0_cls.F.max() < 0:
                # keep at least one points.
                print('===0; max value < 0', out0_cls.F.max())
                _, idx = torch.topk(out0_cls.F.squeeze(), 1)
                keep0[idx] = True
        keeps.append(keep0)

        # If training, force target shape generation, use net.eval() to disable
        if self.training: keep0 += target0
        # Remove voxels
        out0_pruned = self.pruning(out0, keep0.to(out0.device))

        # Decode 1.
        out1 = self.block1(out0_pruned)
        out1_cls = self.conv1_cls(out1)

        # get target 1.
        target1 = self.get_target_by_key(out1, target_key)

        targets.append(target1)
        out_cls.append(out1_cls)

        # get keep 1.
        if adaptive:
            coords_nums1 = self.get_coords_nums_by_key(out1, target_key)
            keep1 = self.keep_adaptive(out1_cls, coords_nums1, rho=rhos[1])
        else:
            keep1 = (out1_cls.F > 0).cpu().squeeze()
            if out1_cls.F.max() < 0:
                # keep at least one points.
                print('===1; max value < 0', out1_cls.F.max())
                _, idx = torch.topk(out1_cls.F.squeeze(), 1)
                keep1[idx] = True
        keeps.append(keep1)

        if self.training: keep1 += target1
        out1_pruned = self.pruning(out1, keep1.to(out1.device))

        # Decode 2.
        out2 = self.block2(out1_pruned)
        out2_cls = self.conv2_cls(out2)

        # get target 2.
        target2 = self.get_target_by_key(out2, target_key)

        targets.append(target2)
        out_cls.append(out2_cls)

        # get keep 2.
        if adaptive:
            coords_nums2 = self.get_coords_nums_by_key(out2, target_key)
            keep2 = self.keep_adaptive(out2_cls, coords_nums2, rho=rhos[2])
        else:
            keep2 = (out2_cls.F > 0).cpu().squeeze()
            if out2_cls.F.max() < 0:
                # keep at least one points.
                print('===2; max value < 0', out2_cls.F.max())
                _, idx = torch.topk(out2_cls.F.squeeze(), 1)
                keep2[idx] = True
        keeps.append(keep2)

        out2_pruned = self.pruning(out2_cls, keep2.to(out2_cls.device))

        return out2_pruned, out_cls, targets, keeps


if __name__ == '__main__':
    encoder = Encoder(8)
    print(encoder)
    decoder = Decoder(8)
    print(decoder)

