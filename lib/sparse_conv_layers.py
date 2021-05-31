import torch
import torch.nn as nn
import MinkowskiEngine as ME

MConv = ME.MinkowskiConvolution
MReLU = ME.MinkowskiReLU
MGenConvTranspose = ME.MinkowskiGenerativeConvolutionTranspose
MConvTranspose = ME.MinkowskiConvolutionTranspose  # TODO: difference when there is no higher strides coordinates?


class AbstractGenerativeUpsample(nn.Module):
    def __init__(self, mapping_target_kernel_size=1, return_fea=True):
        super(AbstractGenerativeUpsample, self).__init__()
        self.mapping_target_kernel_size = mapping_target_kernel_size
        self.return_fea = return_fea
        self.upsample_block = None
        self.classify_block = None
        self.pruning = ME.MinkowskiPruning()

    def forward(self, input_tuple):
        """
        fea: SparseTensor,
        cached_exist: List[SparseTensor], prediction of existence,
        cached_target: List[SparseTensor] or None,
        target_key: MinkowskiEngine.CoordinateMapKey or None.

        During training, features from last layer are used to generate new features.
        Existence prediction of each upsample layer and cooresponding target is used for loss computation.
        Target_key(identical for all upsample layers) is for generating target of each upsample layer.

        During testing, cached_target and target_key are no longer needed.
        """
        fea, cached_exist, cached_target, target_key = input_tuple

        fea = self.upsample_block(fea)
        exist = self.classify_block(fea)

        with torch.no_grad():
            keep = (exist.F.squeeze() > 0)

        if self.training:
            if ME.sparse_tensor_operation_mode() == \
                    ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER:
                assert target_key is None
                cm = ME.global_coordinate_manager
            else:
                cm = fea.coordinate_manager

            strided_target_key = cm.stride(target_key, fea.tensor_stride)
            kernel_map = cm.kernel_map(fea.coordinate_map_key,
                                       strided_target_key,
                                       kernel_size=self.mapping_target_kernel_size,
                                       region_type=1)
            target = torch.zeros(fea.shape[0], dtype=torch.bool, device=fea.device)
            for k, curr_in in kernel_map.items():
                target[curr_in[0].type(torch.int64)] = 1

            keep |= target

            if cached_target is None or cached_target == []:
                cached_target = [target]
            else:
                cached_target.append(target)

            if cached_exist is None or cached_exist == []:
                cached_exist = [exist]
            else:
                cached_exist.append(exist)

        elif not self.training:
            # only return the last pruned existence prediction during training.
            cached_exist = [self.pruning(exist, keep)]

        if self.return_fea:
            fea = self.pruning(fea, keep)
        else: fea = None

        return fea, cached_exist, cached_target, target_key


def generative_upsample_t():
    class GenerativeUpsample(AbstractGenerativeUpsample):
        def __init__(self, mapping_target_kernel_size=1):
            super(GenerativeUpsample, self).__init__(mapping_target_kernel_size)
            self.upsample_block = nn.Sequential(MGenConvTranspose(16, 16, 2, 2, bias=True, dimension=3),
                                                MReLU(inplace=True),
                                                MConv(16, 16, 3, 1, bias=True, dimension=3),
                                                MReLU(inplace=True))
            self.classify_block = MConv(16, 1, 3, 1, bias=True, dimension=3)

    class Compressor(nn.Module):
        def __init__(self):
            super(Compressor, self).__init__()
            # pay attention to receptive field in practice
            self.encoder = nn.Sequential(MConv(1, 8, 3, 1, dimension=3),
                                         MReLU(inplace=True),
                                         MConv(8, 16, 2, 2, dimension=3),
                                         MReLU(inplace=True),
                                         MConv(16, 16, 3, 1, dimension=3),
                                         MReLU(inplace=True),
                                         MConv(16, 16, 2, 2, dimension=3),
                                         MReLU(inplace=True),
                                         MConv(16, 16, 3, 1, dimension=3),
                                         MReLU(inplace=True),
                                         MConv(16, 16, 2, 2, dimension=3),
                                         MReLU(inplace=True),
                                         MConv(16, 16, 3, 1, dimension=3))

            self.decoder = nn.Sequential(GenerativeUpsample(),
                                         GenerativeUpsample())

        def forward(self, xyz):
            if self.training:
                encoded_xyz = self.encoder(xyz)
                out = self.decoder((encoded_xyz, None, None, xyz.coordinate_map_key))
                return out
            else:
                return None

    model = Compressor()
    xyz_c = [ME.utils.sparse_quantize(torch.rand((100, 3)) * 100) for _ in range(16)]
    xyz_f = [torch.ones((_.shape[0], 1), dtype=torch.float32) for _ in xyz_c]
    xyz = ME.utils.sparse_collate(coords=xyz_c, feats=xyz_f)
    xyz = ME.SparseTensor(coordinates=xyz[0], features=xyz[1], tensor_stride=1)

    out = model(xyz)
    print('Done')


if __name__ == '__main__':
    generative_upsample_t()
