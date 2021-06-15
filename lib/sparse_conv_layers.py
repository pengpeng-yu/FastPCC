from typing import Tuple, List, Union, Optional, Any

import torch
import torch.nn as nn
from pytorch3d.ops import knn_points
import MinkowskiEngine as ME

from lib.torch_utils import unbatched_coordinates

MConv = ME.MinkowskiConvolution
MReLU = ME.MinkowskiReLU
MGenConvTranspose = ME.MinkowskiGenerativeConvolutionTranspose
MConvTranspose = ME.MinkowskiConvolutionTranspose


class GenerativeUpsampleMessage:
    def __init__(self,
                 fea: ME.SparseTensor,
                 target_key: ME.CoordinateMapKey = None,
                 points_num_list: List[int] = None,
                 cached_pred: Optional[List[ME.SparseTensor]] = None,
                 cached_target: Optional[List[ME.SparseTensor]] = None):
        self.fea = fea
        self.target_key = target_key
        self.points_num_list = points_num_list.copy() if points_num_list is not None else None
        self.cached_pred = cached_pred or []
        self.cached_target = cached_target or []


class GenerativeUpsample(nn.Module):
    def __init__(self, upsample_block: nn.Module, classify_block: nn.Module,
                 mapping_target_kernel_size=1, loss_type='BCE', dist_upper_bound=2.0, is_last_layer=False):
        super(GenerativeUpsample, self).__init__()
        self.mapping_target_kernel_size = mapping_target_kernel_size
        assert loss_type in ['BCE', 'Dist']
        self.loss_type = loss_type
        self.square_dist_upper_bound = dist_upper_bound ** 2
        self.is_last_layer = is_last_layer
        self.upsample_block = upsample_block
        self.classify_block = classify_block  # classify_block should not change coordinates of upsample_block's output
        # It will consume huge memory if too many unnecessary points are retained after pruning
        self.pruning = ME.MinkowskiPruning()

    def forward(self, message: GenerativeUpsampleMessage):
        """
        fea: SparseTensor,
        cached_pred: List[SparseTensor], prediction of existence or distance,
        cached_target: List[SparseTensor] or None,
        target_key: MinkowskiEngine.CoordinateMapKey or None.

        During training, features from last layer are used to generate new features.
        Existence prediction of each upsample layer and cooresponding target is used for loss computation.
        Target_key(identical for all upsample layers) is for generating target of each upsample layer.

        During testing, cached_target and target_key are no longer needed.
        """
        fea = message.fea

        fea = self.upsample_block(fea)
        pred = self.classify_block(fea)

        with torch.no_grad():
            if self.loss_type == 'BCE':
                if message.points_num_list is not None:
                    target_points_num = message.points_num_list.pop()
                    if pred.F.shape[0] > target_points_num:
                        thres = torch.kthvalue(pred.F, pred.F.shape[0] - target_points_num, dim=0).values
                        keep = (pred.F.squeeze() > thres)
                    else:
                        keep = torch.full_like(pred.F.squeeze(), fill_value=True, dtype=torch.bool)
                else:
                    keep = (pred.F.squeeze() > 0)

            elif self.loss_type == 'Dist':
                if message.points_num_list is not None:
                    target_points_num = message.points_num_list.pop()
                    if pred.F.shape[0] > target_points_num:
                        thres = torch.kthvalue(pred.F, target_points_num, dim=0).values
                        keep = (pred.F.squeeze() <= thres)
                    else:
                        keep = torch.full_like(pred.F.squeeze(), fill_value=True, dtype=torch.bool)
                else:
                    keep = (pred.F.squeeze() < 0.5)

            else:
                raise NotImplementedError

        if self.training:
            keep_target, loss_target = self.get_target(fea, pred, message.target_key, True)

            keep |= keep_target

            message.cached_target.append(loss_target)
            message.cached_pred.append(pred)

            if not self.is_last_layer:
                message.fea = self.pruning(fea, keep)
            else:
                message.fea = None

        elif not self.training:
            if not self.is_last_layer:
                message.fea = self.pruning(fea, keep)
            else:
                message.cached_pred = [self.pruning(pred, keep)]

        return message

    def get_target(self, fea, pred, target_key, require_loss_target):
        # type: (ME.SparseTensor, ME.SparseTensor, ME.CoordinateMapKey, bool) -> Any
        cm = fea.coordinate_manager

        strided_target_key = cm.stride(target_key, fea.tensor_stride)
        kernel_map = cm.kernel_map(fea.coordinate_map_key,
                                   strided_target_key,
                                   kernel_size=self.mapping_target_kernel_size,
                                   region_type=1)
        keep_target = torch.zeros(fea.shape[0], dtype=torch.bool, device=fea.device)
        for _, curr_in in kernel_map.items():
            keep_target[curr_in[0].type(torch.int64)] = 1

        if require_loss_target:
            if self.loss_type == 'BCE':
                loss_target = keep_target

            elif self.loss_type == 'Dist':
                loss_target = torch.zeros(fea.shape[0], dtype=torch.float, device=fea.device)
                strided_target = cm.get_coordinates(strided_target_key)

                for sample_idx in range(strided_target[:, 0].max().item() + 1):
                    strided_target_one_sample = strided_target[strided_target[:, 0] == sample_idx][:, 1:]

                    sample_mapping = fea.C[:, 0] == sample_idx
                    pred_coord_one_sample = pred.C[sample_mapping][:, 1:]

                    dists = knn_points(pred_coord_one_sample[None].type(torch.float),
                                       strided_target_one_sample[None].type(torch.float),
                                       K=1, return_sorted=False).dists[0, :, 0]
                    loss_target[sample_mapping] = dists

                with torch.no_grad():  # TODO: speed
                    pred_mask = pred.F.squeeze() > self.square_dist_upper_bound
                    target_mask = loss_target > self.square_dist_upper_bound
                    bound_target_mask = (~pred_mask) & target_mask
                    ignore_target_mask = pred_mask & target_mask
                    loss_target[bound_target_mask] = self.square_dist_upper_bound
                    loss_target[ignore_target_mask] = pred.F.squeeze()[ignore_target_mask]

            else:
                raise NotImplementedError

            return keep_target, loss_target

        else:
            return keep_target


def generative_upsample_t():
    class DecoderBlock(nn.Module):
        def __init__(self, mapping_target_kernel_size=1):
            super(DecoderBlock, self).__init__()
            upsample_block = nn.Sequential(MGenConvTranspose(16, 16, 2, 2, bias=True, dimension=3),
                                           MReLU(inplace=True),
                                           MConv(16, 16, 3, 1, bias=True, dimension=3),
                                           MReLU(inplace=True))
            classify_block = MConv(16, 1, 3, 1, bias=True, dimension=3)
            self.generative_upsample = GenerativeUpsample(upsample_block, classify_block,
                                                          mapping_target_kernel_size=mapping_target_kernel_size)

        def forward(self, x: GenerativeUpsampleMessage):
            return self.generative_upsample(x)

    class Compressor(nn.Module):
        def __init__(self):
            super(Compressor, self).__init__()
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

            self.decoder = nn.Sequential(DecoderBlock(),
                                         DecoderBlock())

        def forward(self, xyz):
            if self.training:
                encoded_xyz = self.encoder(xyz)
                out = self.decoder(GenerativeUpsampleMessage(fea=encoded_xyz,
                                                             target_key=xyz.coordinate_map_key))
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
