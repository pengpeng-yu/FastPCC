from typing import Tuple, List, Dict, Union, Optional

import torch
try:
    from pytorch3d.ops import knn_points
except ImportError:
    knn_points = None
import MinkowskiEngine as ME
from torch import nn as nn

from lib.metrics.misc import precision_recall


class GenerativeUpsampleMessage:
    def __init__(self,
                 fea: ME.SparseTensor,
                 max_stride_lossy_recon: List[int],
                 target_key: Optional[ME.CoordinateMapKey] = None,
                 points_num_list: Optional[List[List[int]]] = None,
                 cached_fea_list: Optional[List[ME.SparseTensor]] = None):
        self.fea: ME.SparseTensor = fea
        self.max_stride_lossy_recon = max_stride_lossy_recon
        self.target_key: Optional[ME.CoordinateMapKey] = target_key
        self.points_num_list: Optional[List[List[int]]] = \
            points_num_list.copy() if points_num_list is not None else None
        self.cached_fea_list: List[ME.SparseTensor] = cached_fea_list or []
        self.cached_pred_list: List[ME.SparseTensor] = []
        self.cached_target_list: List[torch.Tensor] = []
        self.cached_metric_list: List[Dict[str, Union[int, float]]] = []


class GenerativeUpsample(nn.Module):
    def __init__(self,
                 upsample_block: nn.Module,
                 classify_block: nn.Module,

                 mapping_target_kernel_size=1,
                 mapping_target_region_type='HYPER_CUBE',
                 loss_type='BCE',
                 dist_upper_bound=2.0,
                 requires_metric_during_testing=False):
        super(GenerativeUpsample, self).__init__()
        self.upsample_block = upsample_block
        # classify_block should not change coordinates of upsample_block's output
        self.classify_block = classify_block
        self.mapping_target_kernel_size = mapping_target_kernel_size
        self.mapping_target_region_type = getattr(ME.RegionType, mapping_target_region_type)
        self.loss_type = loss_type
        self.square_dist_upper_bound = dist_upper_bound ** 2
        self.requires_metric_during_testing = requires_metric_during_testing
        self.pruning = ME.MinkowskiPruning()

    def forward(self, message: GenerativeUpsampleMessage):
        fea = message.fea
        fea = self.upsample_block(fea)
        pred = self.classify_block(fea)
        keep = self.get_keep(pred, message.points_num_list, message.max_stride_lossy_recon)

        if self.training:
            keep_target, loss_target = self.get_target(fea, pred, message.target_key, True)
            keep |= keep_target
            message.cached_target_list.append(loss_target)
            message.cached_pred_list.append(pred)

        elif not self.training:
            if self.requires_metric_during_testing:
                keep_target = self.get_target(fea, pred, message.target_key, False)
                message.cached_metric_list.append(
                    precision_recall(pred=keep, tgt=keep_target)
                )

        message.fea = self.pruning(fea, keep)
        return message

    @torch.no_grad()
    def get_keep(self, pred: ME.SparseTensor, points_num_list: List[List[int]],
                 max_stride_lossy_recon: List[int]) -> torch.Tensor:
        max_stride_coord_key = ME.CoordinateMapKey(max_stride_lossy_recon, '' if self.training else 'pruned')
        stride_scaler = [_ // __ for _, __ in zip(max_stride_coord_key.get_tensor_stride(), pred.tensor_stride)]
        pool = ME.MinkowskiMaxPooling(stride_scaler, stride_scaler, dimension=3).to(pred.device)
        un_pool = ME.MinkowskiPoolingTranspose(stride_scaler, stride_scaler, dimension=3).to(pred.device)
        pred_local_max = un_pool(pool(pred, max_stride_coord_key), pred.coordinate_map_key)
        local_max_mask = (pred.F - pred_local_max.F).squeeze(1) != 0
        if self.loss_type == 'BCE':
            if points_num_list is not None:
                target_points_num = points_num_list.pop()
                sample_threshold = []
                for sample_tgt, sample_permutation in zip(target_points_num, pred.decomposition_permutations):
                    sample = pred.F[sample_permutation]
                    assert sample.shape[0] > sample_tgt
                    sample_masked = sample[local_max_mask[sample_permutation]]
                    sample_threshold.append(
                        torch.kthvalue(sample_masked, sample.shape[0] - sample_tgt, dim=0).values)
                threshold = torch.tensor(sample_threshold, device=pred.F.device, dtype=pred.F.dtype)
                threshold = threshold[pred.C[:, 0].to(torch.long)]
            else:
                threshold = 0
            keep = (pred.F.squeeze(dim=1) > threshold)

        elif self.loss_type == 'Dist':
            if points_num_list is not None:
                target_points_num = points_num_list.pop()
                sample_threshold = []
                for sample_tgt, sample_permutation in zip(target_points_num, pred.decomposition_permutations):
                    sample = pred.F[sample_permutation]
                    assert sample.shape[0] > sample_tgt
                    sample_masked = sample[local_max_mask[sample_permutation]]
                    sample_threshold.append(
                        torch.kthvalue(sample, sample_tgt - (sample.shape[0] - sample_masked.shape[0]), dim=0).values)
                threshold = torch.tensor(sample_threshold, device=pred.F.device, dtype=pred.F.dtype)
                threshold = threshold[pred.C[:, 0].to(torch.long)]
            else:
                threshold = 0.5
            keep = (pred.F.squeeze(dim=1) <= threshold)

        else:
            raise NotImplementedError
        keep.logical_or_(~local_max_mask)
        return keep

    @torch.no_grad()
    def get_target(self,
                   fea: ME.SparseTensor,
                   pred: ME.SparseTensor,
                   target_key: ME.CoordinateMapKey,
                   requires_loss_target: bool) \
            -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        cm = fea.coordinate_manager
        strided_target_key = cm.stride(target_key, fea.tensor_stride)
        kernel_map = cm.kernel_map(
            fea.coordinate_map_key,
            strided_target_key,
            kernel_size=self.mapping_target_kernel_size,
            region_type=self.mapping_target_region_type
        )
        keep_target = torch.zeros(fea.shape[0], dtype=torch.bool, device=fea.device)
        for _, curr_in in kernel_map.items():
            keep_target[curr_in[0].type(torch.long)] = 1

        if requires_loss_target:
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

                pred_mask = pred.F.squeeze(dim=1) > self.square_dist_upper_bound
                target_mask = loss_target > self.square_dist_upper_bound
                bound_target_mask = (~pred_mask) & target_mask
                ignore_target_mask = pred_mask & target_mask
                loss_target[bound_target_mask] = self.square_dist_upper_bound
                loss_target[ignore_target_mask] = pred.F.squeeze(dim=1)[ignore_target_mask]

            else:
                raise NotImplementedError
            return keep_target, loss_target

        else:
            return keep_target
