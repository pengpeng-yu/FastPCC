from typing import Tuple, List, Dict, Union, Callable, Optional

import torch
try:
    from pytorch3d.ops import knn_points
except ImportError:
    knn_points = None
import MinkowskiEngine as ME
from torch import nn as nn

from lib.metrics.misc import precision_recall
from lib.entropy_models.continuous_indexed import ContinuousIndexedEntropyModel


def get_act_module(act: Union[str, nn.Module, None]) -> Optional[nn.Module]:
    if isinstance(act, nn.Module):
        act_module = act
    elif act is None or act == 'None':
        act_module = None
    elif act == 'relu':
        act_module = ME.MinkowskiReLU(inplace=True)
    elif act.startswith('leaky_relu'):
        act_module = ME.MinkowskiLeakyReLU(
            negative_slope=float(act.split('(', 1)[1].split(')', 1)[0]),
            inplace=True)
    elif act == 'sigmoid':
        act_module = ME.MinkowskiSigmoid()
    else:
        raise NotImplementedError
    return act_module


class BaseConvBlock(nn.Module):
    def __init__(self,
                 conv_class: Callable,
                 in_channels, out_channels, kernel_size, stride,
                 dilation=1, dimension=3,
                 region_type: str = 'HYPER_CUBE',
                 bn: bool = False,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(BaseConvBlock, self).__init__()

        self.region_type = getattr(ME.RegionType, region_type)

        self.conv = conv_class(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=not bn,
            kernel_generator=ME.KernelGenerator(
                kernel_size,
                stride,
                dilation,
                region_type=self.region_type,
                dimension=dimension),
            dimension=dimension
        )
        self.bn = ME.MinkowskiBatchNorm(out_channels) if bn else None
        self.act = act
        self.act_module = get_act_module(act)

    def forward(self, x, *args, **kwargs):
        x = self.conv(x, *args, **kwargs)
        if self.bn is not None:
            x = self.bn(x)
        if self.act_module is not None:
            x = self.act_module(x)
        return x

    def __repr__(self):
        return \
            f'{self.conv.__class__.__name__.replace("Minkowski", "ME", 1).replace("Convolution", "Conv", 1)}(' \
            f'in={self.conv.in_channels}, out={self.conv.out_channels}, ' \
            f'region_type={self.conv.kernel_generator.region_type}, ' \
            f'kernel_volume={self.conv.kernel_generator.kernel_volume}, ' \
            f'kernel_size={self.conv.kernel_generator.kernel_size}, ' \
            f'stride={self.conv.kernel_generator.kernel_stride}, ' \
            f'dilation={self.conv.kernel_generator.kernel_dilation}, ' \
            f'bn={self.bn is not None}, ' \
            f'act={self.act_module.__class__.__name__.replace("Minkowski", "ME", 1)})'


class ConvBlock(BaseConvBlock):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride,
                 dilation=1, dimension=3,
                 region_type: str = 'HYPER_CUBE',
                 bn: bool = False,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(ConvBlock, self).__init__(
            ME.MinkowskiConvolution,
            in_channels, out_channels, kernel_size, stride,
            dilation, dimension,
            region_type, bn, act
        )


class ConvTransBlock(BaseConvBlock):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride,
                 dilation=1, dimension=3,
                 region_type: str = 'HYPER_CUBE',
                 bn: bool = False,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(ConvTransBlock, self).__init__(
            ME.MinkowskiConvolutionTranspose,
            in_channels, out_channels, kernel_size, stride,
            dilation, dimension,
            region_type, bn, act
        )


class GenConvTransBlock(BaseConvBlock):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride,
                 dilation=1, dimension=3,
                 region_type: str = 'HYPER_CUBE',
                 bn: bool = False,
                 act: Union[str, nn.Module, None] = 'relu'):
        super(GenConvTransBlock, self).__init__(
            ME.MinkowskiGenerativeConvolutionTranspose,
            in_channels, out_channels, kernel_size, stride,
            dilation, dimension,
            region_type, bn, act
        )


class GenerativeUpsampleMessage:
    def __init__(self,
                 fea: ME.SparseTensor,
                 target_key: Optional[ME.CoordinateMapKey] = None,
                 points_num_list: Optional[List[List[int]]] = None,
                 cached_fea_list: Optional[List[Union[ME.SparseTensor]]] = None,
                 em_bytes_list: Optional[List[bytes]] = None):
        self.fea: ME.SparseTensor = fea
        self.target_key: Optional[ME.CoordinateMapKey] = target_key
        self.points_num_list: Optional[List[List[int]]] = \
            points_num_list.copy() if points_num_list is not None else None
        self.cached_fea_list: List[Union[ME.SparseTensor]] = cached_fea_list or []
        self.cached_pred_list: List[Union[ME.SparseTensor]] = []
        self.cached_target_list: List[ME.SparseTensor] = []
        self.cached_metric_list: List[Dict[str, Union[int, float]]] = []
        self.post_fea_hook: Optional[Callable] = None
        self.indexed_em: Optional[ContinuousIndexedEntropyModel] = None
        self.em_loss_dict_list: List[Dict[str, torch.Tensor]] = []
        self.em_flag: str = ''
        self.em_hybrid_hyper_decoder: bool = False
        self.em_decoder_aware_residuals: bool = False
        self.em_upper_fea_grad_scaler: float = 0.0
        self.em_bytes_list: List[bytes] = em_bytes_list or []


class GenerativeUpsample(nn.Module):
    def __init__(self,
                 upsample_block: nn.Module,
                 classify_block: nn.Module,
                 predict_block: nn.Module = None,
                 residual_block: nn.Module = None,

                 mapping_target_kernel_size=1,
                 mapping_target_region_type='HYPER_CUBE',
                 loss_type='BCE',
                 dist_upper_bound=2.0,
                 requires_metric_during_testing=False):
        super(GenerativeUpsample, self).__init__()
        self.upsample_block = upsample_block
        # classify_block should not change coordinates of upsample_block's output
        self.classify_block = classify_block
        self.predict_block = predict_block
        self.residual_block = residual_block
        self.mapping_target_kernel_size = mapping_target_kernel_size
        self.mapping_target_region_type = getattr(ME.RegionType, mapping_target_region_type)
        self.loss_type = 'BCE' if loss_type == 'Focal' else loss_type
        self.square_dist_upper_bound = dist_upper_bound ** 2
        self.requires_metric_during_testing = requires_metric_during_testing
        self.use_residual = self.predict_block is not None and self.residual_block is not None
        self.pruning = ME.MinkowskiPruning()

    def forward(self, message: GenerativeUpsampleMessage):
        """
        fea: SparseTensor,
        cached_pred_list: List[SparseTensor], prediction of existence or distance,
        cached_target_list: List[SparseTensor] or None,
        target_key: MinkowskiEngine.CoordinateMapKey or None.

        During training, features from last layer are used to generate new features.
        Existence prediction of each upsample layer and corresponding target is used for loss computation.
        Target_key(identical for all upsample layers) is for generating target of each upsample layer.

        During testing, cached_target_list and target_key are no longer needed.

        Only supports batch size == 1 during testing if self.use_residual.
        """
        fea = message.fea
        fea = self.upsample_block(fea)
        pred = self.classify_block(fea)
        keep = self.get_keep(pred, message.points_num_list)

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

        if self.use_residual:
            cm = message.fea.coordinate_manager
            cm_key = message.fea.coordinate_map_key
            if self.training:
                cached_feature = message.cached_fea_list.pop()
                if message.em_hybrid_hyper_decoder is True:
                    fea_info_pred = self.predict_block(message.fea).F
                    fea_pred, fea_indexes = torch.split(
                        fea_info_pred,
                        [cached_feature.shape[1], fea_info_pred.shape[1] - cached_feature.shape[1]],
                        dim=1
                    )
                    if message.em_decoder_aware_residuals is True:
                        union = ME.MinkowskiUnion()
                        cached_feature = union(
                            ME.SparseTensor(
                                torch.cat((cached_feature.F, torch.zeros_like(cached_feature.F)), 1),
                                coordinate_map_key=cached_feature.coordinate_map_key,
                                coordinate_manager=cm
                            ),
                            ME.SparseTensor(
                                torch.cat((torch.zeros_like(fea_pred), fea_pred), 1),
                                coordinate_map_key=cm_key,
                                coordinate_manager=cm
                            )
                        )
                    fea = self.residual_block(cached_feature, coordinates=cm_key).F
                    assert fea.shape[1] * (
                        len(message.indexed_em.index_ranges) + 1
                    ) == fea_info_pred.shape[1]
                    if message.post_fea_hook is not None:
                        fea = message.post_fea_hook(fea)
                        fea_pred = message.post_fea_hook(fea_pred)
                    fea_pred_res_tilde, fea_loss_dict = message.indexed_em(
                        (fea - fea_pred), fea_indexes,
                        is_first_forward=len(message.em_loss_dict_list) == 0,  # Assume that the EM is shared
                        x_grad_scaler_for_bits_loss=message.em_upper_fea_grad_scaler
                    )
                    fea_tilde = ME.SparseTensor(
                        features=fea_pred_res_tilde + fea_pred,
                        coordinate_map_key=cm_key,
                        coordinate_manager=cm
                    )
                else:
                    fea = self.residual_block(cached_feature, coordinates=cm_key)
                    if message.post_fea_hook is not None:
                        fea = message.post_fea_hook(fea)
                    fea_indexes = self.predict_block(message.fea)
                    fea_tilde, fea_loss_dict = message.indexed_em(
                        fea, fea_indexes, is_first_forward=len(message.em_loss_dict_list) == 0,
                        x_grad_scaler_for_bits_loss=message.em_upper_fea_grad_scaler
                    )
                message.fea = fea_tilde
                message.em_loss_dict_list.append(fea_loss_dict)

            elif message.em_flag == 'compress':
                cached_feature = message.cached_fea_list.pop()
                if message.em_hybrid_hyper_decoder is True:
                    fea_info_pred = self.predict_block(message.fea).F
                    fea_pred, fea_indexes = torch.split(
                        fea_info_pred,
                        [cached_feature.shape[1], fea_info_pred.shape[1] - cached_feature.shape[1]],
                        dim=1
                    )
                    if message.em_decoder_aware_residuals is True:
                        union = ME.MinkowskiUnion()
                        cached_feature = union(
                            ME.SparseTensor(
                                torch.cat((cached_feature.F, torch.zeros_like(cached_feature.F)), 1),
                                coordinate_map_key=cached_feature.coordinate_map_key,
                                coordinate_manager=cm
                            ),
                            ME.SparseTensor(
                                torch.cat((torch.zeros_like(fea_pred), fea_pred), 1),
                                coordinate_map_key=cm_key,
                                coordinate_manager=cm
                            )
                        )
                    fea = self.residual_block(cached_feature, coordinates=cm_key).F
                    if message.post_fea_hook is not None:
                        fea = message.post_fea_hook(fea)
                        fea_pred = message.post_fea_hook(fea_pred)
                    (fea_bytes,), fea_pred_res_recon = message.indexed_em.compress(
                        fea - fea_pred, fea_indexes,
                    )
                    fea_recon = ME.SparseTensor(
                        features=fea_pred_res_recon + fea_pred,
                        coordinate_map_key=cm_key,
                        coordinate_manager=cm
                    )
                else:
                    fea = self.residual_block(cached_feature, coordinates=cm_key)
                    if message.post_fea_hook is not None:
                        fea = message.post_fea_hook(fea)
                    fea_indexes = self.predict_block(message.fea)
                    (fea_bytes,), fea_recon = message.indexed_em.compress(
                        fea, fea_indexes
                    )
                message.fea = fea_recon
                message.em_bytes_list.append(fea_bytes)

            elif message.em_flag == 'decompress':
                fea_bytes = message.em_bytes_list.pop(0)
                if message.em_hybrid_hyper_decoder is True:
                    fea_info_pred = self.predict_block(message.fea).F
                    fea_recon_channels = fea_info_pred.shape[1] // (
                        len(message.indexed_em.index_ranges) + 1
                    )
                    fea_pred, fea_indexes = torch.split(
                        fea_info_pred,
                        [fea_recon_channels, fea_info_pred.shape[1] - fea_recon_channels], dim=1
                    )
                    if message.post_fea_hook is not None:
                        fea_pred = message.post_fea_hook(fea_pred)
                    fea_pred_res_recon = message.indexed_em.decompress(
                        [fea_bytes], fea_indexes, next(self.parameters()).device
                    )
                    fea_recon = ME.SparseTensor(
                        features=fea_pred_res_recon + fea_pred,
                        coordinate_map_key=cm_key,
                        coordinate_manager=cm
                    )
                else:
                    fea_indexes = self.predict_block(message.fea)
                    fea_recon = message.indexed_em.decompress(
                        [fea_bytes], fea_indexes, next(self.parameters()).device,
                        sparse_tensor_coords_tuple=(cm_key, fea_indexes.coordinate_manager)
                    )
                message.fea = fea_recon

        elif self.predict_block is not None:  # if not self.use_residual
            fea_pred = self.predict_block(message.fea)
            if message.post_fea_hook is not None:
                message.fea = message.post_fea_hook(fea_pred)

        return message

    @torch.no_grad()
    def get_keep(self, pred: ME.SparseTensor, points_num_list: List[List[int]]) \
            -> torch.Tensor:
        if self.loss_type == 'BCE':
            if points_num_list is not None:
                target_points_num = points_num_list.pop()
                sample_threshold = []
                for sample_tgt, sample in zip(target_points_num, pred.decomposed_features):
                    if sample.shape[0] > sample_tgt:
                        sample_threshold.append(torch.kthvalue(sample, sample.shape[0] - sample_tgt, dim=0).values)
                    else:
                        sample_threshold.append(torch.finfo(sample.dtype).min)
                threshold = torch.tensor(sample_threshold, device=pred.F.device, dtype=pred.F.dtype)
                threshold = threshold[pred.C[:, 0].to(torch.long)]
            else:
                threshold = 0
            keep = (pred.F.squeeze(dim=1) > threshold)

        elif self.loss_type == 'Dist':
            if points_num_list is not None:
                target_points_num = points_num_list.pop()
                sample_threshold = []
                for sample_tgt, sample in zip(target_points_num, pred.decomposed_features):
                    if sample.shape[0] > sample_tgt:
                        sample_threshold.append(torch.kthvalue(sample, sample_tgt, dim=0).values)
                    else:
                        sample_threshold.append(torch.finfo(sample.dtype).max)
                threshold = torch.tensor(sample_threshold, device=pred.F.device, dtype=pred.F.dtype)
                threshold = threshold[pred.C[:, 0].to(torch.long)]
            else:
                threshold = 0.5
            keep = (pred.F.squeeze(dim=1) <= threshold)

        else:
            raise NotImplementedError
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


def generative_upsample_t():
    class DecoderBlock(nn.Module):
        def __init__(self, **kwargs):
            super(DecoderBlock, self).__init__()
            upsample_block = nn.Sequential(
                GenConvTransBlock(16, 16, 2, 2, bn=False, act='relu'),
                ConvBlock(16, 16, 3, 1, bn=False, act='relu')
            )
            classify_block = ConvBlock(16, 1, 3, 1, bn=False, act=None)
            self.generative_upsample = GenerativeUpsample(
                upsample_block, classify_block, **kwargs
            )

        def forward(self, x: GenerativeUpsampleMessage):
            return self.generative_upsample(x)

    class Compressor(nn.Module):
        def __init__(self):
            super(Compressor, self).__init__()
            self.encoder = nn.Sequential(
                ConvBlock(1, 8, 3, 1),
                ConvBlock(8, 16, 2, 2),
                ConvBlock(16, 16, 3, 1),
                ConvBlock(16, 16, 2, 2),
                ConvBlock(16, 16, 3, 1),
            )
            self.decoder = nn.Sequential(
                DecoderBlock(), DecoderBlock()
            )

        def forward(self, xyz):
            if self.training:
                encoded_xyz = self.encoder(xyz)
                out = self.decoder(
                    GenerativeUpsampleMessage(
                        fea=encoded_xyz,
                        target_key=xyz.coordinate_map_key
                    )
                )
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
