from typing import Tuple, Union, Dict, Any, Callable
from functools import partial

import torch
import torch.nn as nn
from torch.distributions import Distribution
import MinkowskiEngine as ME

from lib.entropy_models.hyperprior.noisy_deep_factorized.basic import EntropyModel as HyperPriorEntropyModel
from lib.entropy_models.continuous_batched import ContinuousBatchedEntropyModel
from lib.entropy_models.continuous_indexed import ContinuousIndexedEntropyModel, \
    noisy_deep_factorized_indexed_entropy_model_init
from lib.entropy_models.distributions.uniform_noise import NoisyDeepFactorized
from lib.entropy_models.hyperprior.noisy_deep_factorized.utils import BytesListUtils

from lib.torch_utils import concat_loss_dicts


class GeoLosslessEntropyModel(nn.Module):
    def __init__(self,
                 bottom_fea_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 encoder: nn.Module,

                 hyper_decoder_fea: Union[nn.Module, nn.ModuleList],
                 hybrid_hyper_decoder_fea: bool,

                 fea_prior_fn: Callable[..., Distribution],
                 fea_index_ranges: Tuple[int, ...],
                 fea_parameter_fns: Dict[str, Callable[[torch.Tensor], Any]],

                 hyper_decoder_fea_post_op: Callable = lambda x: x,

                 bottleneck_fea_process: str = 'noise',
                 bottleneck_scaler: int = 1,
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 indexes_scaler: float = 1,
                 init_scale: float = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        super(GeoLosslessEntropyModel, self).__init__()
        self.bottom_fea_entropy_model = bottom_fea_entropy_model
        self.encoder = encoder
        self.hyper_decoder_fea = hyper_decoder_fea
        self.hybrid_hyper_decoder_fea = hybrid_hyper_decoder_fea
        self.hyper_decoder_fea_post_op = hyper_decoder_fea_post_op
        self.indexed_entropy_model_fea = ContinuousIndexedEntropyModel(
            prior_fn=fea_prior_fn,
            index_ranges=fea_index_ranges,
            parameter_fns=fea_parameter_fns,
            coding_ndim=2,
            bottleneck_process=bottleneck_fea_process,
            bottleneck_scaler=bottleneck_scaler,
            indexes_bound_gradient=indexes_bound_gradient,
            quantize_indexes=quantize_indexes,
            indexes_scaler=indexes_scaler,
            init_scale=init_scale,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision
        )

    def get_sub_hyper_decoder_fea(self, idx):
        if hasattr(self.hyper_decoder_fea, '__getitem__'):
            return self.hyper_decoder_fea[idx]
        else:
            return self.hyper_decoder_fea

    def forward(self, y_top: ME.SparseTensor, *args, **kwargs):
        cm = y_top.coordinate_manager
        strided_fea_list = self.encoder(y_top, *args, **kwargs)
        *strided_fea_list, bottom_fea = strided_fea_list
        strided_fea_list_len = len(strided_fea_list)
        loss_dict = {}
        bytes_list = []
        strided_fea_tilde_list = []

        if self.training:
            bottom_fea_tilde, fea_loss_dict_or_bytes_list = self.bottom_fea_entropy_model(bottom_fea)
        else:
            bottom_fea_tilde, fea_loss_dict_or_bytes_list, coding_batch_shape = \
                self.bottom_fea_entropy_model(bottom_fea)
            if coding_batch_shape != torch.Size([1]): raise NotImplementedError
        strided_fea_tilde_list.append(bottom_fea_tilde)
        if self.training:
            concat_loss_dicts(loss_dict, fea_loss_dict_or_bytes_list, lambda k: 'fea_bottom_' + k)
        else:
            bytes_list.extend(fea_loss_dict_or_bytes_list)
        lower_fea_tilde = bottom_fea_tilde

        for idx in range(strided_fea_list_len - 1, -1, -1):
            sub_hyper_decoder_fea = self.get_sub_hyper_decoder_fea(idx)
            fea = strided_fea_list[idx]
            coord_target_key = fea.coordinate_map_key

            if self.hybrid_hyper_decoder_fea is True:
                fea_info_pred = sub_hyper_decoder_fea(lower_fea_tilde, coord_target_key).F
                assert fea.F.shape[1] * (
                    len(self.indexed_entropy_model_fea.index_ranges) + 1
                ) == fea_info_pred.shape[1]
                fea_pred, pre_fea_indexes = torch.split(
                    fea_info_pred,
                    [fea.F.shape[1], fea_info_pred.shape[1] - fea.F.shape[1]],
                    dim=1
                )
                fea_indexes = self.hyper_decoder_fea_post_op(pre_fea_indexes)[None]
                fea_pred_res_tilde, fea_loss_dict_or_bytes_list = self.indexed_entropy_model_fea(
                    (fea.F - fea_pred)[None], fea_indexes, is_first_forward=idx == strided_fea_list_len - 1
                )
                lower_fea_tilde = ME.SparseTensor(
                    features=fea_pred_res_tilde[0] + fea_pred,
                    coordinate_map_key=coord_target_key,
                    coordinate_manager=cm
                )
            else:
                fea_indexes = self.hyper_decoder_fea_post_op(
                    sub_hyper_decoder_fea(lower_fea_tilde, coord_target_key)
                )
                fea_tilde, fea_loss_dict_or_bytes_list = self.indexed_entropy_model_fea(
                    fea.F[None], fea_indexes, is_first_forward=idx == strided_fea_list_len - 1
                )
                lower_fea_tilde = ME.SparseTensor(
                    features=fea_tilde[0],
                    coordinate_map_key=coord_target_key,
                    coordinate_manager=cm
                )
            strided_fea_tilde_list.append(lower_fea_tilde)
            if self.training:
                concat_loss_dicts(loss_dict, fea_loss_dict_or_bytes_list, lambda k: f'fea_{idx}_' + k)
            else:
                bytes_list.extend(fea_loss_dict_or_bytes_list)

        return strided_fea_tilde_list[-1], loss_dict if self.training \
            else BytesListUtils.concat_bytes_list(bytes_list)


class GeoLosslessNoisyDeepFactorizedEntropyModel(GeoLosslessEntropyModel):
    def __init__(self,
                 bottom_fea_entropy_model:
                 Union[ContinuousBatchedEntropyModel, HyperPriorEntropyModel],
                 encoder: nn.Module,

                 hyper_decoder_fea: Union[nn.Module, nn.ModuleList],
                 hybrid_hyper_decoder_fea: bool,

                 fea_index_ranges: Tuple[int, ...] = (16, 16, 16, 16),
                 fea_parameter_fns_type: str = 'transform',
                 fea_parameter_fns_factory: Callable[..., nn.Module] = None,
                 fea_num_filters: Tuple[int, ...] = (1, 3, 3, 3, 1),

                 bottleneck_fea_process: str = 'noise',
                 bottleneck_scaler: int = 1,
                 indexes_bound_gradient: str = 'identity_if_towards',
                 quantize_indexes: bool = False,
                 indexes_scaler: float = 1,
                 init_scale: float = 10,
                 tail_mass: float = 2 ** -8,
                 range_coder_precision: int = 16
                 ):
        fea_parameter_fns, fea_indexes_view_fn, fea_modules_to_add = \
            noisy_deep_factorized_indexed_entropy_model_init(
                fea_index_ranges, fea_parameter_fns_type,
                fea_parameter_fns_factory, fea_num_filters
            )
        super(GeoLosslessNoisyDeepFactorizedEntropyModel, self).__init__(
            bottom_fea_entropy_model, encoder,
            hyper_decoder_fea, hybrid_hyper_decoder_fea,
            partial(NoisyDeepFactorized, noise_width=1 / bottleneck_scaler),
            fea_index_ranges, fea_parameter_fns, fea_indexes_view_fn,
            bottleneck_fea_process, bottleneck_scaler,
            indexes_bound_gradient, quantize_indexes, indexes_scaler,
            init_scale, tail_mass, range_coder_precision
        )
        for module_name, module in fea_modules_to_add.items():
            setattr(self, 'fea_' + module_name, module)

    def _apply(self, fn):
        super(GeoLosslessNoisyDeepFactorizedEntropyModel, self)._apply(fn)
        self.indexed_entropy_model_fea.update_prior()
