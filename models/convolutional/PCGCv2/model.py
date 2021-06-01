import os
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from compressai.entropy_models import EntropyBottleneck
from compressai.models.utils import update_registered_buffers

from lib.loss_function import chamfer_loss
from models.convolutional.PCGCv2.layers import Encoder, GenerativeUpsample
from models.convolutional.PCGCv2.model_config import ModelConfig


class PCC(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(PCC, self).__init__()
        self.encoder = Encoder(cfg.compressed_channels,
                               res_blocks_num=3,
                               res_block_type=cfg.res_block_type)
        self.decoder = self.layers = nn.Sequential(GenerativeUpsample(cfg.compressed_channels,
                                                                      64, 3, cfg.res_block_type),
                                                   GenerativeUpsample(64, 32, 3, cfg.res_block_type),
                                                   GenerativeUpsample(32, 16, 3, cfg.res_block_type, return_fea=False))
        self.entropy_bottleneck = EntropyBottleneck(cfg.compressed_channels)
        self.cfg = cfg
        self.log_pred_res('init')

    def log_pred_res(self, mode, preds=None, targets=None,
                     file_path_list: str = None, compressed_strings: bytes = None,
                     voxels_num: int = None, decoder_output: torch.Tensor = None):
        if mode == 'init':
            total_reconstruct_loss = torch.zeros((1, ), dtype=torch.float64)
            samples_num = torch.zeros((1, ), dtype=torch.int64)
            self.register_buffer('total_reconstruct_loss', total_reconstruct_loss)
            self.register_buffer('samples_num', samples_num)

        elif mode == 'reset':
            self.total_reconstruct_loss[...] = 0
            self.samples_num[...] = 0

        elif mode == 'log':
            assert not self.training
            assert isinstance(preds, list) and isinstance(targets, list)

            for pred, target, file_path in zip(preds, targets, file_path_list):
                self.total_reconstruct_loss += chamfer_loss(
                    pred.unsqueeze(0).type(torch.float) / self.cfg.resolution,
                    target.unsqueeze(0).type(torch.float) / self.cfg.resolution)

                if self.cfg.save_path_test_phase != '':
                    file_path = os.path.join(self.cfg.save_path_test_phase, file_path)
                    file_path = os.path.splitext(file_path)[0]
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    compressed_path = file_path + '.txt'
                    headinfo_path = file_path + '_head.txt'
                    reconstructed_path = file_path + '.ply'

                    with open(compressed_path, 'wb') as f:
                        f.write(compressed_strings)
                    with open(headinfo_path, 'w') as f:
                        f.write(f'voxels_num: {voxels_num}\nscaler: {self.cfg.bottleneck_scaler}\n')

                    o3d.io.write_point_cloud(reconstructed_path,
                                             o3d.geometry.PointCloud(
                                                 o3d.utility.Vector3dVector(
                                                     pred.detach().clone().cpu())), write_ascii=True)

            self.samples_num += len(targets)

        elif mode == 'show':
            return {'samples_num': self.samples_num.item(),
                    'mean_recontruct_loss': (self.total_reconstruct_loss / self.samples_num).item()}
        else:
            raise NotImplementedError

    def forward(self, x):
        xyz, file_path_list = x
        xyz = ME.SparseTensor(torch.ones(xyz.shape[0], 1, dtype=torch.float, device=xyz.device),
                              xyz)
        fea = self.encoder(xyz)

        if self.training:
            # TODO: scaler?
            fea_tilde, likelihood = self.entropy_bottleneck(fea.F.T.unsqueeze(0) * self.cfg.bottleneck_scaler)
            fea_tilde = fea_tilde / self.cfg.bottleneck_scaler
            fea_tilde = ME.SparseTensor(fea_tilde.squeeze(0).T,
                                        coordinate_map_key=fea.coordinate_map_key,
                                        coordinate_manager=fea.coordinate_manager)

            _, cached_exist, cached_target, _ = \
                self.decoder((fea_tilde, None, None, xyz.coordinate_map_key))

            bpp_loss = torch.log2(likelihood).sum() * (
                        -self.cfg.bpp_loss_factor / xyz.shape[0])
            bce_loss = sum([nn.functional.binary_cross_entropy_with_logits(exist.F.squeeze(),
                                                                           target.type(exist.F.dtype))
                            for exist, target in zip(cached_exist, cached_target)])
            bce_loss /= len(cached_exist)
            aux_loss = self.entropy_bottleneck.loss() * self.cfg.aux_loss_factor
            loss = bpp_loss + bce_loss + aux_loss

            return {'loss': loss,
                    'bpp_loss': bpp_loss.detach().cpu().item(),
                    'bce_loss': bce_loss.detach().cpu().item(),
                    'aux_loss': aux_loss.detach().cpu().item()}

        else:
            # TODO: decomposed_coordinates_and_features before compressing?
            compressed_strings = self.entropy_bottleneck_compress(fea.F * self.cfg.bottleneck_scaler)
            decompressed_tensors = self.entropy_bottleneck_decompress(compressed_strings, fea.shape[0])
            fea = ME.SparseTensor(decompressed_tensors / self.cfg.bottleneck_scaler,
                                  coordinate_map_key=fea.coordinate_map_key,
                                  coordinate_manager=fea.coordinate_manager)
            _, cached_exist, _, _ = \
                self.decoder((fea, None, None, None))

            decoder_output = cached_exist[-1].decomposed_coordinates
            self.log_pred_res('log', decoder_output, xyz.decomposed_coordinates,
                              file_path_list, compressed_strings[0], fea.shape[0], decoder_output)

            return True

    def load_state_dict(self, state_dict, strict: bool = True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    def entropy_bottleneck_compress(self, encoder_output):
        assert not self.training
        encoder_output = encoder_output.T[None, :, :, None].contiguous()
        return self.entropy_bottleneck.compress(encoder_output)

    def entropy_bottleneck_decompress(self, compressed_strings, points_num):
        assert not self.training
        decompressed_tensors = self.entropy_bottleneck.decompress(compressed_strings, size=(points_num, 1))
        decompressed_tensors = decompressed_tensors.squeeze().T
        return decompressed_tensors


def main_debug():
    data_batch_0 = [
        [0, 0, 2.1, 0, 0],  #
        [0, 1, 1.4, 3, 0],  #
        [0, 0, 4.0, 0, 0]
    ]

    data_batch_1 = [
        [1, 0, 0],  #
        [0, 2, 0],  #
        [0, 0, 3]
    ]

    def to_sparse_coo(data):
        # An intuitive way to extract coordinates and features
        coords, feats = [], []
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                if val != 0:
                    coords.append([i, j])
                    feats.append([val])
        return torch.IntTensor(coords), torch.FloatTensor(feats)

    def sparse_tensor_initialization():
        coords, feats = to_sparse_coo(data_batch_0)
        # collate sparse tensor data to augment batch indices
        # Note that it is wrapped inside a list!!
        coords, feats = ME.utils.sparse_collate(coords=[coords], feats=[feats])
        sparse_tensor = ME.SparseTensor(coordinates=coords, features=feats)

    # noinspection PyPep8Naming
    def sparse_tensor_arithmetics():
        coords0, feats0 = to_sparse_coo(data_batch_0)
        coords0, feats0 = ME.utils.sparse_collate(coords=[coords0], feats=[feats0])

        coords1, feats1 = to_sparse_coo(data_batch_1)
        coords1, feats1 = ME.utils.sparse_collate(coords=[coords1], feats=[feats1])

        # sparse tensors
        A = ME.SparseTensor(coordinates=coords0, features=feats0)
        B = ME.SparseTensor(coordinates=coords1, features=feats1)

        # The following fails
        try:
            C = A + B
        except AssertionError:
            pass

        B = ME.SparseTensor(
            coordinates=coords1,
            features=feats1,
            coordinate_manager=A.coordinate_manager,  # must share the same coordinate manager
            # force_creation=True  # must force creation since tensor stride [1] exists
        )

        C = A + B
        C = A - B
        C = A * B
        C = A / B

        # in place operations
        # Note that it requires the same coords_key (no need to feed coords)
        D = ME.SparseTensor(
            # coords=coords,  not required
            features=feats0,
            coordinate_manager=A.coordinate_manager,  # must share the same coordinate manager
            coordinate_map_key=A.coordinate_map_key  # For inplace, must share the same coords key
        )

        A += D
        A -= D
        A *= D
        A /= D

        # If you have two or more sparse tensors with the same coords_key, you can concatenate features
        E = ME.cat(A, D)

    def operation_mode():
        # Set to share the coords_man by default
        ME.set_sparse_tensor_operation_mode(
            ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
        print(ME.sparse_tensor_operation_mode())

        coords0, feats0 = to_sparse_coo(data_batch_0)
        coords0, feats0 = ME.utils.sparse_collate(coords=[coords0], feats=[feats0])

        coords1, feats1 = to_sparse_coo(data_batch_1)
        coords1, feats1 = ME.utils.sparse_collate(coords=[coords1], feats=[feats1])

        for _ in range(2):
            # sparse tensors
            A = ME.SparseTensor(coordinates=coords0, features=feats0)
            B = ME.SparseTensor(
                coordinates=coords1,
                features=feats1,
                # coords_manager=A.coords_man,  No need to feed the coords_man
                # force_creation=True
            )

            C = A + B

            # When done using it for forward and backward, you must cleanup the coords man
            ME.clear_global_coordinate_manager()

    # noinspection PyPep8Naming
    def decomposition():
        coords0, feats0 = to_sparse_coo(data_batch_0)
        coords1, feats1 = to_sparse_coo(data_batch_1)
        coords, feats = ME.utils.sparse_collate(
            coords=[coords0, coords1], feats=[feats0, feats1])

        # sparse tensors
        A = ME.SparseTensor(coordinates=coords, features=feats)
        conv = ME.MinkowskiConvolution(
            in_channels=1, out_channels=2, kernel_size=3, stride=2, dimension=2)
        B = conv(A)

        # Extract features and coordinates per batch index
        list_of_coords = B.decomposed_coordinates
        list_of_feats = B.decomposed_features
        list_of_coords, list_of_feats = B.decomposed_coordinates_and_features

        # To specify a batch index
        batch_index = 1
        coords = B.coordinates_at(batch_index)
        feats = B.features_at(batch_index)

        # Empty list if given an invalid batch index
        # batch_index = 3
        # 9= print(B.coordinates_at(batch_index))

    def test_m_conv():
        ME.clear_global_coordinate_manager()
        ME.set_sparse_tensor_operation_mode(
            ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
        coords, feats = ME.utils.sparse_collate(coords=[torch.tensor([[3, 5, 7], [8, 9, 1], [2, 3, 3], [2, 3, 4]], dtype=torch.int32)],
                                                feats=[torch.tensor([[0.5, 0.4], [0.1, 0.6], [-0.9, 10], [-0.9, 8]])])
        batch_points = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1)
        m_conv = ME.MinkowskiConvolution(2, 2, 2, 2, dimension=3)
        m_conv_trans = ME.MinkowskiConvolutionTranspose(2, 6, 2, 2, dimension=3)
        m_out = m_conv(batch_points)
        m_out_trans = m_conv_trans(m_out)
        dense_m_out = m_out.dense(shape=torch.Size([1, 3, 10, 10, 10]), min_coordinate=torch.IntTensor([0, 0, 0]))
        batch_dense_points = batch_points.dense(shape=torch.Size([1, 2, 10, 10, 10]))
        print('Done')

    # sparse_tensor_initialization()
    # sparse_tensor_arithmetics()
    # operation_mode()
    # decomposition()
    test_m_conv()


if __name__ == '__main__':
    # main_debug()
    cfg = ModelConfig()
    cfg.resolution = 128
    model = PCC(cfg)
    xyz_c = [ME.utils.sparse_quantize(torch.randint(-128, 128, (100, 3))) for _ in range(16)]
    xyz_f = [torch.ones((_.shape[0], 1), dtype=torch.float32) for _ in xyz_c]
    xyz = ME.utils.sparse_collate(coords=xyz_c, feats=xyz_f)
    out = model(xyz[0])
    out['loss'].backward()
    model.eval()
    model.entropy_bottleneck.update()
    test_out = model(xyz[0])
    print('Done')
