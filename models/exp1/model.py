import torch
import torch.nn as nn
import torch.nn.functional as F
import compressai
from compressai.models.utils import update_registered_buffers

from lib import loss_function
from lib.torch_utils import MLPBlock
from lib.points_layers import TransitionDown, TransformerBlock
from models.exp1.model_config import ModelConfig


class PointEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(PointEncoder, self).__init__()
        self.top_down_blocks = nn.ModuleList()
        self.top_down_blocks.extend((
            nn.Sequential(TransformerBlock(3, 64, cfg.neighbor_num),
                          TransformerBlock(64, 128, cfg.neighbor_num),
                          TransitionDown('uniform', 0.25)),

            nn.Sequential(TransformerBlock(128, 256, cfg.neighbor_num),
                          TransitionDown('uniform', 0.25)),

            nn.Sequential(TransformerBlock(256, 512, cfg.neighbor_num),
                          TransitionDown('uniform', 0.25)),  # inter_fea[0]

            nn.Sequential(TransformerBlock(512, 1024, cfg.neighbor_num),
                          TransitionDown('uniform', 0.25)),  # inter_fea[1]

            nn.Sequential(TransformerBlock(1024, 1024, cfg.neighbor_num),
                          TransitionDown('uniform', 0.25)),  # inter_fea[2]
        ))
        self.trainsition_blocks = nn.ModuleList()
        self.trainsition_blocks.extend((nn.Sequential(TransformerBlock(512, 1024, cfg.neighbor_num),
                                                      TransitionDown(cfg.input_points_num // 16, None, 'uniform')),
                                        TransformerBlock(1024, 512, cfg.neighbor_num),
                                        TransformerBlock(1024, 1024, cfg.neighbor_num)))

        self.init_weights()

    def forward(self, x):
        xyz, fea = x
        xyz, fea = self.top_down_blocks[1](self.top_down_blocks[0]((xyz, fea, None, None)))[:2]
        inter_fea = [(xyz, fea, None, None)]
        for block in self.top_down_blocks[2:]:
            inter_fea.append(block(inter_fea[-1]))
        inter_fea.pop(0)

        inter_fea[2] = self.trainsition_blocks[2](inter_fea[2])
        inter_fea[1] = torch.cat((inter_fea[1][0], inter_fea[2][0]), dim=1), torch.cat((inter_fea[1][1], inter_fea[2][1]), dim=1), None, None

        inter_fea[1] = self.trainsition_blocks[1](inter_fea[1])
        inter_fea[0] = torch.cat((inter_fea[0][0], inter_fea[1][0]), dim=1), torch.cat((inter_fea[0][1], inter_fea[1][1]), dim=1), None, None

        inter_fea[0] = self.trainsition_blocks[0](inter_fea[0])
        return inter_fea[0][:2]

    def init_weights(self):
        torch.nn.init.uniform_(self.trainsition_blocks[0][0].fc2.weight, -10, 10)
        torch.nn.init.uniform_(self.trainsition_blocks[0][0].fc2.bias, -10, 10)
        torch.nn.init.uniform_(self.trainsition_blocks[0][0].shortout_fc.weight, -10, 10)
        torch.nn.init.uniform_(self.trainsition_blocks[0][0].shortout_fc.bias, -10, 10)


class PointCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = PointEncoder(cfg)
        self.encoded_points_num = cfg.input_points_num // 16
        self.encoded_points_dim = 1024

        self.entropy_bottleneck = compressai.entropy_models.EntropyBottleneck(self.encoded_points_dim)

        self.decoder_in_mlp = MLPBlock(1024, 1024, activation='leaky_relu(0.2)', batchnorm='nn.bn1d')
        self.decoder = nn.ModuleList()
        self.decoder.extend([TransformerBlock(1024, 512, cfg.neighbor_num, True),
                             TransformerBlock(512, 256, cfg.neighbor_num, True),
                             TransformerBlock(256, 48, cfg.neighbor_num, False)])

    def forward(self, fea):
        if self.training: ori_fea = fea
        batch_size = fea.shape[0]
        xyz = fea[..., :3]  # B, N, C only coordinate supported
        # encode
        xyz, fea = self.encoder((xyz, fea))

        if self.training:
            fea, likelihoods = self.entropy_bottleneck(fea.permute(0, 2, 1).unsqueeze(3).contiguous())
            fea = fea.squeeze(3).permute(0, 2, 1).contiguous()
            likelihoods = likelihoods.squeeze(3).permute(0, 2, 1).contiguous()
            fea = self.decoder_in_mlp(fea)
            fea_list = [(xyz, fea, None, None)]
            for block in self.decoder:
                fea_list.append(block(fea_list[-1]))
            fea_list.pop(0)
            fea_list = [fea_list[0][1].reshape(batch_size, self.cfg.input_points_num // 8, -1)[:, :, :3],
                        fea_list[1][1].reshape(batch_size, self.cfg.input_points_num // 4, -1)[:, :, :3],
                        fea_list[2][1].reshape(batch_size, self.cfg.input_points_num, self.cfg.input_points_dim)]

            bpp_loss = torch.log2(likelihoods).sum() * (-self.cfg.bpp_loss_factor / (ori_fea.shape[0] * ori_fea.shape[1]))
            reconstruct_reguler_loss = loss_function.chamfer_loss(fea_list[0], ori_fea) * 0.05 + \
                                       loss_function.chamfer_loss(fea_list[1], ori_fea) * 0.1
            reconstruct_loss = loss_function.chamfer_loss(fea_list[2], ori_fea)
            aux_loss = self.entropy_bottleneck.loss() * self.cfg.aux_loss_factor
            loss = reconstruct_reguler_loss + reconstruct_loss + bpp_loss + aux_loss
            return {'aux_loss': aux_loss.detach().cpu().item(),
                    'bpp_loss': bpp_loss.detach().cpu().item(),
                    'reconstruct_loss': reconstruct_loss.detach().cpu().item(),
                    'reconstruct_reguler_loss': reconstruct_reguler_loss.detach().cpu().item(),
                    'loss': loss}
        else:
            compressed_strings = self.entropy_bottleneck_compress(fea)
            decompressed_fea = self.entropy_bottleneck_decompress(compressed_strings)
            decompressed_fea = self.decoder_in_mlp(decompressed_fea)
            decoder_output = xyz, decompressed_fea, None, None
            for block in self.decoder:
                decoder_output = block(decoder_output)
            decoder_output = decoder_output[1]
            decoder_output = decoder_output.reshape(batch_size, self.cfg.input_points_num, self.cfg.input_points_dim)

            return {'encoder_output': fea,
                    'compressed_strings': compressed_strings,
                    'decompressed_fea': decompressed_fea,
                    'decoder_output': decoder_output}

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
        encoder_output = encoder_output.permute(0, 2, 1).unsqueeze(3).contiguous()
        return self.entropy_bottleneck.compress(encoder_output)

    def entropy_bottleneck_decompress(self, compressed_strings):
        assert not self.training
        decompressed_fea = self.entropy_bottleneck.decompress(compressed_strings, size=(self.encoded_points_num, 1))
        decompressed_fea = decompressed_fea.squeeze(3).permute(0, 2, 1)
        return decompressed_fea


def main_t():
    torch.cuda.set_device('cuda:2')
    cfg = ModelConfig()
    cfg.input_points_num = 1024
    model = PointCompressor(cfg)
    model = model.cuda()
    point_cloud = torch.rand(4, cfg.input_points_num, cfg.input_points_dim, device='cuda')
    out = model(point_cloud)  # 4, 16, 256
    out['loss'].backward()
    model.eval()
    model.entropy_bottleneck.update()
    val_out = model(point_cloud)


def point_encoder_t():
    cfg = ModelConfig()
    point_encoder = PointEncoder(cfg)
    pc = torch.rand(2, 8192, 3)
    output = point_encoder(pc)
    print('Done')


if __name__ == '__main__':
    main_t()
