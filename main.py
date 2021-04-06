import open3d
import torch
import torch.nn as nn
import math
from torchvision import transforms
import numpy as np
import cv2
from compressai.zoo import bmshj2018_factorized


def open3d_draw(obj):
    if isinstance(obj, str):
        pc = np.loadtxt(obj, dtype=np.float32, delimiter=',')[:, :3]
    elif isinstance(obj, torch.Tensor):
        pc = obj.detach().cpu().numpy()[..., :3]
    elif isinstance(obj, np.ndarray):
        pc = obj[:, :3]
    else:
        raise NotImplementedError
    if len(pc.shape) == 2:
        pc = [pc]
    elif len(pc.shape) == 3:
        pc = [*pc]
    else:
        raise NotImplementedError
    pc = [open3d.geometry.PointCloud(open3d.utility.Vector3dVector(sub_pc[:, :3])) for sub_pc in pc]
    open3d.visualization.draw_geometries(pc)
    print('Done')


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


def compressai_inference_t():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = bmshj2018_factorized(quality=2, pretrained=True).eval().to(device)
    print(f'Parameters: {sum(p.numel() for p in net.parameters())}')
    img = np.ascontiguousarray(cv2.imread('dataset/000000000139.jpg')[:, :, ::-1])
    img = img[:img.shape[0] // 16 * 16, :img.shape[1] // 16 * 16, :]
    x = torch.from_numpy(img).to(device)
    x = x.permute(2, 0, 1)[None, ...]
    x = x.type(torch.float32) * (1 / 256)

    with torch.no_grad():
        out_net = net.forward(x)
    criterion = RateDistortionLoss()
    loss_dict = criterion(out_net, x)
    out_net['x_hat'].clamp_(0, 1)
    print(out_net.keys())

    rec_net = (out_net['x_hat'].squeeze().cpu().permute(1, 2, 0) * 256).numpy().astype(np.uint8)
    tensor_diff = torch.mean((out_net['x_hat'] - x).abs(), axis=1).squeeze().cpu()
    im_diff = rec_net - img
    cv2.imwrite('dataset/Reconstructed.jpg', rec_net[:, :, ::-1])

    net.update()
    strings = net.entropy_bottleneck.compress(out_net['y_hat'])
    y_hat = net.entropy_bottleneck.decompress(strings, out_net['y'].shape[2:])
    rans_diff = out_net['y_hat'] - y_hat
    print('Done')


if __name__ == '__main__':
    open3d_draw('dataset/airplane_0001.txt')