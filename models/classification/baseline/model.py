from typing import List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.points_layers import PointLayerMessage, TransitionDownWithDistFea, \
    RotationInvariantDistFea, DeepRotationInvariantDistFea, LocalFeatureAggregation as LFA
from lib.torch_utils import MLPBlock
from models.classification.baseline import Config


class Model(nn.Module):
    params_divider: Callable[[str], int] = lambda s: 0

    def __init__(self, cfg: Config):
        super(Model, self).__init__()
        self.cfg = cfg
        # self.neighbor_fea_generator = \
        #     DeepRotationInvariantDistFea(cfg.neighbor_num, cfg.anchor_points, 16, 32, retain_xyz_dists=True)
        self.neighbor_fea_generator = \
            RotationInvariantDistFea(cfg.neighbor_num, cfg.anchor_points, retain_xyz_dists=True)

        # the first layer has no features, thus its in_channels == 0 and mlp_shortcut == None
        self.layers = nn.Sequential(
            LFA(0, self.neighbor_fea_generator, 8, 8),
            LFA(8, self.neighbor_fea_generator, 8, 16),
            TransitionDownWithDistFea(self.neighbor_fea_generator,
                                      16, 16, 32, 'uniform', 0.25),

            LFA(32, self.neighbor_fea_generator, 32, 32),
            LFA(32, self.neighbor_fea_generator, 32, 64),
            TransitionDownWithDistFea(self.neighbor_fea_generator,
                                      64, 64, 128, 'uniform', 0.25),

            LFA(128, self.neighbor_fea_generator, 64, 128),
            LFA(128, self.neighbor_fea_generator, 64, 128),
            TransitionDownWithDistFea(self.neighbor_fea_generator,
                                      128, 128, 128, 'uniform', 0.25),

            LFA(128, self.neighbor_fea_generator, 128, 256),
            LFA(256, self.neighbor_fea_generator, 128, 256),
        )

        self.head = nn.Sequential(nn.Linear(self.layers[-1].out_channels, 256, bias=True),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Linear(256, cfg.classes_num, bias=True))
        self.log_pred_res('init')

    def log_pred_res(self, mode, pred=None, target=None):
        if mode == 'init':
            # 0: samples_num, 1: correct_num, 2: wrong_num, 3: correct_rate
            pred_res = torch.zeros((self.cfg.classes_num, 4), dtype=torch.float32)
            self.register_buffer('pred_res', pred_res)

        elif mode == 'reset':
            self.pred_res[...] = 0

        elif mode == 'log':
            assert not self.training
            assert pred is not None and target is not None
            self.pred_res[:, 0] += torch.bincount(target, minlength=self.cfg.classes_num)
            self.pred_res[:, 1] += torch.bincount(target[pred == target], minlength=self.cfg.classes_num)

        elif mode == 'show':
            self.pred_res[:, 2] = self.pred_res[:, 0] - self.pred_res[:, 1]
            self.pred_res[:, 3] = self.pred_res[:, 1] / self.pred_res[:, 0]
            samples_num = self.pred_res[:, 0].sum().cpu().item()
            correct_num = self.pred_res[:, 1].sum().cpu().item()
            return {'samples_num': samples_num,
                    'correct_num': correct_num,
                    'accuracy': correct_num / samples_num,
                    'mean_accuracy': self.pred_res[:, 3].mean().cpu().item(),
                    'class_info': self.pred_res.clone().cpu()}
        else:
            raise NotImplementedError

    def forward(self, x, requires_fea_in_test=False):
        xyz, target = x
        feature = self.layers(PointLayerMessage(xyz=xyz)).feature
        feature = torch.max(feature, dim=1).values
        feature = self.head(feature)

        if self.training:
            loss = F.cross_entropy(feature, target)
            return {'loss': loss,
                    'ce_loss': loss.detach().cpu().item()}

        else:
            pred = torch.argmax(feature, dim=1)
            if target is not None:
                self.log_pred_res('log', pred, target)
            if requires_fea_in_test:
                return {'pred': pred.detach().cpu(),
                        'feature': feature}
            else:
                return {'pred': pred.detach().cpu()}


def main_t():
    from scipy.spatial.transform import Rotation as R
    try:
        from thop import profile
        from thop import clever_format
        thop = True
    except ModuleNotFoundError:
        thop = False

    cfg = Config()
    device = 0
    torch.cuda.set_device(f'cuda:{device}')

    model = Model(cfg)
    xyz = torch.rand(cfg.input_points_num, 3)
    xyz = torch.stack([xyz, torch.tensor(R.random().apply(xyz.numpy()).astype(np.float32))], dim=0)
    target = torch.randint(0, 40, (2,))

    if thop is True:
        macs, params = profile(model, inputs=((xyz, target),))
        macs, params = clever_format([macs, params], "%.3f")
        print(f'macs: {macs}, params: {params}')

    for module_name, module in model.named_modules():
        if isinstance(module, TransitionDownWithDistFea):
            module.sample_method = 'uniform_batch_unaware'  # for testing difference
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[device])
    xyz = xyz.cuda()
    target = target.cuda()

    train_out = model((xyz, target))
    train_out['loss'].backward()

    model.eval()
    model.module.log_pred_res('reset')
    with torch.no_grad():
        test_out = model((xyz, target), True)

    diff = (test_out['feature'][0] - test_out['feature'][1]).abs()
    print(f'diff max: {diff.max()}, diff min: {diff.min()}, diff mean: {diff.mean()}')

    test_res = model.module.log_pred_res('show')
    print('Done')


if __name__ == '__main__':
    main_t()
