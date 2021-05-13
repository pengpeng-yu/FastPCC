import os
import time
import platform

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def select_device(logger, local_rank, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = ''
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space} CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)"  # bytes to MB
    else:
        s += 'CPU'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    if cuda and local_rank == -1:
        return torch.device('cuda:0')
    elif cuda and local_rank != -1:
        return torch.device('cuda', local_rank)
    else:
        return torch.device('cpu')


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(x, ops, n=100, device=None):
    # profile a pytorch module or list of modules. Example usage:
    #     x = torch.randn(16, 3, 640, 640)  # input
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(x, [m1, m2], n=100)  # profile speed over 100 iterations

    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    x.requires_grad = True
    print(torch.__version__, device.type, torch.cuda.get_device_properties(0) if device.type == 'cuda' else '')
    print(f"\n{'Params':>12s}{'GFLOPS':>12s}{'forward (ms)':>16s}{'backward (ms)':>16s}{'input':>24s}{'output':>24s}")
    for m in ops if isinstance(ops, list) else [ops]:
        m = m.to(device) if hasattr(m, 'to') else m  # device
        m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m  # type
        dtf, dtb, t = 0., 0., [0., 0., 0.]  # dt forward, backward
        try:
            flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPS
        except:
            flops = 0

        for _ in range(n):
            t[0] = time_synchronized()
            y = m(x)
            t[1] = time_synchronized()
            try:
                _ = y.sum().backward()
                t[2] = time_synchronized()
            except:  # no backward method
                t[2] = float('nan')
            dtf += (t[1] - t[0]) * 1000 / n  # ms per op forward
            dtb += (t[2] - t[1]) * 1000 / n  # ms per op backward

        s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
        s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
        p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0  # parameters
        print(f'{p:12}{flops:12.4g}{dtf:16.4g}{dtb:16.4g}{str(s_in):>24s}{str(s_out):>24s}')


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class MLPBlock(nn.Module):
    """
    if version == 'linear':
        input: (N, L_1, ..., L_n, C_in)
        output: (N, L_1, ..., L_n, C_out)
    elif version == 'conv':
        input: (N, C_in, L_1, ..., L_n,)
        output: (N, C_out, L_1, ..., L_n)
    """
    def __init__(self, in_channels, out_channels, activation='leaky_relu(0.2)', batchnorm='nn.bn1d', version='linear',
                 skip_connection=None):
        super(MLPBlock, self).__init__()
        assert version in ['linear', 'conv']
        assert activation is None or activation.split('(', 1)[0] in ['relu', 'leaky_relu']
        assert batchnorm in ['nn.bn1d', 'custom', None]
        assert skip_connection in ['sum', 'concat', None]

        if batchnorm == 'nn.bn1d':
            self.bn = nn.BatchNorm1d(out_channels)
        elif batchnorm == 'custom':
            assert version == 'linear'
            self.bn = BatchNorm1dChnlLast(out_channels)
        elif batchnorm is None:
            self.bn = None
        else: raise NotImplementedError

        if version == 'linear':
            self.mlp = nn.Linear(in_channels, out_channels, bias=self.bn is None)
        elif version == 'conv':
            self.mlp = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=self.bn is None)

        if activation is None:
            self.activation = None
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.startswith('leaky_relu'):
            self.activation = nn.LeakyReLU(negative_slope=float(activation.split('(', 1)[1].split(')', 1)[0]), inplace=True)
        else: raise NotImplementedError

        if self.bn is None and self.activation is None:
            print('Warning: You are using a MLPBlock without activation nor batchnorm, '
                  'which is identical to a nn.Linear(bias=True) object')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.version = version
        self.skip_connection = skip_connection

    def forward(self, x):
        ori_x = x

        if self.version == 'linear':

            ori_shape = x.shape
            if len(ori_shape) != 3:
                x = x.reshape(ori_shape[0], -1, ori_shape[-1])

            x = self.mlp(x)
            if isinstance(self.bn, nn.BatchNorm1d):
                x = x.permute(0, 2, 1)
                x = self.bn(x)
                x = x.permute(0, 2, 1)
            elif self.bn is not None:
                x = self.bn(x)

            if self.activation is not None:
                x = self.activation(x)

            if len(ori_shape) != 3:
                x = x.reshape(*ori_shape[:-1], self.out_channels)

            if self.skip_connection is None:
                pass
            elif self.skip_connection == 'sum':
                x = x + ori_x
            elif self.skip_connection == 'concat':
                x = torch.cat([ori_x, x], dim=-1)
            return x

        elif self.version == 'conv':
            ori_shape = x.shape
            if len(ori_shape) != 3:
                x = x.reshape(ori_shape[0], ori_shape[1], -1)

            x = self.mlp(x)
            if self.bn is not None: x = self.bn(x)
            if self.activation is not None: x = self.activation(x)

            if len(ori_shape) != 3:
                x = x.reshape(ori_shape[0], self.out_channels, *ori_shape[2:])

            if self.skip_connection is None:
                pass
            elif self.skip_connection == 'sum':
                x = x + ori_x
            elif self.skip_connection == 'concat':
                x = torch.cat([ori_x, x], dim=1)
            return x

    def __repr__(self):
        return f'MLPBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
               f'activation={self.activation}, batchnorm={self.bn}, version={self.version}, ' \
               f'skip_connection={self.skip_connection}'


class BatchNorm1dChnlLast(nn.Module):
    # very slow
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1dChnlLast, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            var, mean = torch.var_mean(x, dim=[0, 1], keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
            if self.track_running_stats:
                self.running_mean = self.running_mean * (1 - self.momentum) + mean * self.momentum
                self.running_var = self.running_var * (1 - self.momentum) + var * self.momentum

        else:
            if self.track_running_stats:
                x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            else:
                var, mean = torch.var_mean(x, dim=[0, 1], keepdim=True, unbiased=False)
                x = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x = self.weight * x + self.bias
        return x


if __name__ == '__main__':
    pass

