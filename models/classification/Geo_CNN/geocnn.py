# -*- coding: utf8 -*-

import torch_geometric.transforms as T
from torch_geometric.data import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel

import time
import numpy as np
from models.classification.Geo_CNN.utils import *

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Configuration

file_name = __file__.split('.')[0]
batch_size = 16
num_workers = 4
cuda = torch.cuda.is_available()
accumulation_steps = 1  # Gradient accumulation steps
device_list = [0]  # Can be multiple, for example, [0, 1] use GPU0 and GPU1
torch.cuda.set_device(device_list[0])
device = torch.device('cuda' if cuda else 'cpu')
base_lr = 0.001
epoch = 500
bacc = 0  # Used to record the best accuracy
test_ratio = 0.2  # If bacc reaches lacc, we test every test_ratio epochs
lacc = 0.92
only_test = False

# Load data

pre_transform = UnitSphere()
train_transform = T.Compose([
    ShufflePoints(),
    ChunkPoints(1000),
    ScalePoints((2./3., 3./2.)),
    MovePoints((-0.02, 0.02)),
    #Jitter(),
])
test_transform = T.Compose([
    ChunkPoints(1000),
])

train_dataset = ModelNet40_10000('data/ModelNet40_10000', True, train_transform, pre_transform)
test_dataset = ModelNet40_10000('data/ModelNet40_10000', False, test_transform, pre_transform)

DLF = DataListLoader if cuda else DataLoader
train_loader = DLF(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    num_workers=num_workers
)
test_loader = DLF(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=num_workers
)


# Define model

class MLP_md(nn.Module):
    def __init__(self, channels, norm='bn', resi=False, md=1, **kwargs):
        super(MLP_md, self).__init__()
        self.linears = nn.ModuleList([eval(f'nn.Conv{md}d')(i, o, 1) for i, o in zip(channels, channels[1:])])
        self.norms = nn.ModuleList([Norm(norm, o, md=md, **kwargs) for o in channels[1:]])
        self.actis = nn.ModuleList([nn.ReLU(inplace=True) for _ in channels[1:]])
        self.resi = resi

    def forward(self, x, feature_last=True): # [B, ..., C]
        twoD = len(x.shape) == 2
        if twoD:
            feature_last = False
            x = x.unsqueeze(-1)
        if feature_last: x = x.transpose(1, -1) # [B, C, ...]
        
        for linear, norm, acti in zip(self.linears, self.norms, self.actis):
            inp = x if self.resi else None
            x = linear(x)
            x = norm(x)
            x = acti(x)
            if inp is not None and x.shape == inp.shape:
                x = x + inp
        
        if feature_last:
            x = x.transpose(1, -1) # [B, ..., C]
            x = x.contiguous()
        if twoD: x = x.squeeze(-1)

        return x

class GeoConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, norm='bn'):
        super(GeoConv, self).__init__()
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.lins = nn.ModuleList([nn.Linear(in_channels, hidden_channels) for _ in range(6)])
        self.norm1 = Norm(norm, hidden_channels, md=1)
        self.acti1 = nn.ReLU(inplace=True)
        self.norm2 = Norm(norm, out_channels, md=1)
        self.acti2 = nn.ReLU(inplace=True)

    def forward(self, x, p, B, n, id_euc):
        # x[B*N, C] p[B*N, 3] sid/tid[B*n*k]
        sid_euc, tid_euc = id_euc
        k = int(len(sid_euc) / B / n)
        dev = x.device

        euc_i, euc_j = x[tid_euc], x[sid_euc] # [B*n*k, C]
        edge = euc_j - euc_i
        
        p_diff = p[sid_euc] - p[tid_euc] # [B*n*k, 3]
        p_dis = p_diff.norm(dim=-1, keepdim=True).clamp(min=1e-16) # [B*n*k, 1]
        p_cos = (p_diff / p_dis).cos()**2 # [B*n*k, 3]
        p_cos = p_cos.transpose(0, 1).reshape(-1, B, n, k, 1) # [3, B, n, k, 1]
        bid = (p_diff > 0).long() # [B*n*k, 3]
        bid += torch.tensor([0, 2, 4], device=dev, dtype=torch.long).view(1, 3)
        edge = torch.stack([lin(edge) for lin in self.lins]) # [bases, B*n*k, C]
        edge = torch.stack([edge[bid[:, i], range(B*n*k)] for i in range(3)]) # [3, B*n*k, C]
        edge = edge.view(3, B, n, k, -1)
        edge = edge * p_cos # [3, B, n, k, C]
        edge = edge.sum(dim=0) # [B, n, k, C]
        p_dis = p_dis.view(B, n, k, 1)
        p_r = p_dis.max(dim=2, keepdim=True)[0] * 1.1 # [B, n, 1, 1]
        p_d = (p_r - p_dis)**2 # [B, n, k, 1]
        edge = edge * p_d / p_d.sum(dim=2, keepdim=True) # [B, n, k, C]
        y = edge.sum(dim=2).transpose(1, -1) # [B, C, n]
        y = self.acti1(self.norm1(y)).transpose(1, -1) # [B, n, C]
        x = self.lin1(x[tid_euc[::k]]).view(B, n, -1) # [B, n, C]
        y = x + self.lin2(y) # [B, n, C]
        y = y.transpose(1, -1) # [B, C, n]
        y = self.acti2(self.norm2(y))
        y = y.transpose(1, -1) # [B, n, C]
        y = y.flatten(0, 1) # [B*n, C]
        
        return y

class PointPlus(nn.Module):  # PointNet++
    def __init__(self, in_channels, out_channels, norm='bn', first_layer=False):
        super(PointPlus, self).__init__()
        self.first_layer = first_layer
        self.fc1 = MLP_md([in_channels, out_channels], norm=norm, md=2)

    def forward(self, x, B, n, id_euc):
        # x[B*N, C] sid/tid[B*n*k]
        sid_euc, tid_euc = id_euc
        k = int(sid_euc.size(0) / B / n)

        if self.first_layer:
            x, norm = x[:, :3], x[:, 3:]
            x_i, x_j = x[tid_euc], x[sid_euc] # [B*n*k, C]
            norm_j = norm[sid_euc] # [B*n*k, C]
            edge = torch.cat([x_j - x_i, norm_j], dim=-1) # [B*n*k, C]
        else:
            x_i, x_j = x[sid_euc], x[tid_euc] # [B*n*k, C]
            edge = x_j - x_i
        edge = edge.view(B, n, k, -1) # [B, n, k, C]
        edge = self.fc1(edge) # [B, n, k, C]
        y = edge.max(2)[0] # [B, n, C]
        y = y.view(B*n, -1) # [B*n, C]

        return y

class Net(nn.Module):
    def __init__(self, out_channels, norm='bn'):
        super(Net, self).__init__()

        # branch1
        self.pp1 = PointPlus(6, 64, norm, first_layer=True)
        self.pp2 = PointPlus(64, 128, norm)
        self.pp3 = PointPlus(128, 384, norm)

        # branch2
        self.lin1 = MLP_md([6, 64])
        self.conv1 = GeoConv(64, 64, 128, norm=norm)
        self.lin2 = MLP_md([128, 256])
        self.conv2 = GeoConv(256, 64, 512, norm=norm)

        # master
        self.conv3 = GeoConv(896, 64, 768, norm=norm)
        self.lin3 = MLP_md([768, 2048], norm=norm)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            Norm(norm, 512, channels_per_group=128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            Norm(norm, 256, channels_per_group=128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, out_channels)
        )

    def forward(self, data):
        pos, norm, batch = data.pos, data.norm, data.batch
        dev = pos.device
        B = batch.max().item() + 1
        N = int(pos.size(0) / B)

        x = torch.cat([pos, norm], dim=-1)

        # branch1
        id_euc = knn(pos.view(B, N, -1), 16)
        x1 = self.pp1(x, B, N, id_euc)  # [B*N, C]
        x2 = self.pp2(x1, B, N, id_euc)
        x3 = self.pp3(x2, B, N, id_euc)

        # branch2
        x4 = self.lin1(x.view(B, N, -1)).view(B*N, -1)  # [B*N, C]
        id_euc = sphg(pos, 0.15, batch=batch, max_num_neighbors=16)
        x5 = self.conv1(x4, pos, B, N, id_euc)  # [B*N, C]

        x6 = self.lin2(x5.view(B, N, -1)).view(B*N, -1)
        id_euc = sphg(pos, 0.3, batch=batch, max_num_neighbors=16)
        x7 = self.conv2(x6, pos, B, N, id_euc)  # [B*N, C]

        # master
        x8 = torch.cat([x3, x7], dim=-1)  # [B*N, C]
        id_euc = sphg(pos, 0.6, batch=batch, max_num_neighbors=16)
        x9 = self.conv3(x8, pos, B, N, id_euc)
        x10 = self.lin3(x9.view(B, N, -1))
        x = x10.max(1)[0]  # [B, C]
        
        return self.fc(x)


# Train and test

model = Net(train_dataset.num_classes)
model = model.to(device)
model.load_state_dict(torch.load('weight.pth', map_location=f'cuda:{device_list[0]}'), strict=True)
if cuda: model = DataParallel(model, device_ids=device_list)
optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': base_lr}], lr=base_lr, weight_decay=1e-4)
# optimizer.load_state_dict(torch.load('geocnn_optimizer.pt', map_location=f'cuda:{device_list[0]}').state_dict())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=0.00001, last_epoch=-1)
criterion = cal_loss

def train(epoch):
    model.train()
    test_interval = int(test_ratio * len(train_loader))
    total_loss = correct = num_data = 0
    for i, data_list in enumerate(train_loader):
        since = time.time()
        y = torch.cat([data.y for data in data_list]) if cuda else data_list.y
        y = y.to(device)
        out = model(data_list)
        loss = criterion(out, y)
        loss /= accumulation_steps * len(device_list)
        pred = out.max(dim=1)[1]
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
        correct += pred.eq(y).sum().item()
        num_data += len(data_list)

        show = 100
        if (i + 1) % show == 0:
            print('{:.4f}s [{}/{}] Loss: {:.4f}, Train Accuracy: {:.4f}, LR: {:.5f}'.format(
                time.time() - since, i + 1, len(train_loader), total_loss / show,
                correct / num_data, scheduler.get_lr()[0]))
            total_loss = correct = num_data = 0

        if i % test_interval == 0 and bacc > lacc:
            test(epoch)
            model.train()

def test(epoch=0):
    model.eval()
    global bacc
    correct = 0
    for data_list in test_loader:
        with torch.no_grad():
            out = model(data_list)
        pred = out.max(dim=1)[1]
        y = torch.cat([data.y for data in data_list]) if cuda else data_list.y
        y = y.to(device)
        correct += pred.eq(y).sum().item()

    acc = correct / len(test_loader.dataset)
    if bacc < acc:
        bacc = acc
        torch.save(model.module.state_dict(), f"{file_name}_epoch_{epoch}_{bacc}.pth")
        torch.save(optimizer, f'{file_name}_optimizer.pt')

    print('[Epoch: %d BestAcc: %.4f] Acc: %.4f' % (epoch, bacc, acc))
    return acc

for epoch in range(epoch):
    if not only_test:
        train(epoch)
        scheduler.step(epoch)
    test(epoch)