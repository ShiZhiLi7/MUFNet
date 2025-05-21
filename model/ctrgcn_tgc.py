import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


LEAKY_ALPHA = 0.1


class PointWiseTCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(PointWiseTCN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class DeTGC(nn.Module):
    def __init__(self, in_channels, out_channels, eta, kernel_size=1, stride=1, padding=0, dilation=1,
                 num_scale=1, num_frame=300):
        super(DeTGC, self).__init__()

        self.ks, self.stride, self.dilation = kernel_size, stride, dilation
        self.T = num_frame
        self.num_scale = num_scale

        self.eta = eta
        ref = (self.ks + (self.ks - 1) * (self.dilation - 1) - 1) // 2
        tr = torch.linspace(-ref, ref, self.eta)
        self.tr = nn.Parameter(tr)

        self.conv_out = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(self.eta, 1, 1)),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        res = x
        N, C, T, V = x.size()
        Tout = T // self.stride
        dtype = x.dtype

        # learnable sampling locations
        t0 = torch.arange(0, T, self.stride, dtype=dtype, device=x.device)
        tr = self.tr.to(dtype)
        t0, tr = t0.view(1, 1, -1).expand(-1, self.eta, -1), tr.view(1, self.eta, 1)
        t = t0 + tr
        t = t.view(1, 1, -1, 1)

        # indexing
        tdn = t.detach().floor()
        tup = tdn + 1
        index1, index2 = torch.clamp(tdn, 0, self.T - 1).long(), torch.clamp(tup, 0, self.T - 1).long()
        index1, index2 = index1.expand(N, C, -1, V), index2.expand(N, C, -1, V)

        # sampling
        alpha = tup - t
        x1, x2 = x.gather(-2, index=index1), x.gather(-2, index=index2)
        x = x1 * alpha + x2 * (1 - alpha)
        x = x.view(N, C, self.eta, Tout, V)

        # conv
        x = self.conv_out(x).squeeze(2)
        return x


class MultiScale_TemporalModeling(nn.Module):
    def __init__(self, in_channels, out_channels, eta, kernel_size=5, stride=1, dilations=1,
                 num_scale=1, num_frame=64):
        super(MultiScale_TemporalModeling, self).__init__()

        scale_channels = out_channels // num_scale
        self.num_scale = num_scale if in_channels != 3 else 1

        self.tcn1 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            DeTGC(scale_channels,
                  scale_channels,
                  eta,
                  kernel_size=5,
                  stride=stride,
                  dilation=1,
                  num_scale=num_scale,
                  num_frame=num_frame)
        )

        self.tcn2 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            DeTGC(scale_channels,
                  scale_channels,
                  eta,
                  kernel_size=5,
                  stride=stride,
                  dilation=2,
                  num_scale=num_scale,
                  num_frame=num_frame)
        )

        self.maxpool3x1 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(scale_channels)
        )
        self.conv1x1 = PointWiseTCN(in_channels, scale_channels, stride=stride)

    def forward(self, x):
        x = torch.cat([self.tcn1(x), self.tcn2(x), self.maxpool3x1(x), self.conv1x1(x)], 1)
        return x


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9 or in_channels == 2:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)

        # Q
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))

        # R
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V

        # Z
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, eta=4, num_frame=64, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalModeling(out_channels,
                                                out_channels,
                                                eta=eta,
                                                stride=stride,
                                                num_scale=4,
                                                num_frame=num_frame)

        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 300
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        f = x.detach().clone()
        x = self.drop_out(x)
        x = self.fc(x)
        return x

class Model2d(nn.Module):
    def __init__(self, num_class=60, num_point=18, num_person=2, graph='graph.ntu_rgb_d2d.Graph', graph_args=dict(),
                 in_channels=2,
                 drop_out=0, adaptive=True):
        super(Model2d, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,18,18

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 300
        self.l1 = TCN_GCN_unit(2, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        f = x.detach().clone().cpu()
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        x = self.drop_out(x)
        x = self.fc(x)
        return x, f

class Model_120(nn.Module):
    def __init__(self, num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model_120, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        x = self.fc(x)
        return x


class Model120_2d(nn.Module):
    def __init__(self, num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d2d.Graph', graph_args=dict(),
                 in_channels=2,
                 drop_out=0, adaptive=True):
        super(Model_120, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 2,18,18

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(2, 64, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        x = self.fc(x)
        return x




class Model_10(nn.Module):
    def __init__(self, num_class=10, num_point=20, num_person=1, graph='graph.nucla.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 300
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        f = x.detach().clone()
        x = self.drop_out(x)
        x = self.fc(x)
        return x

class Model10_2d(nn.Module):
    def __init__(self, num_class=10, num_point=18, num_person=1, graph='graph.ntu_rgb_d2d.Graph', graph_args=dict(),
                 in_channels=2,
                 drop_out=0, adaptive=True):
        super(Model2d, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,18,18

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 200
        self.l1 = TCN_GCN_unit(2, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        f = x.detach().clone().cpu()
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        x = self.drop_out(x)
        x = self.fc(x)
        return x, f




class AuxiliaryNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AuxiliaryNetwork, self).__init__()
        self.fc_mean = nn.Linear(input_dim, output_dim)
        self.fc_logvar = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x is the input from the main network
        mean = self.fc_mean(x)
        std = self.fc_logvar(x)
        std = F.softplus(std)
        eps = torch.randn_like(std)
        u = mean + eps * std
        return u


class Model_u(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model_u, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 300
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        # f = x.detach().clone()
        x = self.drop_out(x)

        a = self.au(x)
        x = self.fc(x)
        x = x + a
        # return x,f
        return x

class Model120_u(nn.Module):
    def __init__(self, num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model120_u, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 300
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        # f = x.detach().clone()
        x = self.drop_out(x)

        a = self.au(x)
        x = self.fc(x)
        x = x + a
        # return x,f
        return x

class Model120_2d_u(nn.Module):
    def __init__(self, num_class=120, num_point=18, num_person=2, graph='graph.ntu_rgb_d2d.Graph', graph_args=dict(),
                 in_channels=2,
                 drop_out=0, adaptive=True):
        super(Model120_2d_u, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,18,18

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 300
        self.l1 = TCN_GCN_unit(2, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        # f = x.detach().clone()
        x = self.drop_out(x)

        a = self.au(x)
        x = self.fc(x)
        x = x + a
        # return x,f
        return x

class Model10_u(nn.Module):
    def __init__(self, num_class=10, num_point=20, num_person=1, graph='graph.nucla.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model10_u, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 300
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):


        N, T, V, C= x.size()
        M = 1
        x = x.permute(0, 2, 3, 1).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        f = x.detach().clone()
        x = self.drop_out(x)
        a = self.au(x)
        x = self.fc(x)
        x = x + a
        return x

class Model10_2d_u(nn.Module):
    def __init__(self, num_class=10, num_point=18, num_person=1, graph='graph.ntu_rgb_d2d.Graph', graph_args=dict(),
                 in_channels=2,
                 drop_out=0, adaptive=True):
        super(Model10_2d_u, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,18,18

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 200
        self.l1 = TCN_GCN_unit(2, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):

        N, T, V, C = x.size()
        M = 1
        x = x.permute(0, 2, 3, 1).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        f = x.detach().clone().cpu()
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        x = self.drop_out(x)
        a = self.au(x)
        x = self.fc(x)
        x = x + a
        return x






class AuxiliaryNetwork_mean(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AuxiliaryNetwork_mean, self).__init__()
        self.fc_mean = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mean = self.fc_mean(x)
        u = mean
        return u

class AuxiliaryNetwork_sigma(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AuxiliaryNetwork_sigma, self).__init__()
        self.fc_logvar = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        std = self.fc_logvar(x)
        std = F.softplus(std)
        eps = torch.randn_like(std)
        u = eps * std
        return u


class Model_u_r(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model_u_r, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 300
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork_mean(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        x = self.drop_out(x)

        x = self.fc(x)
        a = torch.randn_like(x)
        x = x + a
        return x
class Model_u_mean(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model_u_mean, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 300
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork_mean(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        x = self.drop_out(x)

        a = self.au(x)
        x = self.fc(x)
        x = x + a
        return x
class Model_u_mean_r(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model_u_mean_r, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 300
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork_mean(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        x = self.drop_out(x)

        a = self.au(x)
        a = a + torch.randn_like(a)
        x = self.fc(x)
        x = x + a
        return x
class Model_u_sigma(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model_u_sigma, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 300
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork_sigma(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        x = self.drop_out(x)

        a = self.au(x)
        x = self.fc(x)
        x = x + a
        return x




class Model10_u_r(nn.Module):
    def __init__(self, num_class=10, num_point=20, num_person=1, graph='graph.nucla.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model10_u_r, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 192
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork_mean(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V = x.size()
        M = 1
        x = x.permute(0, 1, 3, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        x = self.drop_out(x)

        x = self.fc(x)
        a = torch.randn_like(x)
        x = x + a
        return x
class Model10_2d_u_r(nn.Module):
    def __init__(self, num_class=10, num_point=18, num_person=1, graph='graph.ntu_rgb_d2d.Graph', graph_args=dict(),
                 in_channels=2,
                 drop_out=0, adaptive=True):
        super(Model10_2d_u_r, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 200
        self.l1 = TCN_GCN_unit(2, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):

        N, C, T, V = x.size()
        M = 1
        x = x.permute(0, 1, 3, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        x = self.drop_out(x)

        x = self.fc(x)
        a = torch.randn_like(x)
        x = x + a
        return x
class Model10_u_mean(nn.Module):
    def __init__(self, num_class=10, num_point=20, num_person=1, graph='graph.nucla.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model10_u_mean, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 192
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork_mean(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V = x.size()
        M = 1
        x = x.permute(0, 1, 3, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        x = self.drop_out(x)

        a = self.au(x)

        x = self.fc(x)
        x = x + a
        return x
class Model10_2d_u_mean(nn.Module):
    def __init__(self, num_class=10, num_point=18, num_person=1, graph='graph.ntu_rgb_d2d.Graph', graph_args=dict(),
                 in_channels=2,
                 drop_out=0, adaptive=True):
        super(Model10_2d_u_mean, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 200
        self.l1 = TCN_GCN_unit(2, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork_mean(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):

        N, C, T, V = x.size()
        M = 1
        x = x.permute(0, 1, 3, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        x = self.drop_out(x)

        a = self.au(x)

        x = self.fc(x)
        x = x + a
        return x
class Model10_u_mean_r(nn.Module):
    def __init__(self, num_class=10, num_point=20, num_person=1, graph='graph.nucla.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model10_u_mean_r, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 192
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork_mean(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V = x.size()
        M = 1
        x = x.permute(0, 1, 3, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        x = self.drop_out(x)

        a = self.au(x)
        a = a + torch.randn_like(a)
        x = self.fc(x)
        x = x + a
        return x
class Model10_2d_u_mean_r(nn.Module):
    def __init__(self, num_class=10, num_point=18, num_person=1, graph='graph.ntu_rgb_d2d.Graph', graph_args=dict(),
                 in_channels=2,
                 drop_out=0, adaptive=True):
        super(Model10_2d_u_mean_r, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 200
        self.l1 = TCN_GCN_unit(2, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork_mean(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):

        N, C, T, V = x.size()
        M = 1
        x = x.permute(0, 1, 3, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        x = self.drop_out(x)

        a = self.au(x)
        a = a + torch.randn_like(a)
        x = self.fc(x)
        x = x + a
        return x
class Model10_u_sigma(nn.Module):
    def __init__(self, num_class=10, num_point=20, num_person=1, graph='graph.nucla.Graph', graph_args=dict(),
                 in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model10_u_sigma, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 192
        self.l1 = TCN_GCN_unit(3, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork_sigma(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V = x.size()
        M = 1
        x = x.permute(0, 1, 3, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        x = self.drop_out(x)

        a = self.au(x)
        x = self.fc(x)
        x = x + a
        return x
class Model10_2d_u_sigma(nn.Module):
    def __init__(self, num_class=10, num_point=18, num_person=1, graph='graph.ntu_rgb_d2d.Graph', graph_args=dict(),
                 in_channels=2,
                 drop_out=0, adaptive=True):
        super(Model10_2d_u_sigma, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        T = 200
        self.l1 = TCN_GCN_unit(2, 64, A, stride=2, eta=4, num_frame=T // 2, residual=False, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, eta=4, num_frame=T // 2, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, eta=4, num_frame=T // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 2, num_class)
        self.au = AuxiliaryNetwork_sigma(base_channel * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):

        N, C, T, V = x.size()
        M = 1
        x = x.permute(0, 1, 3, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(1)

        x = x.mean(-1)
        x = self.drop_out(x)

        a = self.au(x)
        x = self.fc(x)
        x = x + a
        return x


if __name__ == '__main__':
    model = Model_u(graph='graph.ntu_rgb_d.Graph')

    x = torch.randn((32, 3, 300, 25, 2))
    out = model(x)