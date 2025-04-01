import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
import math


class SE(nn.Module):
    def __init__(self, in_channels, rd_ratio=4, rd_dim=None):
        super().__init__()
        rd_dim = rd_dim or in_channels // rd_ratio
        self.se = nn.Sequential(OrderedDict([
            ('gap', nn.AdaptiveAvgPool2d(1)),
            ('fc1', nn.Conv2d(in_channels, rd_dim, 1, bias=False)),
            ('bn', nn.BatchNorm2d(rd_dim)),
            ('relu', nn.ReLU(inplace=True)),
            ('fc2', nn.Conv2d(rd_dim, in_channels, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        return x * self.se(x)


class TransitionIB(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=6):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.act = nn.ReLU6(inplace=True)
        self.layers = nn.Sequential(OrderedDict([
            ('pw1', nn.Conv2d(inp, hidden_dim, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(hidden_dim)),
            ('act1', self.act),
            ('dw', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
            ('bn2', nn.BatchNorm2d(hidden_dim)),
            ('act2', self.act),
            ('pw2', nn.Conv2d(hidden_dim, oup, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(oup))
        ]))

    def forward(self, x):
        return self.layers(x)


class SameIB(nn.Module):
    def __init__(self, inp, expand_ratio=6):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.act = nn.ReLU6(inplace=True)
        self.layers = nn.Sequential(OrderedDict([
            ('pw1', nn.Conv2d(inp, hidden_dim, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(hidden_dim)),
            ('act1', self.act),
            ('dw', nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False)),
            ('bn2', nn.BatchNorm2d(hidden_dim)),
            ('act2', self.act),
            ('pw2', nn.Conv2d(hidden_dim, inp, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(inp))
        ]))

    def forward(self, x):
        return self.layers(x) + x


class ShareIB(nn.Module):  # Share PW only
    def __init__(self, inp, expand_ratio=6):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.act = nn.ReLU6(inplace=True)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False, groups=hidden_dim)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(inp)
        self.se1 = SE(inp)
        self.se3 = SE(hidden_dim, 24)

    def forward(self, x, weight1, weight3):
        identity = x
        x = self.act(self.bn1(F.conv2d(self.se1(x), weight1)))
        x = self.act(self.bn2(self.dwconv(x)))
        x = self.bn3(F.conv2d(self.se3(x), weight3))
        return x + identity

class ShareCluster(nn.Module):  # Share PW Only 
    def __init__(self, inp, oup, stride, n_kids, expand_ratio=6):
        super().__init__()
        hidden_dim = int(oup * expand_ratio)
        self.transition = TransitionIB(inp, oup, stride)
        self.blocks = nn.ModuleList([ShareIB(oup) for _ in range(n_kids)])
        self.last_block = SameIB(oup)
        self.static_filter_attn1 = nn.Parameter(torch.zeros(n_kids, hidden_dim, 1, 1, 1))
        self.static_filter_attn3 = nn.Parameter(torch.zeros(n_kids, oup, 1, 1, 1))
        self._initialize_weights()

    def forward(self, x):
        x = self.transition(x)
        
        weight1 = self.last_block.layers.pw1.weight.unsqueeze(0) * (1 + self.static_filter_attn1)
        weight3 = self.last_block.layers.pw2.weight.unsqueeze(0) * (1 + self.static_filter_attn3)
        for i, block in enumerate(self.blocks):
            x = block(x, weight1[i], weight3[i])
 
        x = self.last_block(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def get_mobilenet_v2(customize=True):
    model = models.mobilenet_v2(dropout = 0)
    if customize:
        n = 0
        model.features[4] = nn.Sequential()
        model.features[5] = nn.Sequential()
        model.features[6] = ShareCluster(24, 32, 2, 1 + n)
        model.features[7] = nn.Sequential()
        model.features[8] = nn.Sequential()
        model.features[9] = nn.Sequential()
        model.features[10] = ShareCluster(32, 64, 2, 2 + n)
        model.features[11] = nn.Sequential()
        model.features[12] = nn.Sequential()
        model.features[13] = ShareCluster(64, 96, 1, 1 + n)
        model.features[14] = nn.Sequential()
        model.features[15] = nn.Sequential()
        model.features[16] = ShareCluster(96, 160, 2, 1 + n)

    return model

def get_mobilenet_v2_1d2(customize=True):
    model = models.mobilenet_v2(width_mult=0.5, dropout = 0)
    if customize:
        n = 0
        model.features[4] = nn.Sequential()
        model.features[5] = nn.Sequential()
        model.features[6] = ShareCluster(16, 16, 2, 1 + n)
        model.features[7] = nn.Sequential()
        model.features[8] = nn.Sequential()
        model.features[9] = nn.Sequential()
        model.features[10] = ShareCluster(16, 32, 2, 2 + n)
        model.features[11] = nn.Sequential()
        model.features[12] = nn.Sequential()
        model.features[13] = ShareCluster(32, 48, 1, 1 + n)
        model.features[14] = nn.Sequential()
        model.features[15] = nn.Sequential()
        model.features[16] = ShareCluster(48, 80, 2, 1 + n)

    return model