import torch 
from torch import nn 
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict 


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
    

class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        assert kernel_size in [1, 3]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))

        return x
    

class Block(nn.Module):  # ResNet50, PW-3x3Dense-PW  4C->C, C->C, C->4C
    def __init__(self, dim, inp=None, oup=None, stride=1, downsample=False):
        super().__init__()
        inp = inp or dim * 4
        oup = oup or dim * 4
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(inp, oup, 1, stride, bias=False)),
            ('bn', nn.BatchNorm2d(oup))
        ])) if downsample else nn.Identity()
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(inp, dim, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(dim)),
            ('relu', self.relu),
            ('conv2', nn.Conv2d(dim, dim, 3, stride, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(dim)),
            ('relu2', self.relu),
            ('conv3', nn.Conv2d(dim, oup, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(oup))
        ]))
        
    def forward(self, x):
        return self.relu(self.layers(x) + self.downsample(x))


class Block_Share3x3(nn.Module):  # ResNet50, PW-3x3Dense-PW  4C->C, C->C, C->4C
    def __init__(self, dim, inp=None, oup=None, stride=1, downsample=False, rd_ratio=4):
        super().__init__()
        inp = inp or dim * 4
        oup = oup or dim * 4
        self.stride = stride 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(inp, oup, 1, stride, bias=False)),
            ('bn', nn.BatchNorm2d(oup))
        ])) if downsample else nn.Identity()
        self.half1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(inp, dim, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(dim)),
            ('relu', self.relu),
            ('se', SE(dim, rd_ratio))
        ]))
        self.half2 = nn.Sequential(OrderedDict([
            ('bn2', nn.BatchNorm2d(dim)),
            ('relu2', self.relu),
            ('conv3', nn.Conv2d(dim, oup, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(oup))
        ]))
        
    def forward(self, x, weight):
        identity = self.downsample(x)
        x = self.half2(F.conv2d(self.half1(x), weight, stride=self.stride, padding=1))
        return self.relu(x + identity)
    

class NewLayer3(nn.Module):  # Only share 3x3 Dense Conv, F-n(SF), channel/filter/spatial mod
    def __init__(self, dim, n=3):
        super().__init__()
        self.n = n 
        self.block0 = Block(dim, inp=2 * dim, stride=2, downsample=True)
        self.parents = nn.ModuleList([Block(dim) for _ in range(n)])
        self.kids = nn.ModuleList([Block_Share3x3(dim) for _ in range(n)])
        self.static_filter_attn = nn.Parameter(torch.zeros(n, dim, 1, 1, 1))
        self.static_spatial_attn = nn.Parameter(torch.zeros(n, 1, 1, 3, 3))
    
    def forward(self, x):
        x = self.block0(x)
        weights = [self.parents[i].layers.conv2.weight * (1 + self.static_filter_attn[i]) * (1 + self.static_spatial_attn[i]) for i in range(self.n)]
        for i in range(self.n):
            x = self.kids[i](x, weights[i])
            x = self.parents[i](x)
        return x 

class NewLayer4(nn.Module):  # share 3x3 Dense Conv, PW-3x3Dense-PW, F-nS-F, channel/filter/spatial mod
    def __init__(self, dim, n_kids):
        super().__init__()
        self.block0 = Block(dim, inp=2 * dim, stride=2, downsample=True)
        self.blocks = nn.ModuleList([Block_Share3x3(dim) for _ in range(n_kids)])
        self.last_block = Block(dim)
        self.static_filter_attn2 = nn.Parameter(torch.zeros(n_kids, dim, 1, 1, 1))
        self.static_spatial_attn = nn.Parameter(torch.zeros(n_kids, 1, 1, 3, 3))
    
    def forward(self, x):
        x = self.block0(x)
        # weights for middle 3x3 dense convolution
        weights2 = self.last_block.layers.conv2.weight.unsqueeze(0) * (1 + self.static_filter_attn2) * (1 + self.static_spatial_attn)  # [n, C_out, C_in, k, k]
        for i, m in enumerate(self.blocks):
            x = m(x, weights2[i])
        x = self.last_block(x)
        
        return x 
 
def get_resnet50(customize=True):
    model = models.__dict__['resnet50']()
    if customize:
        setattr(model, 'layer3', NewLayer3(dim=256, n=3))
        setattr(model, 'layer4', NewLayer4(512, 1))
        print("layer 3 & 4 have been replaced successfully!")
    
    return model  
