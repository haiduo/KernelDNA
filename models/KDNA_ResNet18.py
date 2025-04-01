import torch 
from torch import nn 
import torch.nn.functional as F
from typing import Optional
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


class NewLayer(nn.Module):  # 6Convs: FS-SF-SF, 223, spatial/filter/channel mod
    def __init__(self, inplanes: int, planes: int, stride: int = 1, rd_ratio: int = 4, 
                 downsample: Optional[nn.Module] = None, 
                 n_kids: int = 3
                 ):
        super().__init__()
        print(f'Number of Child Convs: {n_kids}.')
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Identity() if downsample is None else downsample
        
        self.static_spatial_attn = nn.Parameter(torch.zeros(n_kids, 1, 1, 3, 3), requires_grad=True)
        self.static_filter_attn = nn.Parameter(torch.zeros(n_kids, planes, 1, 1, 1), requires_grad=True)
        self.se = nn.ModuleList([SE(planes, rd_ratio) for _ in range(n_kids)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for _ in range(n_kids)])
        
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: FS
        identity = self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        
        weight1 = self.conv2.weight * (1 + self.static_spatial_attn[0]) * (1 + self.static_filter_attn[0])
        weight2 = self.conv2.weight * (1 + self.static_spatial_attn[1]) * (1 + self.static_filter_attn[1])
        weight3 = self.conv3.weight * (1 + self.static_spatial_attn[2]) * (1 + self.static_filter_attn[2])
        
        
        x = F.conv2d(input=self.se[0](x), weight=weight1, bias=None, stride=1, padding=1)
        x = self.relu(self.bns[0](x) + identity)
        # Block 2: SF
        identity = x  
        x = F.conv2d(input=self.se[1](x), weight=weight2, bias=None, stride=1, padding=1)
        x = self.relu(self.bns[1](x))
        
        x = self.relu(self.bn2(self.conv2(x)) + identity)
        # Block 3: SF
        identity = x
        x = F.conv2d(input=self.se[2](x), weight=weight3, bias=None, stride=1, padding=1)
        x = self.relu(self.bns[2](x))
        
        x = self.relu(self.bn3(self.conv3(x)) + identity)
        
        return x   


def get_resnet18(customize=True):
    model = models.__dict__['resnet18']()
    if customize:
        layers = ['layer3', 'layer4']
        for name, module in model.named_children():
            if name in layers:
                inplanes = module[0].conv1.in_channels
                planes = module[0].conv1.out_channels
                downsample = module[0].downsample
                stride = module[0].conv1.stride
                new_layer = NewLayer(inplanes, planes, stride=stride, downsample=downsample)
                setattr(model, name, new_layer)
                print(f'{name} has been replaced successfully!')
            
    return model
    


