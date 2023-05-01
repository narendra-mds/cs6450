import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_planes, r):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // r),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // r, in_planes),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, r=16):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.se = SEBlock(out_planes, r)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class RegNet(nn.Module):
    def __init__(self, block_sizes, width_factor=1, num_classes=100):
        super(RegNet, self).__init__()
        self.in_planes = int(32*width_factor)
        self.layer1 = self._make_layer(16*width_factor, block_sizes[0], stride=1)
        self.layer2 = self._make_layer(32*width_factor, block_sizes[1], stride=2)
        self.layer3 = self._make_layer(64*width_factor, block_sizes[2], stride=2)
        self.layer4 = self._make_layer(128*width_factor, block_sizes[3], stride=2)
        self.linear = nn.Linear(128*width_factor, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def RegNetX(num_classes=10, **kwargs):
    block_sizes = [5, 5, 5, 5]
    return RegNet(block_sizes, num_classes)
