import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Small residual block used by the original GAP heads and BVP decoder."""

    def __init__(self, inplanes, out_planes, stride=2, downsample=1, res=0, islast=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        self.downsample = downsample
        self.res = res
        self.islast = islast
        if downsample == 1:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.res == 1:
            if self.downsample == 1:
                x = self.down(x)
            out = out + x
        return out if self.islast else F.relu(out, inplace=True)
