import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class Discriminator(nn.Module):

    def __init__(self, conv_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))

        curr_dim = conv_dim
        for i in range (1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.linear = nn.Linear(curr_dim, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.main(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return self.linear(x)


class Generator(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=5):
        super(Generator, self).__init__()

        head = []
        body = []
        tail = []

        head.append(nn.Conv2d(3, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
        head.append(nn.InstanceNorm2d(conv_dim, affine=True))
        head.append(nn.ReLU(inplace=True))

        head.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
        head.append(nn.InstanceNorm2d(conv_dim, affine=True))
        head.append(nn.ReLU(inplace=True))

        # Bottleneck layers.
        for i in range(repeat_num):
            body.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))

        #tail.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        tail.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
        tail.append(nn.InstanceNorm2d(conv_dim, affine=True))
        tail.append(nn.ReLU(inplace=True))
        tail.append(nn.Conv2d(conv_dim, 3, kernel_size=3, stride=1, padding=1, bias=True))

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x_ = self.head(x)
        x_ = self.body(x_) + x_
        x_ = -self.tail(x_) + x
        return x_
