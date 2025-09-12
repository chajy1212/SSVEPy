# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, chans, samples, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
        super().__init__()
        self.chans = chans
        self.samples = samples

        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropoutRate)

        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropoutRate)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)   # (B,1,C,T)
            out = self.forward_features(dummy)
            self.out_dim = out.view(1, -1).shape[1]

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwiseConv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.drop1(x)
        x = self.separableConv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)   # flatten
        return x