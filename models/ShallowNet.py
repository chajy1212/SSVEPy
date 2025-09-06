import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowNet(nn.Module):
    """
    Shallow ConvNet Encoder (Schirrmeister et al., 2017)
    Input : (B, 1, C, T)
    Output: (B, D_eeg) flattened EEG representation
    """
    def __init__(self, chans, samples, dropout=0.5):
        super().__init__()
        self.chans = chans
        self.samples = samples

        # Temporal conv
        self.conv_time = nn.Conv2d(1, 40, (1, 25), bias=False)

        # Spatial conv
        self.conv_spat = nn.Conv2d(40, 40, (chans, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)

        # Pooling + log
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            feat = self.forward_features(dummy)
            self.out_dim = feat.shape[1]


    def forward_features(self, x):
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = x ** 2                               # square nonlinearity
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # safe log
        x = self.drop(x)
        x = x.view(x.size(0), -1)                # flatten
        return x

    def forward(self, x):
        return self.forward_features(x)