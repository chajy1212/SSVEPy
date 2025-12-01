# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DTN(nn.Module):
    """
    Dynamic Template Network
     - Extracts conv features from EEG
     - Global average pooling for dimension reduction
     - Returns latent representation only
    """
    def __init__(self, n_bands, n_features, n_channels, n_samples, n_classes,
                 band_kernel=9, pooling_kernel=2,
                 dropout=0.5, momentum=None, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # One-hot table for template update
        self.register_buffer('encode_table', torch.eye(n_classes, dtype=torch.long))

        # Feature extractor
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('band_layer', nn.Conv2d(1, n_bands, (1, band_kernel),
                                     padding=(0, band_kernel // 2), bias=False)),
            ('spatial_layer', nn.Conv2d(n_bands, n_features, (n_channels, 1),
                                        bias=False)),
            ('temporal_layer1', nn.Conv2d(n_features, n_features, (1, pooling_kernel),
                                          stride=(1, pooling_kernel), bias=False)),
            ('bn', nn.BatchNorm2d(n_features)),
            ('tanh', nn.Tanh()),
            ('temporal_layer2', nn.Conv2d(n_features, n_features, (1, band_kernel),
                                          padding=(0, band_kernel // 2), bias=False)),
        ]))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_drop = nn.Dropout(dropout)
        self.instance_norm = nn.InstanceNorm2d(1)

        # Initialize running templates
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)
            feat = self.feature_extractor(x)
            self._register_templates(n_classes, *feat.shape[1:])


    def _register_templates(self, n_classes, *args):
        """ Initialize class-specific templates """
        self.register_buffer('running_template', torch.zeros(n_classes, *args))
        nn.init.xavier_uniform_(self.running_template, gain=1)


    def _update_templates(self, x, y):
        """ Update templates with exponential moving average (EMA) """
        with torch.no_grad():
            self.num_batches_tracked += 1
            factor = (1.0 / float(self.num_batches_tracked)) if self.momentum is None else self.momentum

            mask = F.one_hot(y, num_classes=self.running_template.shape[0]).float()  # (B, n_classes)
            features = self.feature_extractor(x)                                     # (B, F, C’, T’)

            mask_data = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * features.unsqueeze(1)
            new_template = mask_data.sum(0) / (mask.sum(0).view(-1, 1, 1, 1) + self.eps)

            self.running_template = (1 - factor) * self.running_template + factor * new_template


    def forward(self, x, y=None, return_feat=True):
        """
        x: (B, 1, C, T)
        return_feat=True → (logits=None, feat)
        """
        x = self.instance_norm(x)
        feat = self.feature_extractor(x)        # (B, F, C’, T’)
        feat = self.global_pool(feat)           # (B, F, 1, 1)
        feat = feat.view(feat.size(0), -1)      # (B, F)
        feat = self.fc_drop(feat)

        # Update templates if training
        if self.training and y is not None:
            self._update_templates(x, y)

        if return_feat:
            return None, feat

        return None, feat