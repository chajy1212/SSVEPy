# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class FTN(nn.Module):
    """
    Fixed Template Network
     - Compares input EEG segments with fixed templates in a learned feature space.
     - Returns classification logits + latent representation
    """
    def __init__(self, n_channels, n_samples, n_classes, hidden_dim=64):
        super().__init__()
        self.n_classes = n_classes

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, (1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, hidden_dim, (n_channels, 1), bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Tanh()
        )

        # Fixed templates (frozen, not updated during training)
        self.register_buffer("templates", torch.randn(n_classes, 1, n_channels, n_samples))

        # Classification head
        self.fc = nn.Linear(hidden_dim, n_classes)


    def forward(self, x, return_feat=False):
        """
        Args:
            x (Tensor): EEG input of shape (B, 1, C, T)
            return_feat (bool):
                - True → return (logits, representation)
                - False → return logits only

        Returns:
            logits (Tensor): (B, n_classes) classification output
            feat_x (Tensor): (B, hidden_dim) latent representation
        """
        # EEG feature extraction
        feat_x = self.feature_extractor(x)          # (B, hidden_dim, 1, T')
        feat_x = feat_x.mean(dim=-1).squeeze(-1)    # (B, hidden_dim)

        # Template feature extraction
        t = self.feature_extractor(self.templates)  # (C, hidden_dim, 1, T')
        t = t.mean(dim=-1).squeeze(-1)              # (C, hidden_dim)

        # Cosine similarity between EEG features and templates
        sim = F.linear(F.normalize(feat_x, dim=1),
                       F.normalize(t, dim=1))       # (B, C)

        logits = self.fc(sim)

        if return_feat:
            return logits, feat_x
        return logits, feat_x


class DTN(nn.Module):
    """
    Dynamic Template Network
     - Learns class-specific templates via moving-average update.
     - By default returns only classification logits.
     - Optionally, can also return latent representation (flat_feat).

    Args:
        n_bands (int): number of temporal filter bands
        n_features (int): number of spatial filters
        n_channels (int): number of EEG channels
        n_samples (int): number of time samples
        n_classes (int): number of classes
        band_kernel (int): kernel size for temporal conv
        pooling_kernel (int): pooling kernel size
        dropout (float): dropout probability
        momentum (float): template update momentum (None = 1/t)
        eps (float): numerical stability constant
    """
    def __init__(self,
                 n_bands, n_features,
                 n_channels, n_samples, n_classes,
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

        self.flatten = nn.Flatten()
        self.fc_drop = nn.Dropout(dropout)

        # Initialize running templates
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)    # (B,1,C,T)
            feat = self.feature_extractor(x)
            self._register_templates(n_classes, *feat.shape[1:])

        # Classification head
        self.fc_layer = nn.Linear(feat.numel() // feat.shape[0], n_classes)
        self.instance_norm = nn.InstanceNorm2d(1)


    def _register_templates(self, n_classes, *args):
        """ Initialize class-specific templates with Xavier uniform distribution """
        self.register_buffer('running_template', torch.zeros(n_classes, *args))
        nn.init.xavier_uniform_(self.running_template, gain=1)


    def _update_templates(self, x, y):
        """ Update templates online using exponential moving average """
        with torch.no_grad():
            self.num_batches_tracked += 1
            factor = (1.0 / float(self.num_batches_tracked)) if self.momentum is None else self.momentum

            mask = F.one_hot(y, num_classes=self.running_template.shape[0]).float()     # (B, n_classes)
            features = self.feature_extractor(x)                                        # (B, F, C’, T’)

            # Apply class mask → aggregate per class
            mask_data = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * features.unsqueeze(1)
            new_template = mask_data.sum(0) / (mask.sum(0).view(-1, 1, 1, 1) + self.eps)

            # Exponential moving average update
            self.running_template = (1 - factor) * self.running_template + factor * new_template


    def forward(self, x, y=None, return_feat=True):
        """
        Args:
            x (Tensor): EEG input (B,1,C,T)
            y (Tensor): class labels (B,) → optional (needed for template update in training mode)
            return_feat (bool):
                - True → return (logits, representation)
                - False → return logits only

        Returns:
            logits (Tensor): classification scores (B, n_classes)
            flat_feat (Tensor): latent representation (B,D) if return_feat=True
        """
        x = self.instance_norm(x)                 # (B,1,C,T)
        feat = self.feature_extractor(x)          # (B,F,C’,T’)
        flat_feat = feat.view(feat.size(0), -1)   # (B,D)

        out = self.fc_drop(flat_feat)
        logits = self.fc_layer(out)               # (B,n_classes)

        # Update templates if training
        if self.training and y is not None:
            self._update_templates(x, y)

        if return_feat:
            return logits, flat_feat

        return logits