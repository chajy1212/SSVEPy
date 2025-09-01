# -*- coding:utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset


class SSVEPDataset(Dataset):
    def __init__(self, npz_file, n_classes=None, T_stim=None):
        """
        Args:
            npz_file: path to .npz file
            T_stim: stimulus length (default = EEG segment length)
        """
        data = np.load(npz_file, allow_pickle=True)
        self.epochs = data["epochs"]   # (N,C,T)
        self.labels = data["labels"]   # (N,)
        self.tasks  = data["tasks"]    # (N,)
        self.ch_names = data["ch_names"]
        self.sfreq = float(data["sfreq"])

        self.N, self.C, self.T = self.epochs.shape
        self.T_stim = self.T if T_stim is None else T_stim

        # Hz → class index mapping
        unique_freqs = sorted(np.unique(self.labels))
        self.freq2class = {f: i for i, f in enumerate(unique_freqs)}
        self.class2freq = {i: f for f, i in self.freq2class.items()}
        self.n_classes = len(unique_freqs)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        eeg = self.epochs[idx]  # (C,T)
        label_hz = int(self.labels[idx])  # stimulus frequency
        class_label = self.freq2class[label_hz]

        # --- EEG input ---
        eeg = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0)  # (1,C,T)

        # --- Stimulus reference (sin, cos) ---
        t = np.arange(self.T_stim) / self.sfreq
        f = label_hz
        stim = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)], axis=-1)  # (T_stim,2)
        stim = torch.tensor(stim, dtype=torch.float32)

        assert eeg.shape[-1] == stim.shape[0], "EEG segment length and stimulus length must match (2s)."

        return eeg, stim, class_label