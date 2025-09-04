# -*- coding:utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from moabb.paradigms import SSVEP
from moabb.datasets import Nakanishi2015
from moabb.datasets import Lee2019_SSVEP


class ARDataset(Dataset):
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

        # EEG input
        eeg = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0)  # (1,C,T)

        # Stimulus reference
        t = np.arange(self.T_stim) / self.sfreq
        f = label_hz
        stim = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)], axis=-1)  # (T_stim,2)
        stim = torch.tensor(stim, dtype=torch.float32)

        task = self.tasks[idx]

        assert eeg.shape[-1] == stim.shape[0], "EEG segment length and stimulus length must match (2s)."

        return eeg, stim, class_label, task



class Nakanishi2015Dataset(Dataset):
    """
    MOABB Nakanishi2015 SSVEP Dataset
    Returns per sample:
      - eeg:  (1, C, T)  torch.float32
      - stim: (T, 2)     torch.float32  [sin, cos] @ stimulus frequency
      - label: int       0..(n_classes-1)
    """
    def __init__(self, subjects=[1]):
        dataset = Nakanishi2015()
        paradigm = SSVEP()
        X, labels, _ = paradigm.get_data(dataset=dataset, subjects=subjects)

        self.epochs = X.astype(np.float32)
        le = LabelEncoder()
        self.labels = le.fit_transform(labels)
        self.freqs = le.classes_.astype(float)
        self.sfreq = 256.0

        self.N, self.C, self.T = self.epochs.shape
        self.n_classes = len(np.unique(self.labels))

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        eeg_np = self.epochs[idx]                  # (C, T)
        label = int(self.labels[idx])              # class index
        f = float(self.freqs[label])               # stimulus frequency

        eeg = torch.tensor(eeg_np, dtype=torch.float32).unsqueeze(0)  # (1, C, T)
        t = np.arange(self.T, dtype=np.float32) / self.sfreq
        stim_np = np.stack([np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t)], axis=-1)  # (T,2)
        stim = torch.tensor(stim_np, dtype=torch.float32)

        return eeg, stim, label


class Lee2019Dataset(Dataset):
    """
    MOABB Lee2019 SSVEP Dataset
    Session0 → train, Session1 → test
    Returns (EEG, Stimulus, Label).
    """
    def __init__(self, subjects=[1], train=True, rfreq=250):
        super().__init__()
        # Paradigm: SSVEP
        paradigm = SSVEP()
        dataset = Lee2019_SSVEP()

        # Load data
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)

        if train:
            session_mask = (meta['session'] == "0")
        else:
            session_mask = (meta['session'] == "1")

        X = X[session_mask]
        labels = labels[session_mask]

        # Encode labels (string → int)
        le = LabelEncoder()
        y = le.fit_transform(labels)

        # Convert to torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)       # (N, C, T)
        self.y = torch.tensor(y, dtype=torch.long)          # (N,)
        self.freqs = le.classes_                            # original frequency labels

        # Channel, time, classes
        self.C = self.X.shape[1]
        self.T = self.X.shape[2]
        self.n_classes = len(np.unique(self.y))

        # Resample + filter (as in Colab reference)
        import mne
        info = mne.create_info(ch_names=[f"ch{i}" for i in range(self.C)],
                               sfreq=1000, ch_types="eeg")          # original fs=1000 Hz
        raw = mne.EpochsArray(self.X.numpy(), info)
        raw.resample(rfreq)                                         # downsample to 250 Hz
        raw.filter(l_freq=1, h_freq=40, fir_design="firwin", verbose=False)

        self.X = torch.tensor(raw.get_data(), dtype=torch.float32)  # (N, C, T_resampled)
        self.T = self.X.shape[2]

        # Stimulus references
        self.stim_refs = []
        t = np.arange(self.T) / rfreq
        for label in self.y:
            f = float(self.freqs[label])
            ref = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)], axis=-1)
            self.stim_refs.append(ref)
        self.stim_refs = torch.tensor(np.array(self.stim_refs), dtype=torch.float32)  # (N, T, 2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        eeg = self.X[idx].unsqueeze(0)  # (1, C, T)
        stim = self.stim_refs[idx]      # (T, 2)
        label = self.y[idx]
        return eeg, stim, label