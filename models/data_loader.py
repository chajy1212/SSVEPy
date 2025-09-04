# -*- coding:utf-8 -*-
import mne
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

        # Map frequency ↔ class index
        unique_freqs = sorted(np.unique(self.labels))
        self.freq2class = {f: i for i, f in enumerate(unique_freqs)}
        self.class2freq = {i: f for f, i in self.freq2class.items()}
        self.n_classes = len(unique_freqs)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        eeg = self.epochs[idx]  # (C,T)
        label_hz = int(self.labels[idx])
        class_label = self.freq2class[label_hz]

        # EEG input
        eeg = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0)  # (1,C,T)

        # Generate sinusoidal references
        t = np.arange(self.T_stim) / self.sfreq
        f = label_hz
        stim = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)], axis=-1)  # (T_stim,2)
        stim = torch.tensor(stim, dtype=torch.float32)

        task = self.tasks[idx]

        assert eeg.shape[-1] == stim.shape[0], "EEG segment length and stimulus length must match (2s)."
        return eeg, stim, class_label, task


class Nakanishi2015Dataset(Dataset):
    def __init__(self, subjects=[1], pick_channels="all"):
        dataset = Nakanishi2015()
        paradigm = SSVEP()
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)

        # Label encoding
        le = LabelEncoder()
        self.labels = le.fit_transform(labels)
        self.freqs = le.classes_.astype(float)
        self.sfreq = 256.0

        # True channel names from the original paper
        ch_names = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

        # Build MNE object
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types="eeg")
        raw = mne.EpochsArray(X.astype(np.float32), info)

        # Pick channels if requested
        if pick_channels != "all":
            raw.pick(pick_channels)

        # Bandpass filter
        raw.filter(l_freq=1, h_freq=40, fir_design="firwin", verbose=False)

        # Final data
        self.epochs = raw.get_data().astype(np.float32)  # (N, C, T)
        self.N, self.C, self.T = self.epochs.shape
        self.n_classes = len(np.unique(self.labels))
        self.ch_names = raw.info["ch_names"]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        eeg_np = self.epochs[idx]  # (C, T)
        label = int(self.labels[idx])
        f = float(self.freqs[label])

        eeg = torch.tensor(eeg_np, dtype=torch.float32).unsqueeze(0)  # (1, C, T)

        # Reference signals
        t = np.arange(self.T, dtype=np.float32) / self.sfreq
        stim_np = np.stack([np.sin(2 * np.pi * f * t),
                            np.cos(2 * np.pi * f * t)], axis=-1)  # (T, 2)
        stim = torch.tensor(stim_np, dtype=torch.float32)

        return eeg, stim, label


class Lee2019Dataset(Dataset):
    def __init__(self, subjects=[1], train=True, rfreq=250, pick_channels="all"):
        super().__init__()
        paradigm = SSVEP()
        dataset = Lee2019_SSVEP()

        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)

        if train:
            session_mask = (meta['session'] == "0")
        else:
            session_mask = (meta['session'] == "1")

        X = X[session_mask]
        labels = labels[session_mask]

        le = LabelEncoder()
        y = le.fit_transform(labels)

        mapping_lee2019 = {
            "ch0": "Fp1", "ch1": "Fp2",
            "ch2": "AF3", "ch3": "AF4",
            "ch4": "F7", "ch5": "F3", "ch6": "Fz", "ch7": "F4", "ch8": "F8",
            "ch9": "FC5", "ch10": "FC1", "ch11": "FC2", "ch12": "FC6",
            "ch13": "T7", "ch14": "C3", "ch15": "Cz", "ch16": "C4", "ch17": "T8",
            "ch18": "CP5", "ch19": "CP1", "ch20": "CP2", "ch21": "CP6",
            "ch22": "P7", "ch23": "P3", "ch24": "Pz", "ch25": "P4", "ch26": "P8",
            "ch27": "PO7", "ch28": "PO3", "ch29": "POz", "ch30": "PO4", "ch31": "PO8",
            "ch32": "O1", "ch33": "Oz", "ch34": "O2",
            # Extra posterior/temporal sites (from 62 montage)
            "ch35": "F9", "ch36": "F10",
            "ch37": "FT9", "ch38": "FT10",
            "ch39": "TP9", "ch40": "TP10",
            "ch41": "PO9", "ch42": "PO10",
            # Additional electrodes (fill with standard 10-10 names)
            "ch43": "AF7", "ch44": "AF8",
            "ch45": "FC3", "ch46": "FC4",
            "ch47": "C1", "ch48": "C2", "ch49": "C5", "ch50": "C6",
            "ch51": "CP3", "ch52": "CP4",
            "ch53": "P1", "ch54": "P2", "ch55": "P5", "ch56": "P6",
            "ch57": "PO1", "ch58": "PO2",
            "ch59": "AFz", "ch60": "CPz", "ch61": "Pz"  # duplicates tolerated
        }

        ch_names = [mapping_lee2019.get(f"ch{i}", f"ch{i}") for i in range(X.shape[1])]
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types="eeg")
        raw = mne.EpochsArray(X.astype(np.float32), info)

        if pick_channels != "all":
            raw.pick(pick_channels)

        raw.resample(rfreq)
        raw.filter(l_freq=1, h_freq=40, fir_design="firwin", verbose=False)

        self.X = torch.tensor(raw.get_data(), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.freqs = le.classes_

        self.N, self.C, self.T = self.X.shape
        self.n_classes = len(np.unique(self.y))
        self.ch_names = raw.info["ch_names"]

        # Reference signals
        self.stim_refs = []
        t = np.arange(self.T) / rfreq
        for label in self.y:
            f = float(self.freqs[label])
            ref = np.stack([np.sin(2 * np.pi * f * t),
                            np.cos(2 * np.pi * f * t)], axis=-1)
            self.stim_refs.append(ref)
        self.stim_refs = torch.tensor(np.array(self.stim_refs), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        eeg = self.X[idx].unsqueeze(0)  # (1, C, T)
        stim = self.stim_refs[idx]
        label = self.y[idx]
        return eeg, stim, label