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
    def __init__(self, npz_file, n_classes=None):
        """
        npz_file: path to .npz file
        T_stim: stimulus length (default = EEG segment length)
        """
        data = np.load(npz_file, allow_pickle=True)
        self.epochs = data["epochs"]  # (N,C,T)
        self.labels = data["labels"]  # (N,)
        self.freqs = data["freqs"]    # (N,)
        self.phases = data["phases"]  # (N,)
        self.tasks = data["tasks"]    # (N,)
        self.ch_names = data["ch_names"]
        self.sfreq = float(data["sfreq"])

        self.N, self.C, self.T = self.epochs.shape

        # Map frequency ↔ class index
        unique_freqs = sorted(np.unique(self.freqs))
        self.freq2class = {f: i for i, f in enumerate(unique_freqs)}
        self.class2freq = {i: f for f, i in self.freq2class.items()}
        self.n_classes = len(unique_freqs)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # EEG input
        eeg = torch.tensor(self.epochs[idx], dtype=torch.float32).unsqueeze(0)  # (1, C, T)

        # Convert Hz label → class index
        freq_val = float(self.freqs[idx])
        class_label = self.freq2class[freq_val]

        # Phase
        phase = float(self.phases[idx])

        # Task name
        task = self.tasks[idx]

        return eeg, class_label, task


class Nakanishi2015Dataset(Dataset):
    def __init__(self, subjects=[1], pick_channels="all"):
        dataset = Nakanishi2015()
        dataset.subject_list = list(range(1, 11))
        paradigm = SSVEP()
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)

        # Label encoding
        le = LabelEncoder()
        self.labels = le.fit_transform(labels)
        self.freqs = le.classes_.astype(float)

        # True channel names from the original paper
        ch_names = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

        # Build MNE object
        info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
        raw = mne.EpochsArray(X.astype(np.float32), info)

        # Pick channels if requested
        if pick_channels != "all":
            raw.pick(pick_channels)

        # Bandpass filter
        raw.filter(l_freq=6, h_freq=80, fir_design="firwin", verbose=False)

        self.sfreq = raw.info["sfreq"]

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

        eeg = torch.tensor(eeg_np, dtype=torch.float32).unsqueeze(0)  # (1, C, T)

        return eeg, label


class Lee2019Dataset(Dataset):
    def __init__(self, subjects=[1], train=True, pick_channels="all"):
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
        encoded_labels = le.fit_transform(labels)
        self.labels = torch.tensor(encoded_labels, dtype=torch.long)
        self.freqs = le.classes_.astype(float)

        # Channel mapping (ch1 ~ ch62)
        mapping_lee2019 = {
            "ch1": "Fp1", "ch2": "Fp2", "ch3": "Fp7", "ch4": "F3", "ch5": "Fz", "ch6": "F4", "ch7": "F8",
            "ch8": "FC5", "ch9": "FC1", "ch10": "FC2", "ch11": "FC6", "ch12": "T7", "ch13": "C3",
            "ch14": "Cz", "ch15": "C4", "ch16": "T8", "ch17": "TP9", "ch18": "CP5", "ch19": "CP1",
            "ch20": "CP2", "ch21": "CP6", "ch22": "TP10", "ch23": "P7", "ch24": "P3", "ch25": "Pz",
            "ch26": "P4", "ch27": "P8", "ch28": "PO9", "ch29": "O1", "ch30": "Oz", "ch31": "O2",
            "ch32": "PO10", "ch33": "FC3", "ch34": "FC4", "ch35": "C5", "ch36": "C1", "ch37": "C2",
            "ch38": "C6", "ch39": "CP3", "ch40": "CPz", "ch41": "CP4", "ch42": "P1", "ch43": "P2",
            "ch44": "POz", "ch45": "FT9", "ch46": "FTT9h", "ch47": "TPP7h", "ch48": "TP7", "ch49": "TPP9h",
            "ch50": "FT10", "ch51": "FTT10h", "ch52": "TPP8h", "ch53": "TP8", "ch54": "TPP10h", "ch55": "F9",
            "ch56": "F10", "ch57": "AF7", "ch58": "AF3", "ch59": "AF4", "ch60": "AF8", "ch61": "PO3", "ch62": "PO4"
        }

        ch_names = [mapping_lee2019.get(f"ch{i + 1}", f"ch{i + 1}") for i in range(X.shape[1])]

        info = mne.create_info(ch_names=ch_names, sfreq=1000.0, ch_types="eeg")
        raw = mne.EpochsArray(X.astype(np.float32), info)

        if pick_channels != "all":
            raw.pick(pick_channels)

        # Downsample 1000 → 250 Hz
        raw.resample(250.0, npad="auto")

        # Band-pass filter (1–40 Hz)
        raw.filter(l_freq=1, h_freq=40, fir_design="firwin", verbose=False)

        # Average reference
        raw.set_eeg_reference('average', projection=False)

        self.epochs = raw.get_data().astype(np.float32)  # (N,C,T)
        self.N, self.C, self.T = self.epochs.shape
        self.n_classes = len(np.unique(self.labels))
        self.ch_names = raw.info["ch_names"]
        self.sfreq = raw.info["sfreq"]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        eeg = torch.tensor(self.epochs[idx], dtype=torch.float32).unsqueeze(0)  # (1,C,T)
        label = int(self.labels[idx])
        return eeg, label


class BETADataset(Dataset):
    def __init__(self, npz_file, pick_channels="all"):
        data = np.load(npz_file, allow_pickle=True)

        self.epochs = data["epochs"].astype(np.float32)   # (N, C, T)
        self.labels = np.array(data["labels"], dtype=int)
        self.freqs = np.array(data["freqs"], dtype=float)
        self.phases = np.array(data["phases"], dtype=float)
        self.sfreq = float(data["sfreq"])
        self.ch_names = [str(ch) for ch in data["ch_names"]]

        # pick_channels
        if isinstance(pick_channels, str):
            if pick_channels.lower() == "all":
                pick_channels = "all"
            else:
                pick_channels = [ch.strip() for ch in pick_channels.split(",")]
        elif isinstance(pick_channels, (list, tuple, np.ndarray)):
            pick_channels = [str(ch).strip() for ch in pick_channels]

        if pick_channels != "all":
            name_map = {c: i for i, c in enumerate(self.ch_names)}
            picks = [name_map[ch] for ch in pick_channels if ch in name_map]

            if len(picks) == 0:
                raise ValueError(
                    f"[ERROR] No valid channels found in pick_channels={pick_channels}\n"
                    f"[INFO] Available channels={self.ch_names}"
                )

            self.epochs = self.epochs[:, picks, :]
            self.ch_names = [self.ch_names[i] for i in picks]

        self.N, self.C, self.T = self.epochs.shape
        self.n_classes = len(np.unique(self.labels))

        print(f"[INFO] Loaded {npz_file}, shape={self.epochs.shape}, classes={self.n_classes}, channels={self.ch_names}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sig = self.epochs[idx]  # (C, T)
        lbl = int(self.labels[idx])
        eeg = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)  # (1, C, T)
        return eeg, lbl