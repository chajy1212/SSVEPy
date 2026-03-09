# -*- coding:utf-8 -*-
import mne
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from moabb.paradigms import SSVEP
from moabb.datasets import Nakanishi2015
from moabb.datasets import Lee2019_SSVEP


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
        raw.filter(l_freq=6, h_freq=80, fir_design="firwin", verbose=False)

        # Pick channels if requested
        if pick_channels != "all":
            raw.pick(pick_channels)

        self.sfreq = raw.info["sfreq"]
        self.epochs = raw.get_data().astype(np.float32)  # (N, C, T)
        self.N, self.C, self.T = self.epochs.shape
        self.n_classes = len(np.unique(self.labels))
        self.ch_names = raw.info["ch_names"]

        print(f"  -> [Nakanishi] Loaded Subjects {subjects} | Shape: {self.epochs.shape}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        eeg_np = self.epochs[idx]                                       # (C, T)
        label = int(self.labels[idx])
        eeg = torch.tensor(eeg_np, dtype=torch.float32).unsqueeze(0)    # (1, C, T)
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
        raw.filter(l_freq=3, h_freq=60, fir_design="firwin", verbose=False)

        if pick_channels != "all":
            raw.pick(pick_channels)

        # Downsample 1000 → 250 Hz
        raw.resample(250.0, npad="auto")

        # Average reference
        raw.set_eeg_reference('average', projection=False)

        self.epochs = raw.get_data().astype(np.float32)  # (N, C, T)
        self.N, self.C, self.T = self.epochs.shape
        self.n_classes = len(np.unique(self.labels))
        self.ch_names = raw.info["ch_names"]
        self.sfreq = raw.info["sfreq"]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        eeg = torch.tensor(self.epochs[idx], dtype=torch.float32).unsqueeze(0)  # (1, C, T)
        label = int(self.labels[idx])
        return eeg, label


class Lee2019Dataset_LOSO(Dataset):
    def __init__(self, subjects=[1], pick_channels="all"):
        super().__init__()
        paradigm = SSVEP()
        dataset = Lee2019_SSVEP()

        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)

        subj_ids = np.array(meta['subject'])
        subj_mask = np.isin(subj_ids, subjects)
        X = X[subj_mask]
        labels = labels[subj_mask]
        self.subjects = subj_ids[subj_mask]

        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        self.labels = torch.tensor(encoded_labels, dtype=torch.long)
        self.freqs = le.classes_.astype(float)

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
        raw.filter(l_freq=3, h_freq=60, fir_design="firwin", verbose=False)

        if pick_channels != "all":
            raw.pick(pick_channels)

        raw.resample(250.0, npad="auto")
        raw.set_eeg_reference('average', projection=False)

        self.epochs = raw.get_data().astype(np.float32)
        self.N, self.C, self.T = self.epochs.shape
        self.n_classes = len(np.unique(self.labels))
        self.ch_names = raw.info["ch_names"]
        self.sfreq = raw.info["sfreq"]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        eeg = torch.tensor(self.epochs[idx], dtype=torch.float32).unsqueeze(0)
        label = int(self.labels[idx])
        subj = int(self.subjects[idx])
        return eeg, label, subj