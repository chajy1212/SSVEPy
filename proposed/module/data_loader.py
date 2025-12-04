# -*- coding:utf-8 -*-
import os, glob
import mne
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from moabb.paradigms import SSVEP
from moabb.datasets import Nakanishi2015
from moabb.datasets import Lee2019_SSVEP
from SSVEPAnalysisToolbox.datasets.betadataset import BETADataset


class ARDataset(Dataset):
    def __init__(self, data_root, subject, exp_name, session="train"):
        self.data_root = data_root
        self.subject = int(str(subject).replace("S", ""))
        self.exp_name = exp_name
        self.session = session.lower()

        pattern = os.path.join(
            data_root,
            f"sub-{self.subject:03d}_ses-0{1 if self.session == 'train' else 2}.npz"
        )
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"[ERROR] No file found for {self.exp_name} / Subject {self.subject:02d} / {self.session}")

        npz_file = files[0]
        data = np.load(npz_file, allow_pickle=True)

        self.epochs = data["epochs"]  # (N,C,T)
        self.labels = data["labels"]  # (N,)
        self.freqs = data["freqs"]    # (N,)
        self.phases = data["phases"]  # (N,)
        self.tasks = data["tasks"]    # (N,)
        self.ch_names = data["ch_names"]
        self.sfreq = float(data["sfreq"])

        self.N, self.C, self.T = self.epochs.shape

        # Filter tasks by experiment
        valid_tasks_by_exp = {
            "Exp1": ["LF", "MF"],
            "Exp2": ["SFSP", "SFDP", "DFSP", "DFDP"],
            "Exp3": ["DFDP1", "DFDP3", "DFDP5"]
        }

        valid_tasks = valid_tasks_by_exp.get(self.exp_name, None)
        if valid_tasks is not None:
            mask = np.isin(self.tasks, valid_tasks)
            self.epochs = self.epochs[mask]
            self.labels = self.labels[mask]
            self.freqs = self.freqs[mask]
            self.phases = self.phases[mask]
            self.tasks = self.tasks[mask]
            self.N = len(self.labels)
            print(f"[FILTER] {self.exp_name}: Included tasks → {np.unique(self.tasks)}")

        # Map frequency ↔ class index
        freqs_rounded = np.round(self.freqs, 2)
        phases_rounded = np.round(self.phases, 2)

        # Combine unique pairs
        unique_pairs = sorted(set(zip(freqs_rounded, phases_rounded)))

        # Build mapping
        self.freq_phase2class = {pair: i for i, pair in enumerate(unique_pairs)}
        self.class2freq_phase = {i: pair for pair, i in self.freq_phase2class.items()}
        self.n_classes = len(unique_pairs)

        # Update attributes
        self.freqs = freqs_rounded
        self.phases = phases_rounded

        print(f"[INFO] Loaded {npz_file} | {self.exp_name}, Sub-{self.subject:02d}, {self.session}, Classes={self.n_classes}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # EEG input (C,T) → (1,C,T)
        eeg = torch.tensor(self.epochs[idx], dtype=torch.float32).unsqueeze(0)  # (1, C, T)

        # Convert Hz label → class index
        freq_val = float(self.freqs[idx])
        phase_val = float(self.phases[idx])
        class_label = self.freq_phase2class[(freq_val, phase_val)]

        # Task name
        task = self.tasks[idx]

        return eeg, class_label, freq_val, phase_val, task


class ExpARDataset(Dataset):
    def __init__(self, data_root, subject, exp_name, session="train"):
        self.data_root = data_root
        self.subject = int(str(subject).replace("S", ""))
        self.exp_name = exp_name
        self.session = session.lower()

        pattern = os.path.join(
            data_root,
            f"sub-{self.subject:03d}_ses-0{1 if self.session == 'train' else 2}.npz"
        )
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"[ERROR] No file found for {self.exp_name} / Subject {self.subject:02d} / {self.session}")

        npz_file = files[0]
        data = np.load(npz_file, allow_pickle=True)

        self.epochs = data["epochs"]  # (N,C,T)
        self.labels = data["labels"]  # (N,)
        self.freqs = data["freqs"]    # (N,)
        self.phases = data["phases"]  # (N,)
        self.tasks = data["tasks"]    # (N,)
        self.ch_names = data["ch_names"]
        self.sfreq = float(data["sfreq"])

        self.N, self.C, self.T = self.epochs.shape

        # Filter tasks by experiment
        valid_tasks_by_exp = {
            "Exp1": ["LF", "MF"],
            "Exp2": ["SFSP", "SFDP", "DFSP", "DFDP"],
            "Exp3": ["DFDP1", "DFDP3", "DFDP5"]
        }

        valid_tasks = valid_tasks_by_exp.get(self.exp_name, None)
        if valid_tasks is not None:
            mask = np.isin(self.tasks, valid_tasks)
            self.epochs = self.epochs[mask]
            self.labels = self.labels[mask]
            self.freqs = self.freqs[mask]
            self.phases = self.phases[mask]
            self.tasks = self.tasks[mask]
            self.N = len(self.labels)
            print(f"[FILTER] {self.exp_name}: Included tasks → {np.unique(self.tasks)}")

        # Map frequency ↔ class index
        freqs_rounded = np.round(self.freqs, 2)
        phases_rounded = np.round(self.phases, 2)

        # Combine unique pairs
        unique_pairs = sorted(set(zip(freqs_rounded, phases_rounded)))

        # Build mapping
        self.freq_phase2class = {pair: i for i, pair in enumerate(unique_pairs)}
        self.class2freq_phase = {i: pair for pair, i in self.freq_phase2class.items()}
        self.n_classes = len(unique_pairs)

        # Update attributes
        self.freqs = freqs_rounded
        self.phases = phases_rounded

        print(f"[INFO] Loaded {npz_file} | {self.exp_name}, Sub-{self.subject:02d}, {self.session}, Classes={self.n_classes}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # EEG input (C,T) → (1,C,T)
        eeg = torch.tensor(self.epochs[idx], dtype=torch.float32).unsqueeze(0)  # (1, C, T)

        # Convert Hz label → class index
        freq_val = float(self.freqs[idx])
        phase_val = float(self.phases[idx])
        class_label = self.freq_phase2class[(freq_val, phase_val)]

        # Task name
        task = self.tasks[idx]

        return eeg, class_label, freq_val, task


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


class ExpNakanishi2015Dataset(Dataset):
    def __init__(self, subjects=[1], pick_channels="all"):
        dataset = Nakanishi2015()
        dataset.subject_list = list(range(1, 11))
        paradigm = SSVEP()
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)

        # Label encoding
        le = LabelEncoder()
        label_ids = le.fit_transform(labels)
        class_freqs = le.classes_.astype(float)

        self.labels = label_ids
        self.freqs = class_freqs

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

        # nominal frequency (Hz)
        nominal_freq = float(self.freqs[label])

        return eeg, label, nominal_freq


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


class ExpLee2019Dataset(Dataset):
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

        self.labels = torch.tensor(encoded_labels, dtype=torch.long)  # class index
        self.freqs = le.classes_.astype(float)  # 클래스별 Hz

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

        # Preprocessing
        raw.resample(250.0, npad="auto")
        raw.filter(l_freq=1, h_freq=40, fir_design="firwin", verbose=False)
        raw.set_eeg_reference('average', projection=False)

        self.epochs = raw.get_data().astype(np.float32)  # (N,C,T)
        self.N, self.C, self.T = self.epochs.shape
        self.n_classes = len(np.unique(self.labels))
        self.ch_names = raw.info["ch_names"]
        self.sfreq = raw.info["sfreq"]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        eeg = torch.tensor(self.epochs[idx], dtype=torch.float32).unsqueeze(0)
        label = int(self.labels[idx])

        # nominal_freq 추가
        nominal_freq = float(self.freqs[label])

        return eeg, label, nominal_freq


class Lee2019Dataset_LOSO(Dataset):
    def __init__(self, subjects=[1], pick_channels="all"):
        super().__init__()
        paradigm = SSVEP()
        dataset = Lee2019_SSVEP()

        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)

        # 피험자 필드 추가
        subj_ids = np.array(meta['subject'])

        # 대상 피험자만 선택
        subj_mask = np.isin(subj_ids, subjects)
        X = X[subj_mask]
        labels = labels[subj_mask]
        self.subjects = subj_ids[subj_mask]

        # 라벨 인코딩
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

        if pick_channels != "all":
            raw.pick(pick_channels)

        raw.resample(250.0, npad="auto")
        raw.filter(l_freq=1, h_freq=40, fir_design="firwin", verbose=False)
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
        subj = int(self.subjects[idx])  # 피험자 ID와 함께 반환
        return eeg, label, subj


class ExpLee2019Dataset_LOSO(Dataset):
    def __init__(self, subjects=[1], pick_channels="all"):
        super().__init__()
        paradigm = SSVEP()
        dataset = Lee2019_SSVEP()

        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)

        # 피험자 필드 추가
        subj_ids = np.array(meta['subject'])

        # 대상 피험자만 선택
        subj_mask = np.isin(subj_ids, subjects)

        X = X[subj_mask]
        labels = labels[subj_mask]
        self.subjects = subj_ids[subj_mask]

        # 라벨 인코딩
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

        if pick_channels != "all":
            raw.pick(pick_channels)

        # Preprocessing
        raw.resample(250.0, npad="auto")
        raw.filter(l_freq=8, h_freq=30, fir_design="firwin", verbose=False)
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

        # nominal_freq 추가
        nominal_freq = float(self.freqs[label])

        return eeg, label, nominal_freq, subj


class TorchBETADataset(Dataset):
    def __init__(self, subjects, data_root, pick_channels="all"):
        """
        subjects: list of subject numbers (e.g. [16,17,...])
        data_root: path to folder containing the .mat files / BETA data
        pick_channels: “all” or list of channel names
        """
        # Initialize toolbox dataset
        # The BETADataset constructor takes `path` (root) and optional support path etc.
        self.bs = BETADataset(path=data_root)

        self.subjects = subjects
        self.pick_channels = pick_channels

        # Build index map of (sub_idx, block, trial)
        self.index_map = []
        for s in subjects:
            sub_idx = s - 1  # toolbox uses 0-based indexing internally
            data = self.bs.get_sub_data(sub_idx)  # shape (block, stimulus, ch, samples)
            B, K, C, T = data.shape
            for b in range(B):
                for k in range(K):
                    self.index_map.append((sub_idx, b, k))

        # Preload epochs, labels, freqs, phases, blocks
        epochs = []
        labels = []
        freqs = []
        phases = []
        blocks = []
        stim_freqs = self.bs.stim_info["freqs"]
        stim_phases = self.bs.stim_info["phases"]
        block_num = self.bs.block_num

        for (sub_idx, b, k) in self.index_map:
            data = self.bs.get_sub_data(sub_idx)
            sig = data[b, k, :, :]  # shape (C, T)
            epochs.append(sig.astype(float))
            labels.append(k)
            freqs.append(stim_freqs[k])
            phases.append(stim_phases[k])
            # global block index: you can choose whatever scheme; here:
            blocks.append(b + sub_idx * block_num)

        import numpy as np
        self.epochs = np.stack(epochs, axis=0)    # (N, C, T)
        self.labels = np.array(labels, dtype=int)
        self.freqs = np.array(freqs, dtype=float)
        self.phases = np.array(phases, dtype=float)
        self.blocks = np.array(blocks, dtype=int)

        # apply pick_channels if not all
        if pick_channels != "all":
            name_map = {ch: i for i, ch in enumerate(self.bs.channels)}
            picks = [name_map[ch] for ch in pick_channels if ch in name_map]
            if len(picks) == 0:
                raise ValueError(f"No valid channel in pick_channels {pick_channels}")
            self.epochs = self.epochs[:, picks, :]

        self.N, self.C, self.T = self.epochs.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sig = self.epochs[idx]  # (C, T)
        lbl = int(self.labels[idx])
        freq = float(self.freqs[idx])
        phase = float(self.phases[idx])
        block = int(self.blocks[idx])
        eeg = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)  # (1, C, T)
        return eeg, lbl, freq, phase, block


class ExpBETADataset(Dataset):
    def __init__(self, subjects, data_root, pick_channels="all", blocks=None):
        """
        subjects : [16]                   → 한 명 또는 여러 명
        blocks   : [0,1,2] 또는 [3] 형태   → 어떤 block을 포함할지 선택
        """
        self.bs = BETADataset(path=data_root)
        self.subjects = subjects
        self.pick_channels = pick_channels
        self.blocks_filter = blocks

        # Build index map of (sub_idx, block, trial)
        self.index_map = []
        for s in subjects:
            sub_idx = s - 1  # toolbox uses 0-based indexing internally
            data = self.bs.get_sub_data(sub_idx)  # shape (block, stimulus, ch, samples)
            B, K, C, T = data.shape

            for b in range(B):          # 0~3 block
                for k in range(K):      # 40 classes
                    self.index_map.append((sub_idx, b, k))

        # Preload data
        epochs, labels, freqs, blocks_list = [], [], [], []
        stim_freqs = self.bs.stim_info["freqs"]

        for (sub_idx, b, k) in self.index_map:
            # block filtering
            if self.blocks_filter is not None:
                if b not in self.blocks_filter:
                    continue

            data = self.bs.get_sub_data(sub_idx)
            sig = data[b, k, :, :]

            epochs.append(sig.astype(float))
            labels.append(k)
            freqs.append(stim_freqs[k])
            blocks_list.append(b)

        self.epochs = np.stack(epochs, axis=0)
        self.labels = np.array(labels)
        self.freqs = np.array(freqs)
        self.blocks = np.array(blocks_list)

        # apply pick_channels
        if pick_channels != "all":
            name_map = {ch: i for i, ch in enumerate(self.bs.channels)}
            picks = [name_map[ch] for ch in pick_channels if ch in name_map]
            self.epochs = self.epochs[:, picks, :]

        self.N, self.C, self.T = self.epochs.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sig = self.epochs[idx]  # (C, T)
        eeg = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)  # (1, C, T)
        lbl = int(self.labels[idx])
        freq = float(self.freqs[idx])
        block = int(self.blocks[idx])

        return eeg, lbl, freq, block