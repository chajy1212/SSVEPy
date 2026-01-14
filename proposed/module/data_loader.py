# -*- coding:utf-8 -*-
import os, glob
import mne
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from scipy.signal import butter, lfilter
from sklearn.preprocessing import LabelEncoder

from moabb.paradigms import SSVEP
from moabb.datasets import Nakanishi2015
from moabb.datasets import Lee2019_SSVEP
from SSVEPAnalysisToolbox.datasets.betadataset import BETADataset


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # data shape: (Channel, Time)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=-1)
    return y


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


class Nakanishi2015Dataset(Dataset):
    def __init__(self, subjects=[1], pick_channels="all"):
        dataset = Nakanishi2015()
        dataset.subject_list = list(range(1, 11))

        paradigm = SSVEP(fmin=6, fmax=80, tmin=0.0, tmax=4.16)

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

        self.sfreq = raw.info["sfreq"]
        self.epochs = raw.get_data().astype(np.float32)  # (N, C, T)
        self.N, self.C, self.T = self.epochs.shape
        self.n_classes = len(np.unique(self.labels))
        self.ch_names = raw.info["ch_names"]

        print(f"  -> [Nakanishi] Loaded Subjects {subjects} | Shape: {self.epochs.shape} | [0.0s-4.16s]")

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
        paradigm = SSVEP(fmin=3, fmax=60, tmin=0.0, tmax=4.0)
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
        paradigm = SSVEP(fmin=3, fmax=60, tmin=0, tmax=4.0)
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


class TorchBETADataset(Dataset):
    def __init__(self, subjects, data_root, pick_channels="all", time_window=None, stride=None):
        self.bs = BETADataset(path=data_root)
        self.subjects = subjects
        self.samples = []

        stim_freqs = self.bs.stim_info["freqs"]
        stim_phases = self.bs.stim_info["phases"]
        self.sfreq = self.bs.srate

        self.original_channels = self.bs.channels
        if pick_channels == "all":
            self.picks = list(range(len(self.original_channels)))
        else:
            if isinstance(pick_channels, str):
                target_chs = [ch.strip() for ch in pick_channels.split(',')]
            else:
                target_chs = pick_channels

            available_chs_upper = [ch.upper() for ch in self.original_channels]
            target_chs_upper = [ch.upper() for ch in target_chs]

            self.picks = []
            for target in target_chs_upper:
                if target in available_chs_upper:
                    self.picks.append(available_chs_upper.index(target))

        if time_window:
            self.win_pts = int(time_window * self.sfreq)
            self.stride_pts = int((stride if stride else time_window) * self.sfreq)
        else:
            self.win_pts = None

        # BETA: Cue 0.5s + Stim (2s or 3s) + Rest 0.5s
        cut_front_sec = 0.5
        cut_back_sec = 0.5

        cut_front_pts = int(cut_front_sec * self.sfreq)     # 0.5 * 250 = 125 sample
        cut_back_pts = int(cut_back_sec * self.sfreq)       # 125 sample

        for s in subjects:
            sub_idx = s - 1
            data = self.bs.get_sub_data(sub_idx)    # (Block, Class, Ch, Time)
            B, K, C, T_total = data.shape           # T_total: 1000 (4s) or 750 (3s)

            for b in range(B):
                for k in range(K):
                    raw_sig = data[b, k, self.picks, :]

                    # Slicing (S1~15: 2s, S16~70: 3s)
                    raw_sig = raw_sig[:, cut_front_pts: T_total - cut_back_pts]

                    # Voltage Scaling (Volt -> uV)
                    raw_sig = raw_sig * 1e6

                    # Filter (3-60Hz)
                    raw_sig = butter_bandpass_filter(raw_sig, 3.0, 60.0, self.sfreq, order=4)

                    current_len = raw_sig.shape[-1]

                    # Windowing
                    if self.win_pts is not None and self.win_pts <= current_len:
                        start = 0
                        while start + self.win_pts <= current_len:
                            crop_sig = raw_sig[:, start: start + self.win_pts]
                            self.samples.append((
                                crop_sig.astype(np.float32),
                                k, stim_freqs[k], stim_phases[k], b
                            ))
                            start += self.stride_pts
                    else:
                        self.samples.append((
                            raw_sig.astype(np.float32),
                            k, stim_freqs[k], stim_phases[k], b
                        ))

        self.n_classes = len(stim_freqs)
        self.stim_info = {'freqs': stim_freqs, 'phases': stim_phases}
        if len(self.samples) > 0:
            self.C = self.samples[0][0].shape[0]
            self.T = self.samples[0][0].shape[1]
        else:
            self.C, self.T = 0, 0
        self.blocks = np.array([s[4] for s in self.samples])

        print(f"  -> [BETA] Loaded Subjects {subjects} | Shape: {self.epochs.shape} | [0.0s-3.00s]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sig, lbl, freq, phase, block = self.samples[idx]
        eeg = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)
        return eeg, int(lbl), float(freq), float(phase), int(block)


class Wang2016Dataset(Dataset):
    def __init__(self, subjects, data_root, pick_channels="occipital"):
        self.data_root = data_root
        self.subjects = subjects

        # Frequency: 8Hz ~ 15.8Hz (0.2Hz step), 40 targets
        self.n_classes = 40
        self.freqs = np.arange(8.0, 15.8 + 0.01, 0.2)

        # Phase: 0, 0.5pi, 1.0pi ... (0.5pi step)
        self.phases = (np.arange(40) * 0.5 * np.pi) % (2 * np.pi)

        # Standard Channel Names (64 ch)
        self.ch_names = [
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
            'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
            'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
            'CB1', 'O1', 'OZ', 'O2', 'CB2'
        ]

        self.sfreq = 250.0  # Downsampled rate

        all_epochs = []
        all_labels = []

        # 6s epoch (-0.5s pre, +5.5s post). Stim onset at 0.5s (index 125).
        # Stim duration: 5s.
        # Target: Pure Stimulation [0.0s, 5.0s] relative to onset.
        # Indices in array: 125 (0.5s) to 1375 (5.5s) -> Total 1250 samples (5s)
        start_idx = int(0.5 * self.sfreq)
        end_idx = start_idx + int(5.0 * self.sfreq)

        for subj in subjects:
            file_path = os.path.join(data_root, f"S{subj}.mat")
            if not os.path.exists(file_path):
                print(f"[Error] File not found: {file_path}")
                continue

            try:
                mat = sio.loadmat(file_path)
                # Raw shape: (64, 1500, 40, 6) -> (Chan, Time, Class, Block)
                raw_data = mat['data']
            except Exception as e:
                print(f"[Error] Loading {file_path}: {e}")
                continue

            # Transpose to: (Block, Target, Chan, Time) -> (6, 40, 64, 1500)
            # Block index moves to 0, Target to 1
            data = np.transpose(raw_data, (3, 2, 0, 1))

            # Reshape to List of Trials: (240, 64, 1500)
            # Order: Block 0 (Target 0..39), Block 1 (Target 0..39)...
            data = data.reshape(-1, 64, 1500)

            # Slicing (Pure Stim)
            data = data[..., start_idx:end_idx]

            # Create Labels: 0~39 repeated 6 times
            labels = np.tile(np.arange(40), 6)  # (240,)

            all_epochs.append(data)
            all_labels.append(labels)

        if len(all_epochs) == 0:
            self.N = 0
            return

        # Concatenate all subjects
        self.epochs = np.concatenate(all_epochs, axis=0)  # (N_total, 64, 1500)
        self.labels = np.concatenate(all_labels, axis=0)

        # Bandpass Filter (3-70Hz) - crucial for removing drift and noise
        self.epochs = butter_bandpass_filter(self.epochs, 3.0, 70.0, self.sfreq, order=4)

        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg')
        epochs_mne = mne.EpochsArray(self.epochs, info, verbose=False)

        if pick_channels == "occipital":
            targets = ['PZ', 'POZ', 'PO3', 'PO4', 'PO5', 'PO6', 'OZ', 'O1', 'O2']
            available = [ch for ch in self.ch_names if ch.upper() in targets]
            if len(available) > 0:
                epochs_mne.pick(available)
            else:
                epochs_mne.pick(np.arange(54, 63))  # Fallback index

        elif isinstance(pick_channels, list):
            epochs_mne.pick(pick_channels)

        self.epochs = epochs_mne.get_data().astype(np.float32)  # (N, C, T)
        self.N, self.C, self.T = self.epochs.shape
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        print(f"  -> [Wang2016] Loaded Subjects {subjects} | Shape: {self.epochs.shape} | [0.5s-5.5s]")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        eeg = torch.tensor(self.epochs[idx], dtype=torch.float32).unsqueeze(0)
        label = int(self.labels[idx])
        return eeg, label