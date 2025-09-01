# -*- coding:utf-8 -*-
import re
import mne
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def load_raw(set_file: Path, fs: int):
    """ Load EEGLAB dataset (.set) and return MNE Raw object. """
    fdt_file = set_file.with_suffix(".fdt")
    ch_file  = set_file.with_name(set_file.name.replace("_eeg.set", "_channels.tsv"))
    json_file= set_file.with_name(set_file.name.replace("_eeg.set", "_eeg.json"))

    # channel names
    ch_df = pd.read_csv(ch_file, sep="\t")
    ch_names = ch_df["name"].tolist()
    n_ch = len(ch_names)

    # sampling rate
    meta = json.loads(Path(json_file).read_text())
    sfreq = float(meta.get("SamplingFrequency", fs))

    # load data
    data = np.fromfile(fdt_file, dtype=np.float32)
    n_times = data.size // n_ch
    data = data.reshape((n_ch, n_times), order="F")

    # create Raw object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def convert_dataset_to_npz(base_dir, save_dir, tasks, fs=1024, T=2, pick_channels=None, l_freq=6, h_freq=80):
    """
    Convert EEG dataset into compressed .npz files.

    Args:
        tasks (list[str]): list of tasks to process
        fs (int): sampling frequency
        T (int): epoch duration (seconds)
        pick_channels (list[str]): subset of EEG channels to keep
        l_freq (float): low cutoff frequency (Hz)
        h_freq (float): high cutoff frequency (Hz)
    """
    base_dir = Path(base_dir)
    save_dir = Path(save_dir)

    subjects = sorted(base_dir.glob("sub-*"))
    for subject_dir in tqdm(subjects):
        subject = subject_dir.name
        for session_dir in sorted(subject_dir.glob("ses-*")):
            session = session_dir.name
            eeg_dir = subject_dir / session / "eeg"

            all_epochs, all_labels, all_tasks = [], [], []
            try:
                for task in tasks:
                    set_file = eeg_dir / f"{subject}_{session}_task-{task}_eeg.set"
                    evt_file = eeg_dir / f"{subject}_{session}_task-{task}_events.tsv"
                    if not set_file.exists() or not evt_file.exists():
                        continue

                    # load raw EEG
                    raw = load_raw(set_file, fs)

                    # channel selection
                    if pick_channels is not None:
                        picks = [ch for ch in pick_channels if ch in raw.ch_names]
                        raw.pick(picks)

                    # bandpass filter
                    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)

                    # events
                    events_df = pd.read_csv(evt_file, sep="\t")
                    freqs = events_df["stim_frequency"].values
                    onsets = events_df["onset"].values

                    # map frequency to labels
                    labels = []
                    for f in freqs:
                        if isinstance(f, str):
                            m = re.findall(r"\d+", f)
                            labels.append(int(m[0]) if m else -1)
                        else:
                            labels.append(int(f))
                    labels = np.array(labels, dtype=np.int64)

                    # convert onset to sample index
                    if np.median(onsets) < 100:
                        onsets = (onsets * fs).astype(int)
                    else:
                        onsets = onsets.astype(int)

                    # epoching
                    n_samples = int(fs * T)
                    for onset, lab in zip(onsets, labels):
                        start, stop = onset, onset + n_samples
                        if stop <= raw.n_times:
                            data, _ = raw[:, start:stop]
                            all_epochs.append(data)
                            all_labels.append(lab)
                            all_tasks.append(task)

                if len(all_epochs) == 0:
                    print(f"    [Skip] {subject} {session} (no valid epochs)")
                    continue

                all_epochs = np.stack(all_epochs, axis=0).astype(np.float32)  # (N, C, T)
                all_labels = np.array(all_labels)
                all_tasks = np.array(all_tasks)

                # save as npz
                out_file = save_dir / f"{subject}_{session}.npz"
                np.savez_compressed(out_file,
                                    epochs=all_epochs,
                                    labels=all_labels,
                                    tasks=all_tasks,
                                    ch_names=np.array(raw.ch_names),
                                    sfreq=raw.info["sfreq"])
                print(f"    [Saved] {out_file.name} | x:{all_epochs.shape}, y:{all_labels.shape}, tasks:{all_tasks.shape}")

            except Exception as e:
                print(f"[Error] {subject} {session} failed | {e}")


# ===================== main =====================
if __name__ == "__main__":
    BASE = "your base path"
    SAVE = "your save path"
    TASKS = ["MF", "LF", "SFSP", "SFDP", "DFSP", "DFDP", "DFDP1", "DFDP3", "DFDP5"]
    PICKS = ['PO1', 'PO2', 'P1', 'P5', 'PO7', 'CP2', 'PO8', 'P6', 'CP6', 'CP5', 'PO5', 'CP1',
             'PO4', 'P2', 'CP7', 'PO3', 'POZ', 'PO6', 'CP8', 'CP3', 'CPZ', 'CP4', 'M1', 'M2',
             'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2']

    convert_dataset_to_npz(BASE, SAVE, TASKS, fs=1024, T=2, pick_channels=PICKS)