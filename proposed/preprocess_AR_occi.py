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


def safe_float(x):
    """Convert string/number safely to float. Handles fractions like '10 / 8'."""
    try:
        return float(x)
    except Exception:
        expr = str(x).strip()
        # fraction style "a / b"
        if "/" in expr:
            parts = expr.split("/")
            if len(parts) == 2:
                try:
                    num, den = float(parts[0]), float(parts[1])
                    if den != 0:
                        return num / den
                except Exception:
                    pass
        # fallback: extract first number
        m = re.findall(r"[-+]?\d*\.\d+|\d+", expr)
        return float(m[0]) if m else 0.0


def convert_dataset_to_npz(base_dir, save_dir, tasks, fs=1024, T=2, pick_channels=None):
    """
    Convert EEG dataset into compressed .npz files.
    """
    base_dir = Path(base_dir)
    save_dir = Path(save_dir)

    subjects = sorted(base_dir.glob("sub-*"))
    for subject_dir in tqdm(subjects):
        subject = subject_dir.name
        for session_dir in sorted(subject_dir.glob("ses-*")):
            session = session_dir.name
            eeg_dir = subject_dir / session / "eeg"

            all_epochs, all_labels, all_tasks, all_freqs, all_phases = [], [], [], [], []
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
                    raw.filter(l_freq=5, h_freq=95, fir_design="firwin", verbose=False)
                    raw.notch_filter(freqs=[50])

                    # events
                    events_df = pd.read_csv(evt_file, sep="\t")
                    freqs = [safe_float(f) for f in events_df["stim_frequency"].values]
                    phases = [safe_float(p) for p in
                              events_df["stim_phase"].values] if "stim_phase" in events_df.columns else np.zeros(len(freqs))
                    onsets = events_df["onset"].values

                    # map frequency to labels
                    labels = np.arange(len(freqs), dtype=np.int64)

                    # convert onset to sample index
                    if np.median(onsets) < 100:
                        onsets = (onsets * fs).astype(int)
                    else:
                        onsets = onsets.astype(int)

                    # epoching
                    n_samples = int(fs * T)
                    for onset, lab, f, p in zip(onsets, labels, freqs, phases):
                        start, stop = onset, onset + n_samples
                        if stop <= raw.n_times:
                            data, _ = raw[:, start:stop]
                            all_epochs.append(data)
                            all_labels.append(lab)
                            all_tasks.append(task)
                            all_freqs.append(float(f))
                            all_phases.append(float(p))

                if len(all_epochs) == 0:
                    print(f"    [Skip] {subject} {session} (no valid epochs)")
                    continue

                all_epochs = np.stack(all_epochs, axis=0).astype(np.float32)  # (N, C, T)
                all_labels = np.array(all_labels)
                all_tasks = np.array(all_tasks)
                all_freqs = np.array(all_freqs, dtype=float)
                all_phases = np.array(all_phases, dtype=float)

                # save as npz
                out_file = save_dir / f"{subject}_{session}.npz"
                np.savez_compressed(out_file,
                                    epochs=all_epochs,
                                    labels=all_labels,
                                    freqs=all_freqs,
                                    phases=all_phases,
                                    tasks=all_tasks,
                                    ch_names=np.array(raw.ch_names),
                                    sfreq=raw.info["sfreq"])
                print(f"    [Saved] {out_file.name} | x:{all_epochs.shape}, y:{all_labels.shape}, classes:{len(np.unique(all_labels))}, freqs:{all_freqs.shape}, phases:{all_phases.shape}, tasks:{all_tasks.shape}")

            except Exception as e:
                print(f"[Error] {subject} {session} failed | {e}")


# ===================== main =====================
if __name__ == "__main__":
    BASE = "/home/brainlab/Workspace/jycha/SSVEP/data/26764735"
    SAVE = "/home/brainlab/Workspace/jycha/SSVEP/processed_npz_occi"
    TASKS = ["MF", "LF", "SFSP", "SFDP", "DFSP", "DFDP", "DFDP1", "DFDP3", "DFDP5"]
    PICKS = ['PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'POZ', 'O1', 'O2', 'OZ']

    convert_dataset_to_npz(BASE, SAVE, TASKS, fs=1024, T=2, pick_channels=PICKS)