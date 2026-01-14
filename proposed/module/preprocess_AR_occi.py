# -*- coding:utf-8 -*-
import re
import mne
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def load_raw(set_file: Path, fs: int):
    """
    Load EEGLAB dataset (.set/.fdt) and return an MNE Raw object.

    Args:
        set_file (Path): Path to the .set file.
        fs (int): Default sampling frequency if not found in metadata.

    Returns:
        mne.io.RawArray: Loaded raw EEG data.
    """
    fdt_file = set_file.with_suffix(".fdt")
    ch_file  = set_file.with_name(set_file.name.replace("_eeg.set", "_channels.tsv"))
    json_file= set_file.with_name(set_file.name.replace("_eeg.set", "_eeg.json"))

    # Load Channel Names
    ch_df = pd.read_csv(ch_file, sep="\t")
    ch_names = ch_df["name"].tolist()
    n_ch = len(ch_names)

    # Get Sampling Rate
    meta = json.loads(Path(json_file).read_text())
    sfreq = float(meta.get("SamplingFrequency", fs))

    # Load Binary Data (.fdt)
    data = np.fromfile(fdt_file, dtype=np.float32)
    n_times = data.size // n_ch
    data = data.reshape((n_ch, n_times), order="F")     # Fortran-like order for EEGLAB

    # Create MNE Raw Object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    return raw


def safe_float(x):
    """
    Convert string/number safely to float.
    Handles complex formats like fractions (e.g., '10 / 8').
    """
    try:
        return float(x)
    except Exception:
        expr = str(x).strip()
        # Handle fraction style "a / b"
        if "/" in expr:
            parts = expr.split("/")
            if len(parts) == 2:
                try:
                    num, den = float(parts[0]), float(parts[1])
                    if den != 0:
                        return num / den
                except Exception:
                    pass
        # Fallback: extract the first numeric sequence
        m = re.findall(r"[-+]?\d*\.\d+|\d+", expr)
        return float(m[0]) if m else 0.0


def convert_dataset_to_npz(base_dir, save_dir, tasks, fs=1024, pick_channels=None):
    """
    Convert AR-SSVEP EEG dataset into compressed .npz files.

    Preprocessing steps (Paper Standards):
    1. Notch Filter: 49–51 Hz (Target 50Hz)
    2. Band-pass Filter: 5–95 Hz
    3. Epoch Window: [-0.5, 3.14] sec (Total 3.64 sec)
    4. Labeling: Unique mapping of (Frequency, Phase) pairs to Class IDs
    """
    base_dir = Path(base_dir)
    save_dir = Path(save_dir)

    # Epoch Parameters
    tmin = -0.5
    tmax = 3.14
    T = tmax - tmin   # 3.64 sec
    n_samples = int(T * fs)

    subjects = sorted(base_dir.glob("sub-*"))

    for subject_dir in tqdm(subjects):
        subject = subject_dir.name

        for session_dir in sorted(subject_dir.glob("ses-*")):
            session = session_dir.name
            eeg_dir = subject_dir / session / "eeg"

            all_epochs, all_labels, all_tasks = [], [], []
            all_freqs, all_phases = [], []

            try:
                for task in tasks:
                    set_file = eeg_dir / f"{subject}_{session}_task-{task}_eeg.set"
                    evt_file = eeg_dir / f"{subject}_{session}_task-{task}_events.tsv"

                    if not set_file.exists() or not evt_file.exists():
                        continue

                    raw = load_raw(set_file, fs)

                    if pick_channels is not None:
                        picks = [ch for ch in pick_channels if ch in raw.ch_names]
                        raw.pick(picks)

                    raw.notch_filter(freqs=[50], notch_widths=2, verbose=False)
                    raw.filter(l_freq=5, h_freq=95, fir_design="firwin", verbose=False)

                    # Load Events
                    events_df = pd.read_csv(evt_file, sep="\t")

                    freqs = [safe_float(f) for f in events_df["stim_frequency"].values]
                    phases = [safe_float(p) for p in
                              events_df["stim_phase"].values] if "stim_phase" in events_df.columns else np.zeros(len(freqs))

                    onsets = events_df["onset"].values

                    # Convert onset to samples (Handle seconds vs samples logic)
                    if np.median(onsets) < 100:
                        onsets = (onsets * fs).astype(int)
                    else:
                        onsets = onsets.astype(int)

                    # Class Mapping (Unique Frequency-Phase Pairs)
                    freq_phase_pairs = list(zip(freqs, phases))
                    unique_pairs = {pair: idx for idx, pair in enumerate(sorted(set(freq_phase_pairs)))}

                    labels = [unique_pairs[pair] for pair in freq_phase_pairs]

                    # Epoching
                    for onset, lab, f, p in zip(onsets, labels, freqs, phases):
                        start = onset + int(tmin * fs)
                        stop = start + n_samples

                        if start >= 0 and stop <= raw.n_times:
                            data, _ = raw[:, start:stop]
                            all_epochs.append(data)
                            all_labels.append(lab)
                            all_tasks.append(task)
                            all_freqs.append(float(f))
                            all_phases.append(float(p))

                if len(all_epochs) == 0:
                    print(f"    [Skip] {subject} {session} (no valid epochs)")
                    continue

                all_epochs = np.stack(all_epochs, axis=0).astype(np.float32)

                out_file = save_dir / f"{subject}_{session}.npz"
                np.savez_compressed(
                    out_file,
                    epochs=all_epochs,
                    labels=np.array(all_labels),
                    freqs=np.array(all_freqs, dtype=float),
                    phases=np.array(all_phases, dtype=float),
                    tasks=np.array(all_tasks),
                    ch_names=np.array(raw.ch_names),
                    sfreq=raw.info["sfreq"]
                )

                print(f"    [Saved] {out_file.name} | x:{all_epochs.shape}, classes:{len(set(all_labels))}")

            except Exception as e:
                print(f"[Error] {subject} {session} failed | {e}")


# ===================== main =====================
if __name__ == "__main__":
    BASE = "/home/brainlab/Workspace/jycha/SSVEP/data/AR"
    SAVE = "/home/brainlab/Workspace/jycha/SSVEP/processed_npz_occi"
    TASKS = ["MF", "LF", "SFSP", "SFDP", "DFSP", "DFDP", "DFDP1", "DFDP3", "DFDP5"]
    PICKS = ['PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'POZ', 'O1', 'O2', 'OZ']

    convert_dataset_to_npz(BASE, SAVE, TASKS, fs=1024, pick_channels=PICKS)