import os
import argparse
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt


def bandpass_filter(data, sfreq, low=6, high=80, order=4):
    nyq = 0.5 * sfreq
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype="band")
    return filtfilt(b, a, data, axis=-1)


def preprocess_beta_subject(mat_file, out_dir):
    mat = sio.loadmat(mat_file)
    eeg = mat["data"][0,0]["EEG"]   # (Channel, Sample, Block, Target) = (64, 3s*250Hz=750, 4, 40) or (64, 4s*250Hz=1000, 4, 40)
    suppl = mat["data"][0,0]["suppl_info"]

    # shape check
    C, T, B, K = eeg.shape
    print(f"[INFO] {mat_file} → EEG shape {eeg.shape}")

    # === trial duration by subject ===
    sid = int(os.path.splitext(os.path.basename(mat_file))[0][1:])  # e.g. "S16" → 16
    if sid <= 15:
        stim_len = 2.0  # seconds
    else:
        stim_len = 3.0

    sfreq = int(suppl["srate"][0, 0][0, 0])
    stim_samples = int(stim_len * sfreq)

    # === stimulation-only segment ===
    # original epoch includes: 0.5s pre + stim + 0.5s post
    start = int(0.5 * sfreq)
    end = start + stim_samples
    eeg_stim = eeg[:, start:end, :, :]  # (C, stim_samples, B, K)

    # apply band-pass filtering
    for b in range(B):
        for k in range(K):
            eeg_stim[:, :, b, k] = bandpass_filter(eeg_stim[:, :, b, k], sfreq, 6, 80)

    # reshape (N, C, T)
    eeg_reshaped = eeg_stim.transpose(3, 2, 0, 1).reshape(K * B, C, stim_samples).astype(np.float32)

    # labels (0 ~ K-1 repeated B times)
    labels = np.tile(np.arange(K), B)

    # block indices (0 ~ B-1 repeated per target)
    blocks = np.repeat(np.arange(B), K)

    # === metadata ===
    freqs_all = np.array(suppl["freqs"][0, 0]).squeeze()  # (40,)
    phases_all = np.array(suppl["phases"][0, 0]).squeeze()  # (40,)

    # freq/phase per trial (expand to N=K*B)
    freqs = np.tile(freqs_all, B)  # (N,)
    phases = np.tile(phases_all, B)  # (N,)

    # block-specific phase correction (0, π/2, π, 3π/2)
    block_phase_shift = {0: 0, 1: np.pi / 2, 2: np.pi, 3: 3 * np.pi / 2}
    phase_shifts = np.array([block_phase_shift[b] for b in blocks], dtype=float)

    phases = phases + phase_shifts  # (N,) 최종 trial별 phase

    # channel names
    chan_raw = suppl["chan"][0, 0].squeeze()
    ch_names = [ch[0] for ch in chan_raw[:, -1]]

    # save npz
    out_file = os.path.join(out_dir, f"S{sid}.npz")
    np.savez(out_file,
             epochs=eeg_reshaped,   # (N, C, T)
             labels=labels,         # (N,)
             blocks=blocks,         # (N,)
             freqs=freqs,           # (N,)
             phases=phases,         # (N,)
             sfreq=sfreq,
             ch_names=np.array(ch_names))

    print(f"[INFO] Saved {out_file}, shape={eeg_reshaped.shape}, samples={eeg_reshaped.shape[0]}, "
          f"classes={K}, blocks={np.unique(blocks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/brainlab/Workspace/jycha/SSVEP/data/12264401")
    parser.add_argument("--out_root", type=str, default="/home/brainlab/Workspace/jycha/SSVEP/processed_beta")
    args = parser.parse_args()

    for f in sorted(os.listdir(args.data_root)):
        if f.endswith(".mat"):
            preprocess_beta_subject(os.path.join(args.data_root, f), args.out_root)