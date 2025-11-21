import numpy as np
import scipy.signal as sg
import torch

class StimulusAutoEstimator:
    """
    Dataset-agnostic estimator (no nominal freq)
    - Finds dominant frequency only from EEG
    - Works with AR, Lee, Nakanishi, BETA datasets
    """

    def __init__(self,
                 search_range=(6.0, 20.0),   # 탐색 범위 (Hz)
                 freq_step=0.1,              # resolution
                 smooth_window=5,
                 min_amp_threshold=0.03,
                 debug=False):
        self.search_range = search_range
        self.freq_step = freq_step
        self.smooth_window = smooth_window
        self.min_amp_threshold = min_amp_threshold
        self.debug = debug

    @torch.no_grad()
    def estimate(self, eeg, sfreq):
        """
        EEG only → dominant frequency detection
        return: adj_freq (B,)
        """

        eeg_np = eeg.detach().cpu().numpy()

        if eeg_np.ndim == 4:
            B, _, C, T = eeg_np.shape
            eeg_np = eeg_np.reshape(B, C, T)
        else:
            B, C, T = eeg_np.shape

        adj_freq = []

        for i in range(B):

            # channel-average for better SNR
            sig = eeg_np[i].mean(axis=0)

            # ------------ Short-Time Fourier Transform (STFT) ------------
            f, _, Zxx = sg.stft(sig, fs=sfreq, nperseg=sfreq//2)
            power = np.abs(Zxx).mean(axis=1)  # (F,)

            # restrict to search range
            valid = (f >= self.search_range[0]) & (f <= self.search_range[1])
            f_valid = f[valid]
            p_valid = power[valid]

            if np.max(p_valid) < self.min_amp_threshold:
                # 신호가 너무 약하면 0 Hz 반환
                adj_freq.append(0.0)
                continue

            # dominant freq
            f_hat = float(f_valid[np.argmax(p_valid)])
            adj_freq.append(f_hat)

            if self.debug and i == 0:
                print(f"[StimEstimator] Dominant freq detected: {f_hat:.3f} Hz")

        return torch.tensor(adj_freq, dtype=torch.float32, device=eeg.device)


class ARExp1_AutoEstimator:
    def __init__(self,
                 lf_range=(7.5, 15.5),      # LF
                 mf_range=(22.0, 31.0),     # MF
                 freq_step=0.05,
                 smooth_window=5,
                 min_amp_threshold=0.03,
                 debug=False):
        self.lf_range = lf_range
        self.mf_range = mf_range
        self.freq_step = freq_step
        self.smooth_window = smooth_window
        self.min_amp_threshold = min_amp_threshold
        self.debug = debug

    @torch.no_grad()
    def estimate(self, eeg, sfreq):
        eeg_np = eeg.detach().cpu().numpy()

        if eeg_np.ndim == 4:
            B, _, C, T = eeg_np.shape
            eeg_np = eeg_np.reshape(B, C, T)
        else:
            B, C, T = eeg_np.shape

        adj_freq = []

        for i in range(B):
            sig = eeg_np[i].mean(axis=0)

            f, _, Zxx = sg.stft(sig, fs=sfreq, nperseg=sfreq//2)
            power = np.abs(Zxx).mean(axis=1)

            # ---- LF peak ----
            lf_mask = (f >= self.lf_range[0]) & (f <= self.lf_range[1])
            f_lf = f[lf_mask]
            p_lf = power[lf_mask]
            lf_hat = f_lf[np.argmax(p_lf)] if len(f_lf) > 0 else None

            # ---- MF peak ----
            mf_mask = (f >= self.mf_range[0]) & (f <= self.mf_range[1])
            f_mf = f[mf_mask]
            p_mf = power[mf_mask]
            mf_hat = f_mf[np.argmax(p_mf)] if len(f_mf) > 0 else None

            # ---- Choose stronger ----
            lf_max = np.max(p_lf) if len(p_lf) > 0 else -1
            mf_max = np.max(p_mf) if len(p_mf) > 0 else -1

            if lf_max < self.min_amp_threshold and mf_max < self.min_amp_threshold:
                adj_freq.append(0.0)
            else:
                f_hat = lf_hat if lf_max >= mf_max else mf_hat
                adj_freq.append(float(f_hat))

            if self.debug and i == 0:
                print(f"[StimEstimator] LF={lf_hat:.2f}Hz({lf_max:.3f}) "
                      f" MF={mf_hat:.2f}Hz({mf_max:.3f}) "
                      f"→ Selected {f_hat:.2f}Hz")

        return torch.tensor(adj_freq, dtype=torch.float32, device=eeg.device)