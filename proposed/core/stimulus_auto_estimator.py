# stimulus_auto_estimator.py
import numpy as np
import scipy.signal as sg
import torch


class StimulusAutoEstimator:
    """
    Automatically estimates actual (effective) stimulus frequency & phase
    from EEG using band-pass filtering + Hilbert transform + smoothing.

    Works for AR-SSVEP and jitter-heavy setups.

    Params
    ------
    freq_width : float
        Half-width of frequency search band (Hz).
    smooth_window : int
        Moving average smoothing window.
    min_amp_threshold : float
        Minimum amplitude required to trust the SSVEP response.

    EEG shape
    ---------
    eeg : (B, C, T) or (B, 1, C, T) torch Tensor
    nominal_freq : (B,) Hz
    nominal_phase : (B,) rad
    """

    def __init__(self,
                 freq_width=0.7,        # ±0.7 Hz search band (AR 장치 jitter 고려)
                 smooth_window=5,
                 min_amp_threshold=0.03,
                 debug=False):
        self.freq_width = freq_width
        self.smooth_window = smooth_window
        self.min_amp_threshold = min_amp_threshold
        self.debug = debug

    def estimate(self, eeg, nominal_freq, nominal_phase, sfreq):
        eeg_np = eeg.detach().cpu().numpy()

        # ------------------------------
        # 1) Shape standardization
        # ------------------------------
        if eeg_np.ndim == 4:  # (B, 1, C, T)
            B, _, C, T = eeg_np.shape
            eeg_np = eeg_np.reshape(B, C, T)
        elif eeg_np.ndim == 3:
            B, C, T = eeg_np.shape
        else:
            raise ValueError(f"[StimulusAutoEstimator] Invalid EEG shape: {eeg_np.shape}")

        f0_arr = nominal_freq.detach().cpu().view(-1).numpy()
        p0_arr = nominal_phase.detach().cpu().view(-1).numpy()

        adj_freq = []
        adj_phase = []

        for i in range(B):
            f0 = float(f0_arr[i])
            p0 = float(p0_arr[i])

            # ------------------------------
            # 2) Narrow-band filter range
            # ------------------------------
            low = max(0.1, f0 - self.freq_width)
            high = f0 + self.freq_width
            nyq = sfreq / 2

            if high >= nyq:
                high = nyq - 0.1

            # ------------------------------
            # 3) Channel-averaged EEG
            # ------------------------------
            sig = eeg_np[i].mean(axis=0)

            # ------------------------------
            # 4) Bandpass Filter (Butterworth)
            # ------------------------------
            try:
                sos = sg.butter(4, [low/nyq, high/nyq], btype='bandpass', output='sos')
                filtered = sg.sosfilt(sos, sig)
            except Exception:
                adj_freq.append(f0)
                adj_phase.append(p0)
                continue

            # Low SNR → skip
            if np.max(np.abs(filtered)) < self.min_amp_threshold:
                adj_freq.append(f0)
                adj_phase.append(p0)
                continue

            # ------------------------------
            # 5) Hilbert Transform
            # ------------------------------
            analytic = sg.hilbert(filtered)
            inst_phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(inst_phase) * sfreq / (2*np.pi)

            # ------------------------------
            # 6) Smoothing
            # ------------------------------
            if len(inst_freq) > self.smooth_window:
                kernel = np.ones(self.smooth_window) / self.smooth_window
                inst_freq = sg.convolve(inst_freq, kernel, mode="same")

            # robust representative freq
            f_hat = np.median(inst_freq)
            if np.isnan(f_hat) or f_hat <= 0:
                f_hat = f0

            # ------------------------------
            # 7) Phase initial alignment
            # ------------------------------
            # phase at the first cycle
            phi_hat = inst_phase[0]
            if np.isnan(phi_hat):
                phi_hat = p0

            adj_freq.append(float(f_hat))
            adj_phase.append(float(phi_hat))

            # ---- optional debug print ----
            if self.debug and i == 0:
                print(f"[StimulusAutoEstimator] Nominal {f0:.3f}Hz → Estimated {f_hat:.3f}Hz")
                print(f"Phase: {p0:.3f} rad → {phi_hat:.3f} rad")
                print("----------------------------------------------------")

        adj_freq = torch.tensor(adj_freq, dtype=torch.float32, device=eeg.device)
        adj_phase = torch.tensor(adj_phase, dtype=torch.float32, device=eeg.device)

        return adj_freq, adj_phase