import numpy as np
import scipy.signal as sg
import torch

class StimulusAutoEstimator:
    """
    Frequency-only estimator
    Stable across all datasets:
    - Nakanishi
    - Lee
    - BETA
    - AR
    """

    def __init__(self,
                 freq_width=0.7,
                 smooth_window=5,
                 min_amp_threshold=0.03,
                 debug=False):
        self.freq_width = freq_width
        self.smooth_window = smooth_window
        self.min_amp_threshold = min_amp_threshold
        self.debug = debug

    @torch.no_grad()
    def estimate(self, eeg, nominal_freq, sfreq):
        """
        Returns:
            adj_freq : (B,)
        """

        eeg_np = eeg.detach().cpu().numpy()

        # shape: (B, C, T) or (B,1,C,T)
        if eeg_np.ndim == 4:
            B, _, C, T = eeg_np.shape
            eeg_np = eeg_np.reshape(B, C, T)
        else:
            B, C, T = eeg_np.shape

        nominal_freq = nominal_freq.detach().cpu().numpy()
        adj_freq = []

        for i in range(B):

            f0 = float(nominal_freq[i])

            # filter band
            low = max(0.1, f0 - self.freq_width)
            high = f0 + self.freq_width
            nyq = sfreq / 2

            if high >= nyq:
                high = nyq - 0.1

            # channel average
            sig = eeg_np[i].mean(axis=0)

            # bandpass filtering
            try:
                sos = sg.butter(4, [low/nyq, high/nyq], btype='bandpass', output='sos')
                filtered = sg.sosfilt(sos, sig)
            except:
                adj_freq.append(f0)
                continue

            # SNR threshold check
            if np.max(np.abs(filtered)) < self.min_amp_threshold:
                adj_freq.append(f0)
                continue

            analytic = sg.hilbert(filtered)
            inst_phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(inst_phase) * sfreq / (2*np.pi)

            # smoothing
            if len(inst_freq) > self.smooth_window:
                kernel = np.ones(self.smooth_window) / self.smooth_window
                inst_freq = sg.convolve(inst_freq, kernel, mode="same")

            f_hat = np.median(inst_freq)
            if np.isnan(f_hat) or f_hat <= 0:
                f_hat = f0

            adj_freq.append(float(f_hat))

            if self.debug and i == 0:
                print(f"[StimEstimator] {f0:.3f} Hz -> {f_hat:.3f} Hz")

        return torch.tensor(adj_freq, dtype=torch.float32, device=eeg.device)