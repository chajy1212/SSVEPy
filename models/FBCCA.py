# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
from sklearn import cross_decomposition
from typing import Sequence


class Classifier(object):
    def __init__(self,
                 freq_list: Sequence[float],
                 fs: float,
                 second: float,
                 n_sub_bands: int = 5,
                 n_harmonic: int = 3,
                 high_cutoff_hz: float = 90.0,
                 filter_order: int = 5):

        self.freq_list = list(freq_list)
        self.fs = fs
        self.second = second
        self.n_sub_bands = n_sub_bands
        self.n_harmonic = n_harmonic
        self.n_samples = int(round(self.fs * self.second))
        self.t = np.linspace(0, self.second, self.n_samples, endpoint=False)

        self.high_cutoff_hz = float(high_cutoff_hz)
        self.filter_order = filter_order

        self.cca = cross_decomposition.CCA(n_components=1)
        self.weights = np.array(
            [(m + 1) ** (-1.25) + 0.25 for m in range(self.n_sub_bands)],
            dtype=np.float64,
        )

        self.filter_bank = self._design_filter_bank()
        self.reference_signals = self._generate_reference_signals()

    def _design_filter_bank(self):
        filter_bank = []
        nyquist = self.fs / 2.0
        high_cutoff = min(self.high_cutoff_hz, nyquist - 1.0)
        for m in range(1, self.n_sub_bands + 1):
            low_cutoff = max(1.0, 8.0 * m)
            b, a = signal.butter(
                self.filter_order, [low_cutoff, high_cutoff], btype="bandpass", fs=self.fs
            )
            filter_bank.append((b, a))
        return filter_bank

    def _generate_reference_signals(self):
        reference_signals = {}
        for freq in self.freq_list:
            comps = []
            for h in range(1, self.n_harmonic + 1):
                comps.append(np.sin(2 * np.pi * h * freq * self.t))
                comps.append(np.cos(2 * np.pi * h * freq * self.t))
            reference_signals[freq] = np.stack(comps, axis=1)
        return reference_signals

    def _score_for_freq(self, x_time_by_chan, freq):
        y_ref = self.reference_signals[freq]
        sub_corr_list = []

        for b, a in self.filter_bank:
            x_filter = signal.filtfilt(b, a, x_time_by_chan, axis=0)
            x_c, y_c = self.cca.fit_transform(x_filter, y_ref)
            r = np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1]
            sub_corr_list.append(r)

        sub_corr_list = np.asarray(sub_corr_list, dtype=np.float64)
        score = np.sum(self.weights * (sub_corr_list ** 2))
        return float(score)

    def _predict_single_trial(self, eeg_trial):
        x_tc = eeg_trial.T
        scores = [self._score_for_freq(x_tc, f) for f in self.freq_list]
        best_idx = int(np.argmax(scores))
        return self.freq_list[best_idx]

    def predict(self, x: np.ndarray):
        predictions = [self._predict_single_trial(trial) for trial in x]
        return np.array(predictions)


if __name__ == "__main__":
    pass