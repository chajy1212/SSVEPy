# -*- coding:utf-8 -*-
import numpy as np
from sklearn import cross_decomposition
from typing import Sequence



class Classifier(object):
    def __init__(self,
                 freq_list: Sequence[float],
                 fs: float,
                 second: float,
                 n_harmonic: int = 5):

        self.freq_list = freq_list
        self.fs = fs
        self.second = second
        self.n_samples = int(fs * second)
        self.n_harmonic = n_harmonic

        self.t = np.linspace(0, second, self.n_samples, endpoint=False)
        self.cca = cross_decomposition.CCA(n_components=1)
        self.reference_signals = self._generate_reference_signals()

    def _generate_reference_signals(self):
        reference_signals = {}
        for freq in self.freq_list:
            comps = []
            for h in range(1, self.n_harmonic + 1):
                comps.append(np.sin(2 * np.pi * h * freq * self.t))
                comps.append(np.cos(2 * np.pi * h * freq * self.t))
            reference_signals[freq] = np.stack(comps, axis=1)  # (T, 2H)
        return reference_signals

    def _predict_single_trial(self, eeg_trial):
        correlations = []
        eeg_trial_transposed = eeg_trial.T

        for freq in self.freq_list:
            ref_signal = self.reference_signals[freq]
            x_c, y_c = self.cca.fit_transform(eeg_trial_transposed, ref_signal)
            corr = np.corrcoef(x_c.T, y_c.T)[0, 1]
            correlations.append(corr)

        detected_freq_index = np.argmax(correlations).astype(np.int32)
        return self.freq_list[detected_freq_index]

    def predict(self, x):
        predictions = [self._predict_single_trial(trial) for trial in x]
        return np.array(predictions)


if __name__ == '__main__':
    pass