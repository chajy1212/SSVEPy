# -*- coding:utf-8 -*-
import numpy as np
from scipy.linalg import eigh


class Classifier(object):
    def __init__(self, freq_list):
        self.freq_list = freq_list
        self.trained_model = {}

    @staticmethod
    def _train_trca_single_freq(eeg_data):
        n_trials, n_channels, n_samples = eeg_data.shape

        x = eeg_data - eeg_data.mean(axis=2, keepdims=True)

        q = np.einsum('nct,ndt->cd', x, x)

        a = x.sum(axis=0)   # (C, T)
        s = a @ a.T - q     # (C, C)

        eps = 1e-8
        q = q + eps * np.eye(n_channels)

        eigenvalues, eigenvectors = eigh(s, q)
        w = eigenvectors[:, -1]      # (C,)
        return w

    def fit(self, x, y):
        """
            x: (n_trials, n_channels, n_samples)
            y: (n_trials,)
        """
        for freq in self.freq_list:
            freq_train_data = x[y == freq]                      # (N_f, C, T)
            w = self._train_trca_single_freq(freq_train_data)   # (C,)
            mean_freq_data = np.mean(freq_train_data, axis=0)   # (C, T)
            template = w @ mean_freq_data                       # (C,) @ (C,T) → (T,)
            self.trained_model[freq] = {'w': w, 'template': template}
        return self

    def _predict_single_trial(self, eeg_trial):
        """ eeg_trial: (n_channels, n_samples) """
        correlations = []

        for freq in self.freq_list:
            if freq not in self.trained_model:
                correlations.append(-1)
                continue

            w = self.trained_model[freq]['w']                   # (C,)
            template = self.trained_model[freq]['template']     # (T,)

            filtered_test_data = w @ eeg_trial                  # (C,) @ (C,T) → (T,)
            corr = np.corrcoef(filtered_test_data, template)[0, 1]
            correlations.append(corr)

        detected_freq_index = np.argmax(correlations)
        return self.freq_list[detected_freq_index]

    def predict(self, x):
        """ x: (n_trials, n_channels, n_samples) """
        predictions = [self._predict_single_trial(trial) for trial in x]
        return np.array(predictions)


if __name__ == "__main__":
    pass