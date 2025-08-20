# -*- coding:utf-8 -*-
import numpy as np
from scipy.linalg import eigh


class Classifier(object):
    def __init__(self, freq_list):
        self.freq_list = freq_list
        self.trained_model = {}

    @staticmethod
    def _train_trca_single_freq(eeg_data):
        n_trials, n_samples, n_channels = eeg_data.shape

        x = eeg_data - eeg_data.mean(axis=1, keepdims=True)

        q = np.einsum('ntc,ntd->cd', x, x)

        a = x.sum(axis=0)
        s = a.T @ a - q

        eps = 1e-8
        q = q + eps * np.eye(n_channels)

        eigenvalues, eigenvectors = eigh(s, q)
        w = eigenvectors[:, -1]
        return w

    def fit(self, x, y):
        for freq in self.freq_list:
            freq_train_data = x[y == freq]

            freq_train_data_transposed = freq_train_data.transpose(0, 2, 1)
            w = self._train_trca_single_freq(freq_train_data_transposed)
            mean_freq_data = np.mean(freq_train_data_transposed, axis=0)
            template = mean_freq_data @ w

            self.trained_model[freq] = {'w': w, 'template': template}
        return self

    def _predict_single_trial(self, eeg_trial):
        correlations = []
        eeg_trial_transposed = eeg_trial.T

        for freq in self.freq_list:
            if freq not in self.trained_model:
                correlations.append(-1)
                continue

            w = self.trained_model[freq]['w']
            template = self.trained_model[freq]['template']

            filtered_test_data = eeg_trial_transposed @ w

            corr = np.corrcoef(filtered_test_data, template)[0, 1]
            correlations.append(corr)

        detected_freq_index = np.argmax(correlations)
        return self.freq_list[detected_freq_index]

    def predict(self, x):
        predictions = [self._predict_single_trial(trial) for trial in x]
        return np.array(predictions)


if __name__ == '__main__':
    pass