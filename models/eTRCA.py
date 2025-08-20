# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict, List, Sequence
from scipy.linalg import eigh


class Classifier(object):
    def __init__(self, freq_list: Sequence[float]) -> None:
        self.freq_list = freq_list
        self.trained_model: Dict[float, Dict[str, np.ndarray]] = {}

        self._filter_freqs: List[float] = []
        self._w: np.ndarray | None = None
        self._templates: Dict[float, np.ndarray] = {}
        self._templates_proj: np.ndarray | None = None
        self._freq_to_index: Dict[float, int] = {}

    @staticmethod
    def _remove_dc(x_tc):
        return x_tc - x_tc.mean(axis=0, keepdims=True)

    @staticmethod
    def _train_trca_single_freq(eeg_data_ntc):
        _, _, n_channels = eeg_data_ntc.shape

        x_ntc = eeg_data_ntc - eeg_data_ntc.mean(axis=1, keepdims=True)

        q = np.einsum("ntc,ntd->cd", x_ntc, x_ntc)

        a_tc = x_ntc.sum(axis=0)
        s = a_tc.T @ a_tc - q

        eps = 1e-8
        q = q + eps * np.eye(n_channels)
        _, vecs = eigh(s, q)
        w = vecs[:, -1]
        return w

    def fit(self, x, y):
        self.trained_model.clear()
        self._filter_freqs = []
        self._w = None
        self._templates.clear()
        self._templates_proj = None
        self._freq_to_index.clear()

        for freq in self.freq_list:
            freq_mask = (y == freq)
            if not np.any(freq_mask):
                continue

            freq_trials = x[freq_mask]
            freq_ntc = np.transpose(freq_trials, (0, 2, 1))

            w_f = self._train_trca_single_freq(freq_ntc)

            template_tc = freq_ntc.mean(axis=0)
            template_tc = self._remove_dc(template_tc)

            self.trained_model[freq] = {
                "w": w_f,
                "template_tc": template_tc,
            }
            self._filter_freqs.append(freq)
            self._templates[freq] = template_tc

        w_cols = [self.trained_model[f]["w"] for f in self._filter_freqs]
        self._w = np.stack(w_cols, axis=1)

        trained_freqs = [f for f in self.freq_list if f in self._templates]
        self._freq_to_index = {f: i for i, f in enumerate(trained_freqs)}

        k_filters = self._w.shape[1]
        t_len = next(iter(self._templates.values())).shape[0]
        self._templates_proj = np.zeros((len(trained_freqs), k_filters, t_len),
                                        dtype=float)
        for f, idx in self._freq_to_index.items():
            y_tc = self._templates[f]
            y_proj_tk = y_tc @ self._w
            self._templates_proj[idx] = y_proj_tk.T
        return self

    @staticmethod
    def _corr_1d(a, b):
        a_mean = a - a.mean()
        b_mean = b - b.mean()
        a_std = a_mean.std(ddof=1)
        b_std = b_mean.std(ddof=1)
        if a_std <= 1e-12 or b_std <= 1e-12:
            return 0.0
        return float((a_mean @ b_mean) / ((len(a) - 1) * a_std * b_std))

    def _predict_single_trial(self, eeg_trial):
        if self._w is None or self._templates_proj is None:
            raise RuntimeError("Model is not fitted.")

        x_tc = self._remove_dc(eeg_trial.T)

        x_proj_kt = (x_tc @ self._w).T

        scores = []
        for f in self.freq_list:
            if f not in self._freq_to_index:
                scores.append(float("-inf"))
                continue

            idx = self._freq_to_index[f]
            y_proj_kt = self._templates_proj[idx]

            r_sum = 0.0
            for k in range(x_proj_kt.shape[0]):
                r_sum += self._corr_1d(x_proj_kt[k], y_proj_kt[k])
            scores.append(r_sum)

        best_idx = int(np.argmax(scores))
        return self.freq_list[best_idx]

    def predict(self, x):
        predictions = [self._predict_single_trial(trial) for trial in x]
        return np.array(predictions)

if __name__ == "__main__":
    pass