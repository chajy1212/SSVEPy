# -*- coding:utf-8 -*-
import numpy as np
from typing import Sequence
from scipy.linalg import eigh, qr

class Classifier(object):
    def __init__(self,
                 freq_list: Sequence[float],
                 fs: float,
                 second: float,
                 n_harmonic: int,
                 l: int,
                 n_components: int):

        self.freq_list = freq_list
        self.fs = fs
        self.second = second
        self.n_samples = int(fs * second)
        self.n_harmonic = n_harmonic
        self.l = l
        self.n_components = n_components

        self.t = np.arange(1, self.n_samples+1) / self.fs

        self.projection_matrices  = {}
        self.w = None
        self.templates = {}

    def _time_delay_augment(self, trial):
        """ trial: (C, T) → X̃: ((l+1)C, T) """
        n_channels, n_samples = trial.shape
        augment = [trial]
        for d in range(1, self.l + 1):
            delayed = np.zeros_like(trial)
            delayed[:, d:] = trial[:, :-d]
            augment.append(delayed)
        return np.vstack(augment)   # ((l+1)C , T)

    def _build_projection_matrix(self, freq):
        sin = [np.sin(2 * np.pi * (h+1) * freq * self.t) for h in range(self.n_harmonic)]
        cos = [np.cos(2 * np.pi * (h+1) * freq * self.t) for h in range(self.n_harmonic)]
        y = np.vstack(sin + cos)    # (2*Nh, T)

        q, _ = qr(y.T, mode="economic")

        return q @ q.T

    def _make_Xa(self, trial, freq):
        """
            Xa = [X̃, X̃p]
            trial: (C,T) → Xa: ((l+1)C, 2T)
        """
        x_augmented = self._time_delay_augment(trial)         # ((l+1)C, T)
        p = self.projection_matrices[freq]
        xp = x_augmented @ p
        xa = np.concatenate([x_augmented, xp], axis=1)  # ((l+1)C, 2T)

        return xa

    def fit(self, trials, labels):
        """
            trials : (n_trials, n_channels, n_samples)
            labels : (n_trials,)
        """
        n_trials, n_channels, n_samples = trials.shape

        for f in self.freq_list:
            self.projection_matrices[f] = self._build_projection_matrix(f)

        xa_trials = [self._make_Xa(trials[i], labels[i]) for i in range(n_trials)]

        class_means = {}
        for f in self.freq_list:
            idx = np.where(labels == f)[0]
            class_means[f] = np.mean([xa_trials[i] for i in idx], axis=0)

        global_mean = np.mean(xa_trials, axis=0)

        hb_blocks = [class_means[f] - global_mean for f in self.freq_list]
        hb = np.concatenate(hb_blocks, axis=1) / np.sqrt(len(self.freq_list))

        hw_blocks = [xa_trials[i] - class_means[labels[i]] for i in range(n_trials)]
        hw = np.concatenate(hw_blocks, axis=1) / np.sqrt(n_trials)

        sb = hb @ hb.T
        sw = hw @ hw.T + 1e-8 * np.eye(hw.shape[0])

        eigenvals, eigenvecs = eigh(sb, sw)
        self.w = eigenvecs[:, -self.n_components:]

        for f in self.freq_list:
            idx = np.where(labels == f)[0]
            projected_trials = [self.w.T @ xa_trials[i] for i in idx]
            self.templates[f] = np.mean(projected_trials, axis=0)  # (n_components, 2T)

        return self

    def _predict_single_trial(self, trial):
        scores = {}
        for f in self.freq_list:
            xa = self._make_Xa(trial, f)
            z = self.w.T @ xa                       # (n_components, 2T)
            template = self.templates[f]            # (n_components, 2T)

            zf, tf = z.ravel(), template.ravel()    # (n_components * 2T,)
            zf = (zf - zf.mean()) / (zf.std() + 1e-8)
            tf = (tf - tf.mean()) / (tf.std() + 1e-8)
            scores[f] = (zf @ tf) / len(zf)

        return max(scores, key=scores.get)

    def predict(self, trials):
        predictions = [self._predict_single_trial(trial) for trial in trials]
        return np.array(predictions)


if __name__ == "__main__":
    pass