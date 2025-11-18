import numpy as np
import scipy.signal as sg
import torch

class StimulusAutoEstimator:
    """
    Frequency-only estimator
    Stable across all datasets: Nakanishi, Lee, BETA, AR
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
        Returns: adj_freq : (B,)
        """
        # GPU 텐서 → CPU numpy로 변환
        eeg_np = eeg.detach().cpu().numpy()

        # 입력 shape 통일: (B, C, T) or (B,1,C,T)
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

            # channel average → 여러 채널을 평균내면 노이즈가 줄고 SNR이 올라감
            sig = eeg_np[i].mean(axis=0)

            # nominal frequency 주변 ±freq_width (기본 0.7Hz)로 bandpass filtering → SSVEP의 pure sinusoid 성분만 남기는 과정
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

            # Hilbert 변환으로 instantaneous frequency 계산
            # 즉, 신호가 실제로 몇 Hz로 진동하는지 “샘플 단위로” 계산할 수 있음
            analytic = sg.hilbert(filtered)
            inst_phase = np.unwrap(np.angle(analytic))              # 신호의 순간 위상 (instantaneous phase)
            inst_freq = np.diff(inst_phase) * sfreq / (2*np.pi)     # 위상이 시간에 따라 얼마나 빨리 증가하는지 계산하면 → 순간 주파수=위상 증가 속도

            # smoothing
            # EEG는 noisy해서 순간주파수 값이 출렁거릴 수 있음 → moving average로 부드럽게 만듦
            if len(inst_freq) > self.smooth_window:
                kernel = np.ones(self.smooth_window) / self.smooth_window
                inst_freq = sg.convolve(inst_freq, kernel, mode="same")

            # robust representative value 선택 (median)
            # 평균이 아니라 median을 쓰는 이유: 1) 노이즈 spike에 영향을 받지 않음, 2) 안정적이고 robust한 중심값을 제공함
            f_hat = np.median(inst_freq)

            # 이상치 처리 → 만약 계산이 망하거나 불안정하면 그냥 nominal을 사용
            if np.isnan(f_hat) or f_hat <= 0:
                f_hat = f0

            adj_freq.append(float(f_hat))

            # 한 batch에서 첫 sample만 비교해서 디버깅 보여줌
            if self.debug and i == 0:
                print(f"[StimEstimator] {f0:.3f} Hz -> {f_hat:.3f} Hz")

        return torch.tensor(adj_freq, dtype=torch.float32, device=eeg.device)   # 모델 입력인 torch tensor로 형태 복귀