import torch
import torch.nn as nn
import torch.nn.functional as F

class StimulusAutoCorrector(nn.Module):
    """
    Self-supervised Δf correction module.
    Δf는 EEG 특징과 Stimulus Template 특징의 cosine similarity를 최대화하는 방향으로 학습됨.
    Ground truth effective frequency가 없어도 학습 가능.
    """
    def __init__(self, eeg_channels=8, hidden_dim=64):
        super().__init__()

        # ---------------------------------------------------------
        # Small CNN-based EEG encoder
        # ---------------------------------------------------------
        self.eeg_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=(eeg_channels, 1), stride=1),
            nn.ReLU()
        )

        # Δf regressor
        self.regressor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.proj_eeg = nn.Linear(528, 128)  # EEGBranch output dim → 128
        self.proj_stim = nn.Linear(64, 128)  # StimulusBranch hidden_dim → 128

    # ---------------------------------------------------------
    # forward( ) : Δf + corrected_freq 계산 수행
    # ---------------------------------------------------------
    def forward(self, eeg, nominal_freq):
        """
        eeg: (B, 1, C, T)
        nominal_freq: (B,)
        """
        eeg = eeg.float()
        nominal_freq = nominal_freq.float()

        # 1) CNN EEG features
        feat = self.eeg_encoder(eeg)  # (B, 32, 1, 1)
        feat = feat.mean(dim=[2, 3])  # (B, 32)

        # 2) append nominal frequency
        nominal = nominal_freq.unsqueeze(1)  # (B, 1)
        x = torch.cat([feat, nominal], dim=1)  # (B, 33)

        # 3) Δf regression
        delta_f = self.regressor(x).squeeze(1)  # (B,)

        # 4) corrected frequency
        corrected_freq = nominal_freq + delta_f

        return corrected_freq, delta_f


    # =====================================================================
    # Self-supervised cosine-based loss
    # =====================================================================
    def compute_ssl_loss(self, eeg_feat, stim_feat):
        """
        eeg_feat: EEGBranch
        stim_feat: StimulusBranch(corrected_freq)
        Loss: minimize -cosine_similarity → maximize alignment
        """
        eeg_z = self.proj_eeg(eeg_feat)  # (B,128)
        stim_z = self.proj_stim(stim_feat)  # (B,128)

        cos_sim = F.cosine_similarity(eeg_z, stim_z, dim=1)

        loss = -cos_sim.mean()  # cosine similarity maximization
        return loss