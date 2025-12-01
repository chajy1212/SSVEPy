import torch
import torch.nn as nn
import torch.nn.functional as F

class StimulusAutoCorrector(nn.Module):
    """
    Self-supervised Δf correction module
      - Δf는 EEG 특징과 Stimulus Template 특징의 cosine similarity를 최대화하는 방향으로 학습됨
      - Ground truth effective frequency가 없어도 학습 가능
      - eeg_feat_global, stim_feat dimension 자동 인식하여 projection layer 생성됨
    """
    def __init__(self, eeg_channels, hidden_dim=64, eeg_feat_dim=None, stim_feat_dim=None):
        super().__init__()

        # Small CNN encoder for Δf regression
        self.eeg_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=(eeg_channels, 1), stride=1),
            nn.ReLU(),
        )

        # Δf regressor
        self.regressor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # projection heads (initialized later)
        self.proj_eeg = None
        self.proj_stim = None

        self.eeg_feat_dim = eeg_feat_dim
        self.stim_feat_dim = stim_feat_dim


    # ---------------------------------------------------------
    # Build projection heads dynamically AFTER seeing data
    # ---------------------------------------------------------
    def build_projection(self, eeg_feat_dim, stim_feat_dim, proj_dim=128):
        self.proj_eeg = nn.Linear(eeg_feat_dim, proj_dim).to(next(self.parameters()).device)
        self.proj_stim = nn.Linear(stim_feat_dim, proj_dim).to(next(self.parameters()).device)


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

        # CNN encoder
        feat = self.eeg_encoder(eeg)  # (B, 32, 1, 1)
        feat = feat.mean(dim=[2, 3])  # (B, 32)

        # concat nominal freq
        nominal = nominal_freq.unsqueeze(1)
        x = torch.cat([feat, nominal], dim=1)

        # Δf regression
        delta_f = self.regressor(x).squeeze(1)

        corrected_freq = nominal_freq + delta_f

        return corrected_freq, delta_f

    # =====================================================================
    # Self-supervised cosine loss
    # =====================================================================
    def compute_ssl_loss(self, eeg_feat, stim_feat):
        """
        eeg_feat: EEGBranch
        stim_feat: StimulusBranch(corrected_freq)
        Loss: minimize -cosine_similarity → maximize alignment
        """
        # lazy initialization
        if self.proj_eeg is None:
            self.build_projection(eeg_feat.shape[1], stim_feat.shape[1])

        # Projection
        eeg_z = self.proj_eeg(eeg_feat)
        stim_z = self.proj_stim(stim_feat)

        cos_sim = F.cosine_similarity(eeg_z, stim_z, dim=1)
        loss = -cos_sim.mean()

        return loss