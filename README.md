# SSVEPy
Steady State Visually Evoked Potential (SSVEP) Library 

## 📂 Repository Structure
```
SSVEPy/
│
├── proposed/
│   ├── core/
│   │   ├── EEGNet.py                               # EEGNet 기반 EEG feature encoder
│   │   ├── DTN.py                                  # Dynamic Template Network
│   │   ├── dual_attention.py                       # Dual Attention 모듈 — EEG ↔ Stimulus/Template
│   │   ├── simple_attention.py                     # 단일 attention baseline
│   │   └── stimulus.py                             # StimulusEncoder — sin/cos reference 신호 latent feature 인코딩
│   │
│   ├── modules/
│   │   ├── branches.py                             # EEG / Stimulus / Template branch Encoder 정의 및 결합
│   │   └── data_loader.py                          # EEG 데이터셋 로더
│   │
│   ├── training/
│   │   ├── train_ar.py                             # AR dataset — session split
│   │   ├── train_lee.py                            # Lee2019 dataset — session split
│   │   ├── train_nakanishi.py                      # Nakanishi2015 dataset — random split
│   │   ├── train_beta.py                           # BETA dataset — LOSO
│   │   ├── loso_ar.py                              # AR dataset — LOSO
│   │   ├── loso_lee.py                             # Lee2019 dataset — LOSO
│   │   └── loso_nakanishi.py                       # Nakanishi2015 dataset — LOSO
│   │
│   ├── ablation/
│   │   ├── ablation_eegnet_dtn.py                  # EEGNet + DTN
│   │   ├── ablation_eegnet_dtn_stim_dual.py        # EEGNet + Stimulus + DTN + Dual Attention
│   │   ├── ablation_eegnet_dtn_stim_concat.py      # EEGNet + DTN + Stimulus (Concat Two Attentions)
│   │   ├── ablation_eegnet_dtn_stim_element.py     # EEGNet + DTN + Stimulus (Element-wise Two Attentions)
│   │   ├── ablation_eegnet_stim.py                 # EEGNet + Stimulus 구조 실험
│   │   └── ablation_only_eegnet.py                 # EEGNet 단독 baseline
│   │
│   └── preprocess/
│       ├── preprocess_AR.py                        # AR dataset raw EEG → .npz 변환 (전체 채널)
│       └── preprocess_AR_occi.py                   # AR dataset raw EEG → .npz 변환 (후두부 채널만)
│ 
├── model/
│   ├── FBCCA.py
│   ├── TDCA.py
│   ├── TRCA.py
│   ├── eTRCA.py
│   └── sCCA.py
│
└── README.md
```
