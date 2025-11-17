# SSVEPy
Steady State Visually Evoked Potential (SSVEP) Library 

## 📂 Repository Structure
```
SSVEPy/
│
├── proposed/
│   ├── core/
│   │   ├── DTN.py                                          # Dynamic Template Network
│   │   ├── EEGNet.py                                       # EEGNet 기반 EEG feature encoder
│   │   ├── dual_attention.py                               # Dual Attention — EEG ↔ Stimulus/Template
│   │   ├── simple_attention.py                             # 단일 attention baseline
│   │   ├── stimulus.py                                     # StimulusEncoder — sin/cos reference 신호 latent feature 인코딩
│   │   └── stimulus_auto_estimator.py                      # 자극 주파수 자동 보정
│   │
│   ├── module/
│   │   ├── branches.py                                     # EEG / Stimulus / Template branch Encoder 정의 및 결합
│   │   └── data_loader.py                                  # EEG 데이터셋 로더
│   │
│   ├── train/
│   │   ├── train_ar.py                                     # AR dataset — session split
│   │   ├── train_lee.py                                    # Lee2019 dataset — session split
│   │   ├── train_nakanishi.py                              # Nakanishi2015 dataset — random split
│   │   ├── loso_ar.py                                      # AR dataset — LOSO
│   │   ├── loso_beta.py                                    # BETA dataset — LOSO
│   │   ├── loso_lee.py                                     # Lee2019 dataset — LOSO
│   │   ├── loso_nakanishi.py                               # Nakanishi2015 dataset — LOSO
│   │   ├── exp_ar.py                                       # AR dataset — Auto-Estimated session split
│   │   ├── exp_lee.py                                      # Lee2019 dataset — Auto-Estimated session split
│   │   ├── exp_loso_ar.py                                  # AR dataset — Auto-Estimated LOSO
│   │   ├── exp_loso_beta.py                                # BETA dataset — Auto-Estimated LOSO
│   │   ├── exp_loso_lee.py                                 # Lee2019 dataset — Auto-Estimated LOSO
│   │   └── exp_loso_nakanishi.py                           # Nakanishi2015 dataset — Auto-Estimated LOSO
│   │
│   ├── ablation/
│   │   ├── session_split/
│   │   │   ├── ablation_eegnet_dtn.py                      # EEGNet + DTN
│   │   │   ├── ablation_eegnet_dtn_stim_concat.py          # EEGNet + DTN + Stimulus (Concat Two Attentions)
│   │   │   ├── ablation_eegnet_dtn_stim_dual.py            # EEGNet + Stimulus + DTN + Dual Attention
│   │   │   ├── ablation_eegnet_dtn_stim_element.py         # EEGNet + DTN + Stimulus (Element-wise Two Attentions)
│   │   │   ├── ablation_eegnet_stim.py                     # EEGNet + Stimulus 구조 실험
│   │   │   └── ablation_only_eegnet.py                     # EEGNet 단독 baseline
│   │   │
│   │   └── LOSO/
│   │       ├── loso_ablation_eegnet_dtn.py                 # EEGNet + DTN
│   │       ├── loso_ablation_eegnet_dtn_stim_concat.py     # EEGNet + DTN + Stimulus (Concat Two Attentions)
│   │       ├── loso_ablation_eegnet_dtn_stim_dual.py       # EEGNet + Stimulus + DTN + Dual Attention
│   │       ├── loso_ablation_eegnet_dtn_stim_element.py    # EEGNet + DTN + Stimulus (Element-wise Two Attentions)
│   │       ├── loso_ablation_eegnet_only.py                # EEGNet 단독 baseline
│   │       └── loso_ablation_eegnet_stim.py                # EEGNet + Stimulus 구조 실험
│   │
│   └── preprocess/
│       ├── preprocess_AR.py                                # AR dataset raw EEG → .npz 변환 (전체 채널)
│       └── preprocess_AR_occi.py                           # AR dataset raw EEG → .npz 변환 (후두부 채널만)
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
