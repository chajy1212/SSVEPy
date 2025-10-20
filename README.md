# SSVEPy
Steady State Visually Evoked Potential (SSVEP) Library 

## 📂 Repository Structure
```
SSVEPy/
│
├── proposed/
│   ├── DTN.py     				             # Dynamic Template Network (DTN)
│   ├── EEGNet.py               		     # EEGNet 기반 EEG 인코더
│   ├── ablation_eegnet_dtn.py               # LOSO Ablation Study (EEGNet + DTN)
│   ├── ablation_eegnet_stim.py              # LOSO Ablation Study (EEGNet + Stimulus)
│   ├── ablation_eegnet_dtn_stim_concat.py   # LOSO Ablation Study (EEGNet + DTN + Stimulus + Concat Two Attentions)
│   ├── ablation_eegnet_dtn_stim_element.py  # LOSO Ablation Study (EEGNet + DTN + Stimulus + Element-wise sum Two Attentions)
│   ├── ablation_full_model.py               # LOSO Ablation Study (EEGNet + DTN + Stimulus + Dual Attention)
│   ├── ablation_only_eegnet.py              # LOSO Ablation Study (EEGNet only)
│   ├── branches.py             		     # EEG, Stimulus, Template branch encoders
│   ├── data_loader.py          		     # Dataset 정의
│   ├── dual_attention.py       		     # Dual Attention 모듈, EEG→Key/Value, Stimulus/Template→Query 결합
│   ├── loso_ar.py       			         # AR Dataset LOSO 학습 파이프라인
│   ├── loso_lee.py       			         # Lee Dataset LOSO 학습 파이프라인
│   ├── loso_nakanishi.py       		     # Nakanishi Dataset LOSO 학습 파이프라인
│   ├── preprocess_AR.py        		     # AR Dataset Raw EEG → .npz 변환 유틸 (All Channel)
│   ├── preprocess_AR_occi.py  		         # AR Dataset Raw EEG → .npz 변환 유틸 (Occipital Channel)
│   ├── simple_attention.py                  # Simple Attention 모듈
│   ├── stimulus.py             		     # StimulusEncoder, 자극 reference 신호(sin/cos) latent feature 추출
│   ├── train_ar.py                		     # AR Dataset Session Split 학습 파이프라인
│   ├── train_beta.py               	     # BETA Dataset LOSO 학습 파이프라인
│   ├── train_lee.py                         # Lee Dataset Session Split 학습 파이프라인
│   └── train_nakanishi.py                   # Nakanishi Dataset Random Split 학습 파이프라인
│
├── model/
│   ├── FBCCA.py
│   ├── TDCA.py
│   ├── TRCA.py
│   ├── eTRCA.py
│   └── sCCA.py
│
└── README.md                   		     # Project documentation
```
