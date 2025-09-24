# SSVEPy
Steady State Visually Evoked Potential (SSVEP) Library 

## 📂 Repository Structure
```
SSVEPy/
│
├── proposed/
│   ├── DTN.py     			        # Dynamic Template Network (DTN)
│   ├── EEGNet.py               	# EEGNet 기반 EEG 인코더
│   ├── branches.py             	# EEG, Stimulus, Template branch encoders
│   ├── data_loader.py          	# Dataset 정의
│   ├── dual_attention.py       	# Dual Attention 모듈, EEG→Key/Value, Stimulus/Template→Query 결합
│   ├── preprocess_AR.py        	# AR Dataset Raw EEG → .npz 변환 유틸
│   ├── stimulus.py             	# StimulusEncoder, 자극 reference 신호(sin/cos) latent feature 추출
│   └── train.py                	# 전체 학습 파이프라인
│
├── model/
│   ├── FBCCA.py
│   ├── TDCA.py
│   ├── TRCA.py
│   ├── eTRCA.py
│   └── sCCA.py
│
└── README.md                   	# Project documentation
```
