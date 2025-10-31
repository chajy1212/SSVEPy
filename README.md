# SSVEPy
Steady State Visually Evoked Potential (SSVEP) Library 

## 📂 Repository Structure
```
SSVEPy/
│
├── proposed/
│   ├── core/                                       			# 모델의 핵심 구성요소 (EEG, DTN, Attention, Stimulus Encoder)
│   │   ├── EEGNet.py                               		# EEGNet 기반 spatio-temporal EEG feature encoder
│   │   ├── DTN.py                                  		# Dynamic Template Network — class prototype 학습 및 latent feature 추출
│   │   ├── dual_attention.py                       		# Dual Attention 모듈 — EEG↔Stimulus/Template 간 상호주의 결합
│   │   ├── simple_attention.py                     		# 단일 attention baseline (Dual 대비 단순화된 비교용)
│   │   └── stimulus.py                             		# StimulusEncoder — sin/cos reference 신호 latent feature 인코딩
│   │
│   ├── modules/                                    		# 공통 모듈 (branch 구조 및 데이터 로딩)
│   │   ├── branches.py                             		# EEG / Stimulus / Template branch 인코더 정의 및 결합
│   │   └── data_loader.py                          		# EEG 데이터셋 로더 (train/val/test split 및 배치 구성)
│   │
│   ├── training/                                   			# 각 데이터셋별 학습 및 평가 파이프라인
│   │   ├── train_ar.py                             		# AR dataset — session split 학습 파이프라인
│   │   ├── train_lee.py                            		# Lee2019 dataset — LOSO 학습 파이프라인
│   │   ├── train_nakanishi.py                      		# Nakanishi2015 dataset — random split 학습 파이프라인
│   │   ├── train_beta.py                           		# BETA dataset — 전용 학습 파이프라인
│   │   ├── loso_ar.py                              		# AR dataset — Leave-One-Subject-Out 평가 스크립트
│   │   ├── loso_lee.py                             		# Lee2019 dataset — LOSO 평가 스크립트
│   │   └── loso_nakanishi.py                       		# Nakanishi2015 dataset — LOSO 평가 스크립트
│   │
│   ├── ablation/                                   			# 모델 구성 요소별 성능 비교 실험 (Ablation Studies)
│   │   ├── ablation_eegnet_dtn.py                  		# EEGNet + DTN 구조 비교 실험
│   │   ├── ablation_eegnet_dtn_stim_dual.py        	# EEGNet + Stimulus + DTN + Dual Attention (제안 구조)
│   │   ├── ablation_eegnet_dtn_stim_concat.py      	# EEGNet + DTN + Stimulus (Concat Attention)
│   │   ├── ablation_eegnet_dtn_stim_element.py     	# EEGNet + DTN + Stimulus (Element-wise Attention)
│   │   ├── ablation_eegnet_stim.py                 		# EEGNet + Stimulus 구조 실험 (DTN 제외)
│   │   └── ablation_only_eegnet.py                 		# EEGNet 단독 baseline 비교
│   │
│   ├── preprocess/                                 		# 데이터 전처리 및 .npz 변환 유틸리티
│   │   ├── preprocess_AR.py                        		# AR dataset raw EEG → .npz 변환 (전채널)
│   │   ├── preprocess_AR_occi.py                   		# AR dataset raw EEG → .npz 변환 (후두부 채널만)
│   │   └── (etc)                                   			# 추가 dataset preprocessing 스크립트
│   └── train_nakanishi.py               				# Nakanishi Dataset Random Split 학습 파이프라인
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
