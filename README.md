# SSVEPy
Steady State Visually Evoked Potential (SSVEP) Library 

## 📂 Repository Structure
```
SSVEPy/
│
├── proposed/
│   │
│   ├── module/
│   │   ├── DTN.py                                          # Dynamic Template Network
│   │   ├── EEGNet.py                                       # EEG feature encoder
│   │   ├── branches.py                                     # EEG / Stimulus / Template branch Encoder
│   │   ├── data_loader.py                                  # Data Loader
│   │   ├── dual_attention.py                               # Dual Attention — EEG ↔ Stimulus/Template
│   │   ├── stimulus.py                                     # StimulusEncoder
│   │   └── preprocess_AR_occi.py                           # AR dataset raw EEG → npz (Occipital channels)
│   │
│   ├── train/
│   │   ├── kfold_beta.py                                   # BETA dataset — KFold CV on 4 blocks
│   │   ├── kfold_wang.py                                   # Wang2016 dataset — KFold CV on 6 blocks
│   │   ├── loso_lee.py                                     # Lee2019 dataset — LOSO
│   │   ├── loso_nakanishi.py                               # Nakanishi2015 dataset — LOSO
│   │   ├── train_ar.py                                     # AR dataset — session split
│   │   └── train_lee.py                                    # Lee2019 dataset — session split
│   │
│   └── results/
│       ├── ...                                             # ...
│       └── ...                                             # ...
│
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
