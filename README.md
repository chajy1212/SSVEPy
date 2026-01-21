# SSVEPy
Steady State Visually Evoked Potential (SSVEP) Library 

## 📂 Repository Structure
```
SSVEPy/
│
├── proposed/
│   │
│   ├── module/
│   │   ├── branches.py                                     # EEG / Stimulus / Template branch Encoder
│   │   ├── data_loader.py                                  # Data Loader
│   │   ├── dtn.py                                          # Dynamic Template Network
│   │   ├── dual_attention.py                               # Dual Attention — EEG ↔ Stimulus/Template
│   │   ├── eegnet.py                                       # EEG feature encoder
│   │   ├── stimulus.py                                     # Stimulus feature encoder
│   │   └── stimulus_auto_corrector.py                      # Stimulus Auto Corrector
│   │
│   ├── results/
│   │   ├── time_window.py                                  # Evaluates performance across different time windows (Session split)
│   │   ├── time_window_loso.py                             # Evaluates performance across different time windows (LOSO)
│   │   ├── visualize_time_window.py                        # Plots Accuracy and ITR trends over varying signal lengths
│   │   └── visualize_umap.py                               # Visualizes latent feature distributions using UMAP
│   │
│   └── train/
│       ├── ablation_stimulus_auto_corrector.py             # Lee2019 dataset — Ablation Study add Stimulus Auto Corrector (Session split)
│       ├── ablation_stimulus_auto_corrector_loso.py        # Lee2019 dataset — Ablation Study add Stimulus Auto Corrector (LOSO)
│       ├── ablation_wo_dual.py                             # Lee2019 dataset — Ablation Study w/o Dual Branch
│       ├── ablation_wo_stim.py                             # Lee2019 dataset — Ablation Study w/o Stimulus Branch
│       ├── ablation_wo_temp.py                             # Lee2019 dataset — Ablation Study w/o Template Branch
│       ├── loso_lee.py                                     # Lee2019 dataset — LOSO
│       ├── loso_nakanishi.py                               # Nakanishi2015 dataset — LOSO
│       └── train_lee.py                                    # Lee2019 dataset — Session split
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
