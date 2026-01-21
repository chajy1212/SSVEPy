# -*- coding:utf-8 -*-
import os
import umap
import torch
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_loader import Lee2019Dataset_LOSO, Lee2019Dataset
from branches import EEGBranch


# ===== Reproducibility =====
def set_seed(seed=777):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_total_model_size(*models):
    total_params = sum(p.numel() for m in models for p in m.parameters())
    trainable_params = sum(p.numel() for m in models for p in m.parameters() if p.requires_grad)
    print(f"[Total Model Size]")
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {trainable_params:,}")
    print(f"  Memory Estimate      : {total_params * 4 / (1024 ** 2):.2f} MB\n")


# ===== ITR function =====
def compute_itr(acc, n_classes, trial_time, eps=1e-12):
    if acc <= 0 or n_classes <= 1:
        return 0.0
    acc = min(max(acc, eps), 1 - eps)
    itr = (np.log2(n_classes) +
           acc * np.log2(acc) +
           (1 - acc) * np.log2((1 - acc) / (n_classes - 1)))
    itr = 60.0 / trial_time * itr
    return itr


# ===== Subject parser =====
def parse_subjects(subjects_arg, dataset_name=""):
    if subjects_arg.lower() == "all":
        if dataset_name == "Lee2019":
            subjects = list(range(1, 55))  # 1 ~ 54
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        return subjects

    subjects = []
    for part in subjects_arg.split(","):
        part = part.strip()
        if part == "":
            continue
        if "-" in part:
            start, end = part.split("-")
            subjects.extend(range(int(start), int(end) + 1))
        else:
            subjects.append(int(part))
    return subjects


# ===== Feature Extraction Function =====
@torch.no_grad()
def extract_features(eeg_branch, dataloader, device):
    """
    Extracts latent features from the EEG Branch.
    """
    eeg_branch.eval()

    all_features = []
    all_labels = []

    for batch in dataloader:
        # Handle different return formats from DataLoader
        if len(batch) == 3:
            eeg, label, _ = batch
        else:
            eeg, label = batch

        eeg = eeg.to(device)

        # Pass through EEG Branch to extract latent features
        # If return_sequence=True, shape is (Batch, Time, Feat)
        features = eeg_branch(eeg, return_sequence=True)

        # Global Average Pooling: (Batch, Seq, Dim) -> (Batch, Dim)
        # Summarize the sequence into a single vector by averaging over time
        if len(features.shape) == 3:
            features = features.mean(dim=1)

        all_features.append(features.cpu().numpy())
        all_labels.append(label.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


# ===== Main =====
def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Channel tag
    if args.pick_channels == "all":
        ch_tag = "allch"
    else:
        ch_tag = "".join(args.pick_channels)

    # Select target subject for visualization
    target_subj = args.target_subject
    print(f"========== UMAP Visualization for Subject {target_subj} ==========")

    # Path to saved model weights
    model_dir = "/home/brainlab/Workspace/jycha/SSVEP/model_path"
    model_path = os.path.join(model_dir, f"LOSOLee2019_sub{target_subj}_EEGNet_{ch_tag}.pth")

    # Check model existence and fallback logic
    if not os.path.exists(model_path):
        print(f"[Error] Model not found: {model_path}")
        model_name_alt = f"Lee2019_Sub{target_subj}_EEGNet_{ch_tag}.pth"
        model_path_alt = os.path.join(model_dir, model_name_alt)
        if os.path.exists(model_path_alt):
            model_path = model_path_alt
            print(f"[INFO] Found alternative model: {model_path}")
        else:
            return

    # Load Dataset
    # LOSO:
    test_dataset = Lee2019Dataset_LOSO(subjects=[target_subj], pick_channels=args.pick_channels)
    # Session Split:
    # test_dataset = Lee2019Dataset(subjects=[target_subj], pick_channels=args.pick_channels)

    n_channels = test_dataset.C
    n_samples = test_dataset.T
    n_classes = test_dataset.n_classes
    sfreq = test_dataset.sfreq
    trial_time = n_samples / sfreq

    print(f"[INFO] Dataset: Lee2019")
    print(f"[INFO] Test samples: {len(test_dataset)}")
    print(f"[INFO] Channels used ({n_channels}): {', '.join(args.pick_channels)}")
    print(f"[INFO] Input shape: (C={n_channels}, T={n_samples}), Classes={n_classes}, Trial={trial_time}s")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"[INFO] Dataset loaded. Samples: {len(test_dataset)}")

    # Initialize Model
    eeg_branch = EEGBranch(chans=n_channels, samples=n_samples).to(device)

    print(f"[INFO] Loading model weights...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load state dict based on checkpoint structure
    if "model_state" in checkpoint:
        eeg_branch.load_state_dict(checkpoint["model_state"]["eeg"])
    elif "eeg_branch" in checkpoint:
        eeg_branch.load_state_dict(checkpoint["eeg_branch"])
    else:
        print("[Error] Unknown checkpoint format.")
        return

    # Extract Features
    print("[INFO] Extracting features...")
    features, labels = extract_features(eeg_branch, test_loader, device)
    print(f"[INFO] Features shape: {features.shape}")

    # UMAP
    print("[INFO] Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=20,
                        min_dist=0.1,
                        n_components=2,
                        random_state=42,
                        metric='cosine')
    embedding = reducer.fit_transform(features)

    # Plot Results
    print("[INFO] Plotting...")
    plt.figure(figsize=(10, 8))

    # Create DataFrame for Seaborn
    df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    df['Class'] = labels.astype(str)
    df.sort_values(by='Class', inplace=True)

    # Scatter Plot
    sns.scatterplot(
        data=df,
        x='UMAP1',
        y='UMAP2',
        hue='Class',
        palette='viridis',
        s=80,
        alpha=0.8,
        edgecolor='white',
        linewidth=0.5
    )

    plt.title(f'Learned Feature Space (Subject {target_subj})', fontsize=16, pad=15)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend(
        title='Class',
        loc='best',
        frameon=True,
        fontsize=12,
        title_fontsize=13,
    )

    plt.tight_layout()

    save_dir = "/home/brainlab/Workspace/jycha/SSVEP/result"
    save_path = os.path.join(save_dir, f"UMAP_Subject{target_subj}.eps")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Done] Plot saved to {save_path}")

    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Suggestion: For LOSO models use subject 43, for Session-Split models use subject 31 as default examples
    parser.add_argument("--target_subject", type=int, default=43, help="Subject ID to visualize")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pick_channels", type=str, default="P3,P4,P7,P8,Pz,PO9,PO10,O1,O2,Oz", help=" 'all' ")
    args = parser.parse_args()

    if isinstance(args.pick_channels, str):
        if args.pick_channels.lower() == "all":
            args.pick_channels = "all"
        else:
            cleaned = args.pick_channels.strip("[]")
            args.pick_channels = [ch.strip().strip("'").strip('"') for ch in cleaned.split(",")]

    main(args)