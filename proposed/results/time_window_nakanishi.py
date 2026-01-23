# -*- coding:utf-8 -*-
import os
import torch
import random
import argparse
import numpy as np
from sympy.physics.units import momentum
from torch.utils.data import DataLoader

from data_loader import Lee2019Dataset_LOSO
from dual_attention import DualAttention
from branches import EEGBranch, StimulusBranch, TemplateBranch
from stimulus_auto_corrector import StimulusAutoCorrector


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
        if dataset_name == "Nakanishi2015":
            subjects = list(range(1, 11))  # 1 ~ 10
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


# ===== Time Window Evaluation Function =====
@torch.no_grad()
def evaluate_time_windows(eeg_branch, stim_branch, temp_branch, dual_attn, corrector,
                          dataloader, device, n_classes, sfreq, dataset_freqs,
                          time_windows=[0.5, 1.0, 2.0, 3.0, 4.0]):
    """
    LOSO Model Evaluation: Measures Accuracy and ITR for various time window lengths using the saved model.
    """
    eeg_branch.eval()
    stim_branch.eval()
    temp_branch.eval()
    dual_attn.eval()
    corrector.eval()

    results = {}
    candidate_indices = torch.arange(n_classes).to(device)

    if not isinstance(dataset_freqs, torch.Tensor):
        candidate_freqs = torch.tensor(dataset_freqs, dtype=torch.float32).to(device)
    else:
        candidate_freqs = dataset_freqs.to(device)

    # Check data length using a dummy batch
    dummy_eeg, _, _ = next(iter(dataloader))
    max_pts = dummy_eeg.shape[-1]

    print("\n  --- Time Window Analysis (LOSO Inference) ---")

    for tw in time_windows:
        crop_pts = int(tw * sfreq)

        # Skip if window is longer than available data
        if crop_pts > max_pts:
            print(f"  [Skip] Window {tw}s is longer than data")
            continue

        all_preds, all_labels = [], []

        for eeg, label, _ in dataloader:
            eeg, label = eeg.to(device), label.to(device)
            B = eeg.size(0)

            # Slicing: Crop data to the current time window
            eeg_cropped = eeg[..., :crop_pts]

            # Model Forward
            eeg_feat = eeg_branch(eeg_cropped, return_sequence=True)

            batch_scores = []

            # Pattern Matching Loop
            for cls_idx, freq_val in zip(candidate_indices, candidate_freqs):
                freq_batch = freq_val.view(1).expand(B)

                corrected_freq, _ = corrector(eeg_cropped, freq_batch)
                stim_feat = stim_branch(corrected_freq)

                cls_batch = cls_idx.view(1).expand(B)
                temp_feat = temp_branch(eeg_cropped, cls_batch)

                logits, _, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)
                batch_scores.append(logits[:, cls_idx].unsqueeze(1))

            batch_scores = torch.cat(batch_scores, dim=1)
            preds = batch_scores.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc = (all_preds == all_labels).float().mean().item()
        itr = compute_itr(acc, n_classes, tw)

        results[tw] = {'acc': acc, 'itr': itr}
        print(f"  Time: {tw:.1f}s | Acc: {acc * 100:.2f}% | ITR: {itr:.2f}")

    return results


# ===== Main =====
def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Channel tag
    if args.pick_channels == "all":
        ch_tag = "allch"
    else:
        ch_tag = "".join(args.pick_channels)

    subjects = parse_subjects(args.subjects, "Lee2019")

    # Dictionary to store aggregated results
    final_time_analysis = {tw: {'acc': [], 'itr': []} for tw in [0.5, 1.0, 2.0, 3.0, 4.0]}

    for test_subj in subjects:
        print(f"\n========== [LOSO Evaluation: Test Subject {test_subj:02d}] ==========")

        model_dir = "/home/brainlab/Workspace/jycha/SSVEP/model_path"
        model_path = os.path.join(model_dir, f"LOSOStimAutoCorrNakanishi2015_sub{test_subj}_EEGNet_{ch_tag}.pth")

        if not os.path.exists(model_path):
            print(f"[Warning] Model file not found: {model_path}. Skipping...")
            continue

        # Load Test Dataset
        test_dataset = Lee2019Dataset_LOSO(subjects=[test_subj], pick_channels=args.pick_channels)

        n_channels = test_dataset.C
        n_samples = test_dataset.T
        n_classes = test_dataset.n_classes
        sfreq = test_dataset.sfreq
        trial_time = n_samples / sfreq
        freqs = list(getattr(test_dataset, "freqs", np.linspace(8, 15, n_classes)))

        print(f"[INFO] Dataset: Lee2019")
        print(f"[INFO] Subjects used ({len(subjects)}): {subjects}")
        print(f"[INFO] Test samples: {len(test_dataset)}")
        print(f"[INFO] Channels used ({n_channels}): {', '.join(args.pick_channels)}")
        print(f"[INFO] Input shape: (C={n_channels}, T={n_samples}), Classes={n_classes}, Trial={trial_time}s")

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        eeg_branch = EEGBranch(chans=n_channels, samples=n_samples).to(device)
        stim_branch = StimulusBranch(T=n_samples,
                                     sfreq=sfreq,
                                     hidden_dim=args.d_query,
                                     n_harmonics=3).to(device)
        temp_branch = TemplateBranch(n_bands=8, n_features=32,
                                     n_channels=n_channels,
                                     n_samples=n_samples,
                                     n_classes=n_classes,
                                     D_temp=args.d_query,
                                     momentum=0.1).to(device)
        dual_attn = DualAttention(d_eeg=eeg_branch.feature_dim,
                                  d_query=args.d_query,
                                  d_model=args.d_model,
                                  num_heads=4,
                                  proj_dim=n_classes).to(device)
        corrector = StimulusAutoCorrector(eeg_channels=n_channels,
                                          hidden_dim=64).to(device)

        # Load Saved Weights
        print(f"[Info] Loading LOSO model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, weights_only=False)

            # Check dictionary structure
            if "model_state" in checkpoint:
                state = checkpoint["model_state"]
                eeg_branch.load_state_dict(state["eeg"])
                stim_branch.load_state_dict(state["stim"])
                temp_branch.load_state_dict(state["temp"])
                dual_attn.load_state_dict(state["attn"])
                if "corr" in state:
                    corrector.load_state_dict(state["corr"])
                elif "corrector" in state:
                    corrector.load_state_dict(state["corrector"])
            else:
                eeg_branch.load_state_dict(checkpoint["eeg_branch"])
                stim_branch.load_state_dict(checkpoint["stim_branch"])
                temp_branch.load_state_dict(checkpoint["temp_branch"])
                dual_attn.load_state_dict(checkpoint["dual_attn"])
                if "corrector" in checkpoint:
                    corrector.load_state_dict(checkpoint["corrector"])

            best_acc_log = checkpoint.get("best_acc", 0.0)
            print(f"[Info] Model loaded successfully (Train Best Acc: {best_acc_log:.4f})")

        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            continue

        # Execute Time Window Evaluation
        time_results = evaluate_time_windows(
            eeg_branch, stim_branch, temp_branch, dual_attn, corrector,
            test_loader, device, n_classes, sfreq, freqs,
            time_windows=[0.5, 1.0, 2.0, 3.0, 4.0]
        )

        # Aggregate Results
        for tw, metrics in time_results.items():
            if tw in final_time_analysis:
                final_time_analysis[tw]['acc'].append(metrics['acc'])
                final_time_analysis[tw]['itr'].append(metrics['itr'])

    # Print Final Results
    print(f"\n\n========== FINAL LOSO Time Window Analysis (Avg over {len(subjects)} subjects) ==========")
    print(f"{'Time (s)':<10} | {'Mean Acc (%)':<25} | {'Mean ITR (bits/min)':<25}")
    print("-" * 70)

    for tw in sorted(final_time_analysis.keys()):
        accs = final_time_analysis[tw]['acc']
        itrs = final_time_analysis[tw]['itr']

        if len(accs) > 0:
            mean_acc = np.mean(accs) * 100
            std_acc = np.std(accs) * 100

            mean_itr = np.mean(itrs)
            std_itr = np.std(itrs)

            print(f"{tw:<10.1f} | {mean_acc:.2f} ± {std_acc:.2f}{'':<8} | {mean_itr:.2f} ± {std_itr:.2f}")
        else:
            print(f"{tw:<10.1f} | N/A {'':<21} | N/A")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, default="all", help=" '1,2,3', '1-10', '1-5,7,9-12', 'all' ")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d_query", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--pick_channels", type=str, default="PO3,PO4,PO7,PO8,POz,O1,O2,Oz", help=" 'all' ")
    args = parser.parse_args()

    if isinstance(args.pick_channels, str):
        if args.pick_channels.lower() == "all":
            args.pick_channels = "all"
        else:
            cleaned = args.pick_channels.strip("[]")
            args.pick_channels = [ch.strip().strip("'").strip('"') for ch in cleaned.split(",")]

    main(args)