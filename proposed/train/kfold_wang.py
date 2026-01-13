# -*- coding:utf-8 -*-
import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data_loader import Wang2016Dataset
from dual_attention import DualAttention
from branches import EEGBranch, StimulusBranchWithPhase, TemplateBranch


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
        if dataset_name == "Wang2016":
            subjects = list(range(1, 36))  # 1 ~ 35
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


# ===== Train Function =====
def train_one_epoch(eeg_branch, stim_branch, temp_branch, dual_attn,
                    dataloader, optimizer, ce_criterion, device, freqs_tensor, phases_tensor):
    eeg_branch.train()
    stim_branch.train()
    temp_branch.train()
    dual_attn.train()

    total_loss = 0.0
    all_preds, all_labels = [], []

    for eeg, label in dataloader:
        eeg, label = eeg.to(device), label.to(device)
        optimizer.zero_grad()

        # Feature Extraction
        eeg_feat = eeg_branch(eeg, return_sequence=True)

        # Stimulus Feature (JFPM: Freq + Phase)
        # Retrieve frequency and phase corresponding to the label
        current_freqs = freqs_tensor[label]
        current_phases = phases_tensor[label]
        stim_feat = stim_branch(current_freqs, current_phases)

        # Template Feature
        temp_feat = temp_branch(eeg, label)

        # Dual Attention
        logits, _, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)

        # Loss
        loss = ce_criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        _, pred = logits.max(1)
        all_preds.append(pred.detach().cpu())
        all_labels.append(label.detach().cpu())

    acc = (torch.cat(all_preds) == torch.cat(all_labels)).float().mean().item()
    return total_loss / len(all_labels), acc


# ===== Evaluate Function =====
@torch.no_grad()
def evaluate(eeg_branch, stim_branch, temp_branch, dual_attn,
             dataloader, ce_criterion, device, n_classes, trial_time, freqs_tensor, phases_tensor):
    eeg_branch.eval()
    stim_branch.eval()
    temp_branch.eval()
    dual_attn.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []
    candidate_indices = torch.arange(n_classes).to(device)

    for eeg, label in dataloader:
        eeg, label = eeg.to(device), label.to(device)
        B = eeg.size(0)

        # EEG Feature
        eeg_feat = eeg_branch(eeg, return_sequence=True)
        batch_scores = []

        # Pattern Matching Loop
        for k in candidate_indices:
            # (A) Stimulus: Candidate Freq & Phase
            f_val = freqs_tensor[k].view(1).expand(B)
            p_val = phases_tensor[k].view(1).expand(B)
            stim_feat = stim_branch(f_val, p_val)

            # (B) Template: Candidate Class Index
            cls_batch = k.view(1).expand(B)
            temp_feat = temp_branch(eeg, cls_batch)

            # (C) Dual Attention
            logits, _, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)

            # Store score for class k
            batch_scores.append(logits[:, k].unsqueeze(1))

        # Select class with highest score
        batch_scores = torch.cat(batch_scores, dim=1)
        loss = ce_criterion(batch_scores, label)
        total_loss += loss.item() * B

        all_preds.append(batch_scores.argmax(dim=1).cpu())
        all_labels.append(label.cpu())

    acc = (torch.cat(all_preds) == torch.cat(all_labels)).float().mean().item()
    itr = compute_itr(acc, n_classes, trial_time)
    return total_loss / len(all_labels), acc, itr


# ===== Main (6-Fold Block CV) =====
def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Channel Tag
    if args.pick_channels == "all":
        ch_tag = "allch"
    elif args.pick_channels == "occipital":
        ch_tag = "occipital"
    else:
        ch_tag = "custom"

    subjects = list(range(1, 36))  # 1 ~ 35
    final_accs, final_itrs = [], []

    print(f"========== Wang2016 Benchmark (Subject-Specific 6-Fold CV) ==========")
    print(f"Channels: {args.pick_channels}")

    for subj in subjects:
        print(f"\n[Subject {subj:02d}] Loading Data...")

        # Load Full Dataset for the Subject
        full_ds = Wang2016Dataset(subjects=[subj], data_root=args.wang_data_root, pick_channels=args.pick_channels)

        if full_ds.N == 0:
            print(f"[Warning] No data found for Subject {subj}. Skipping...")
            continue

        n_channels = full_ds.C
        n_samples = full_ds.T
        n_classes = full_ds.n_classes
        sfreq = full_ds.sfreq
        trial_time = n_samples / sfreq

        # Prepare Frequency/Phase Tensors
        freqs_t = torch.tensor(full_ds.freqs, dtype=torch.float32).to(device)
        phases_t = torch.tensor(full_ds.phases, dtype=torch.float32).to(device)

        subj_fold_accs = []
        subj_itrs = []

        # 6-Fold Cross-Validation (Block-based)
        # Wang2016 has 6 blocks. We use 5 for training and 1 for testing.
        for fold in range(6):
            print(f"\n  --- Subject {subj:02d} | Fold {fold + 1}/6 (Block {fold}) ---")

            # Data Split: Wang2016 is ordered by Block then Class (40 classes per block)
            indices = np.arange(len(full_ds))
            test_block_mask = (indices // 40) == fold

            test_idx = indices[test_block_mask]
            train_idx = indices[~test_block_mask]

            print(f"[INFO] Train/Test samples: {len(train_idx)}/{len(test_idx)}")
            print(f"[INFO] Input shape: (C={n_channels}, T={n_samples}), Classes={n_classes}, Trial={trial_time}s")

            train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=args.batch_size, shuffle=False)

            eeg_branch = EEGBranch(chans=n_channels, samples=n_samples).to(device)
            stim_branch = StimulusBranchWithPhase(T=n_samples,
                                                  sfreq=sfreq,
                                                  hidden_dim=args.d_query,
                                                  n_harmonics=5,
                                                  out_dim=args.d_query).to(device)
            temp_branch = TemplateBranch(n_bands=8,
                                         n_features=32,
                                         n_channels=n_channels,
                                         n_samples=n_samples,
                                         n_classes=n_classes,
                                         D_temp=args.d_query).to(device)
            dual_attn = DualAttention(d_eeg=eeg_branch.feature_dim,
                                      d_query=args.d_query,
                                      d_model=args.d_model,
                                      num_heads=4,
                                      proj_dim=n_classes).to(device)

            if subj == 1 and fold == 0:
                print_total_model_size(eeg_branch, stim_branch, temp_branch, dual_attn)

            params = list(eeg_branch.parameters()) + list(stim_branch.parameters()) + \
                     list(temp_branch.parameters()) + list(dual_attn.parameters())

            optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            best_fold_acc = 0.0
            best_fold_itr = 0.0
            best_epoch = 0

            # Training Loop
            for ep in range(1, args.epochs + 1):
                t_loss, t_acc = train_one_epoch(
                    eeg_branch, stim_branch, temp_branch, dual_attn,
                    train_loader, optimizer, criterion, device, freqs_t, phases_t
                )
                v_loss, v_acc, itr = evaluate(
                    eeg_branch, stim_branch, temp_branch, dual_attn,
                    test_loader, criterion, device, n_classes, trial_time, freqs_t, phases_t
                )
                scheduler.step()

                if v_acc > best_fold_acc:
                    best_fold_acc = v_acc
                    best_fold_itr = itr
                    best_epoch = ep
                    best_mark = "(*)"

                    save_dir = "/home/brainlab/Workspace/jycha/SSVEP/model_path"
                    save_path = os.path.join(save_dir, f"Wang2016_S{subj}_Fold{fold}_EEGNet_{ch_tag}.pth")
                    torch.save({
                        "epoch": best_epoch,
                        "best_acc": best_fold_acc,
                        "best_itr": best_fold_itr,
                        "eeg_branch": eeg_branch.state_dict(),
                        "stim_branch": stim_branch.state_dict(),
                        "temp_branch": temp_branch.state_dict(),
                        "dual_attn": dual_attn.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }, save_path)
                else:
                    best_mark = ""

                print(f"[Epoch {ep:03d}] "
                      f"Train Loss: {t_loss:.5f} | Train Acc: {t_acc:.5f} || "
                      f"Test Loss: {v_loss:.5f} | Test Acc: {v_acc:.5f} | "
                      f"ITR: {itr:.4f} {best_mark}")

            print(f"  > Fold {fold + 1} Best Result: Acc {best_fold_acc:.4f} (Ep {best_epoch})")
            subj_fold_accs.append(best_fold_acc)
            subj_itrs.append(best_fold_itr)

        # Subject Average
        mean_acc = np.mean(subj_fold_accs)
        mean_itr = np.mean(subj_itrs)
        print(f"[Subject {subj} Avg] Acc: {mean_acc:.4f}, ITR: {mean_itr:.2f}")

        final_accs.append(mean_acc)
        final_itrs.append(mean_itr)

    print(f"\n========== FINAL RESULT ==========")
    print(f"Mean Acc: {np.mean(final_accs):.4f} ± {np.std(final_accs):.4f}")
    print(f"Mean ITR: {np.mean(final_itrs):.2f} ± {np.std(final_itrs):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wang_data_root", type=str, default="/home/brainlab/Workspace/jycha/SSVEP/data/Wang")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--d_query", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--pick_channels", type=str, default="occipital", help="'occipital' or 'all'")
    args = parser.parse_args()

    # Custom Channel Parsing
    if args.pick_channels not in ["all", "occipital"]:
        cleaned = args.pick_channels.strip("[]")
        args.pick_channels = [ch.strip().strip("'").strip('"') for ch in cleaned.split(",")]

    main(args)