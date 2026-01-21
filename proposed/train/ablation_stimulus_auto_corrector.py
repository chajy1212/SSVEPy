# -*- coding:utf-8 -*-
import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import Lee2019Dataset
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


# ===== Train / Eval =====
def train_one_epoch(eeg_branch, stim_branch, temp_branch, dual_attn, corrector,
                    dataloader, optimizer, ce_criterion, device, dataset_freqs, args):
    # Trains the model for one epoch including the Auto-Corrector logic.
    eeg_branch.train()
    stim_branch.train()
    temp_branch.train()
    dual_attn.train()
    corrector.train()

    all_preds, all_labels = [], []
    total_loss_val = 0.0

    # Prepare frequency tensor
    if not isinstance(dataset_freqs, torch.Tensor):
        freqs_tensor = torch.tensor(dataset_freqs, dtype=torch.float32).to(device)
    else:
        freqs_tensor = dataset_freqs.to(device)

    for eeg, label in dataloader:
        eeg, label = eeg.to(device), label.to(device)
        B = eeg.size(0)

        optimizer.zero_grad()

        # Get Nominal Frequency (based on ground truth label)
        nominal_freqs = freqs_tensor[label]  # (B,)

        # Frequency Correction (Auto Correction)
        # Input EEG and Nominal Freq -> Predict corrected Freq and Delta_f
        current_freqs, delta_f = corrector(eeg, nominal_freqs)

        # Feature Extraction
        eeg_feat = eeg_branch(eeg, return_sequence=True)    # (B, T, D)
        stim_feat = stim_branch(current_freqs)              # (B, D) <-- Input corrected frequency
        temp_feat = temp_branch(eeg, label)                 # (B, D)

        # Dual Attention & Classification Loss
        logits, _, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)
        cls_loss = ce_criterion(logits, label)

        # SSL Loss (Alignment)
        # Encourage EEG features and corrected Stimulus features to be similar (Cosine Sim)
        ssl_loss = corrector.compute_ssl_loss(eeg_feat, stim_feat)

        # Total Loss
        loss = cls_loss + (args.lambda_ssl * ssl_loss)

        loss.backward()
        optimizer.step()

        total_loss_val += loss.item() * B

        _, pred = logits.max(1)
        all_preds.append(pred.detach().cpu())
        all_labels.append(label.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss_val / len(all_labels)
    acc = (all_preds == all_labels).float().mean().item()

    return avg_loss, acc


@torch.no_grad()
def evaluate(eeg_branch, stim_branch, temp_branch, dual_attn, corrector,
             dataloader, ce_criterion, device, n_classes, trial_time, dataset_freqs):
    # Evaluates the model with Auto-Correction during Pattern Matching.
    eeg_branch.eval()
    stim_branch.eval()
    temp_branch.eval()
    dual_attn.eval()
    corrector.eval()

    all_preds, all_labels = [], []
    total_loss_val = 0.0
    total_samples = 0

    candidate_indices = torch.arange(n_classes).to(device)

    if not isinstance(dataset_freqs, torch.Tensor):
        candidate_freqs = torch.tensor(dataset_freqs, dtype=torch.float32).to(device)
    else:
        candidate_freqs = dataset_freqs.to(device)

    for eeg, label in dataloader:
        eeg, label = eeg.to(device), label.to(device)
        B = eeg.size(0)
        total_samples += B

        # EEG Feature
        eeg_feat = eeg_branch(eeg, return_sequence=True)

        batch_scores = []

        # Pattern Matching Loop (Iterate over all candidate frequencies)
        for cls_idx, freq_val in zip(candidate_indices, candidate_freqs):
            # (A) Prepare candidate frequency
            f_batch = freq_val.view(1).expand(B)

            # Perform correction during evaluation as well
            # Predict "How much would the frequency drift if this EEG belongs to class k?"
            corrected_f_batch, _ = corrector(eeg, f_batch)

            # Generate Stimulus Feature using corrected frequency
            stim_feat = stim_branch(corrected_f_batch)

            # (B) Template Feature
            cls_batch = cls_idx.view(1).expand(B)
            temp_feat = temp_branch(eeg, cls_batch)

            # (C) Dual Attention
            logits, _, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)

            # Extract probability (score) for the current class index
            score = logits[:, cls_idx]
            batch_scores.append(score.unsqueeze(1))

        # Select the class with the highest score
        batch_scores = torch.cat(batch_scores, dim=1)
        preds = batch_scores.argmax(dim=1)

        loss = ce_criterion(batch_scores, label)
        total_loss_val += loss.item() * B

        all_preds.append(preds.cpu())
        all_labels.append(label.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss_val / total_samples
    acc = (all_preds == all_labels).float().mean().item()
    itr = compute_itr(acc, n_classes, trial_time)

    return avg_loss, acc, itr


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

    all_accs, all_itrs = [], []

    for subj in subjects:
        print(f"\n========== [Subject {subj:02d}] ==========")

        train_dataset = Lee2019Dataset(subjects=[subj], train=True, pick_channels=args.pick_channels)
        test_dataset = Lee2019Dataset(subjects=[subj], train=False, pick_channels=args.pick_channels)

        n_channels = train_dataset.C
        n_samples = train_dataset.T
        n_classes = train_dataset.n_classes
        sfreq = train_dataset.sfreq
        trial_time = n_samples / sfreq
        freqs = list(getattr(train_dataset, "freqs"))

        print(f"[INFO] Train/Test samples: {len(train_dataset)}/{len(test_dataset)}")
        print(f"[INFO] Input shape: (C={n_channels}, T={n_samples}), Classes={n_classes}, Trial={trial_time}s")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
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
                                     D_temp=args.d_query).to(device)
        dual_attn = DualAttention(d_eeg=eeg_branch.feature_dim,
                                  d_query=args.d_query,
                                  d_model=args.d_model,
                                  num_heads=4,
                                  proj_dim=n_classes).to(device)
        corrector = StimulusAutoCorrector(eeg_channels=n_channels,
                                          hidden_dim=64).to(device)

        print_total_model_size(eeg_branch, stim_branch, temp_branch, dual_attn, corrector)

        params = list(eeg_branch.parameters()) + list(stim_branch.parameters()) + \
                 list(temp_branch.parameters()) + list(dual_attn.parameters()) + \
                 list(corrector.parameters())

        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        ce_criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_acc, best_itr, best_epoch = 0.0, 0.0, 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                eeg_branch, stim_branch, temp_branch, dual_attn, corrector,
                train_loader, optimizer, ce_criterion, device,
                dataset_freqs=freqs, args=args
            )
            test_loss, test_acc, itr = evaluate(
                eeg_branch, stim_branch, temp_branch, dual_attn, corrector,
                test_loader, ce_criterion, device,
                n_classes=n_classes, trial_time=trial_time,
                dataset_freqs=freqs
            )

            scheduler.step()

            print(f"\n[Epoch {epoch:03d}] "
                  f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} || "
                  f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f} | "
                  f"ITR: {itr:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                best_itr = itr
                best_epoch = epoch

                save_dir = "/home/brainlab/Workspace/jycha/SSVEP/model_path"
                save_path = os.path.join(save_dir, f"StimAutoCorrLee2019_Sub{subj}_EEGNet_{ch_tag}.pth")

                state_dict = {
                    "epoch": best_epoch,
                    "best_acc": best_acc,
                    "best_itr": best_itr,
                    "eeg_branch": eeg_branch.state_dict(),
                    "stim_branch": stim_branch.state_dict(),
                    "temp_branch": temp_branch.state_dict(),
                    "dual_attn": dual_attn.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "corrector": corrector.state_dict()
                }

                torch.save(state_dict, save_path)

                print(f"\n[Save] Epoch {best_epoch} → Best model "
                      f"(Acc={best_acc:.5f}, ITR={best_itr:.4f}) saved to {save_path}")

        all_accs.append(best_acc)
        all_itrs.append(best_itr)

    print(f"\n[Final Summary - Forced Correction]")
    print(f"Mean Acc: {np.mean(all_accs):.5} ± {np.std(all_accs):.5f}")
    print(f"Mean ITR: {np.mean(all_itrs):.4f} ± {np.std(all_itrs):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, default="all", help=" '1,2,3', 'all' ")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--d_query", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lambda_ssl", type=float, default=0.1, help="Weight for SSL Cosine Similarity Loss")
    parser.add_argument("--pick_channels", type=str, default="P3,P4,P7,P8,Pz,PO9,PO10,O1,O2,Oz", help=" 'all' ")

    args = parser.parse_args()

    # Channel parsing
    if isinstance(args.pick_channels, str):
        if args.pick_channels.lower() == "all":
            args.pick_channels = "all"
        else:
            cleaned = args.pick_channels.strip("[]")
            args.pick_channels = [ch.strip().strip("'").strip('"') for ch in cleaned.split(",")]

    main(args)