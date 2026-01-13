# -*- coding:utf-8 -*-
import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data_loader import TorchBETADataset
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
    """
    Compute Information Transfer Rate (ITR) in bits/min.
    acc: accuracy (0~1)
    n_classes: number of target classes
    trial_time: trial length in seconds
    """
    if acc <= 0 or n_classes <= 1:
        return 0.0

    acc = min(max(acc, eps), 1 - eps)  # avoid log(0) or log(negative)

    itr = (np.log2(n_classes) +
           acc * np.log2(acc) +
           (1 - acc) * np.log2((1 - acc) / (n_classes - 1)))
    itr = 60.0 / trial_time * itr
    return itr


# ===== Subject parser =====
def parse_subjects(subjects_arg, dataset_name=""):
    """
    subjects_arg: e.g. "1,2,3", "1-10", "1-5,7,9-12", "all"
    """
    if subjects_arg.lower() == "all":
        if dataset_name == "BETA":
            subjects = list(range(1, 71))  # 1 ~ 70
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
def train_one_epoch(eeg_branch, stim_branch, temp_branch, dual_attn,
                    dataloader, optimizer, ce_criterion, device):
    eeg_branch.train()
    stim_branch.train()
    temp_branch.train()
    dual_attn.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for eeg, label, freq, phase, _ in dataloader:
        eeg, label = eeg.to(device), label.to(device)
        freq, phase = freq.to(device), phase.to(device)

        optimizer.zero_grad()

        eeg_feat = eeg_branch(eeg, return_sequence=True)
        stim_feat = stim_branch(freq, phase)
        temp_feat = temp_branch(eeg, label)

        logits, _, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)

        loss = ce_criterion(logits, label) + dual_attn.loss_entropy
        loss.backward()
        optimizer.step()

        batch_size = label.size(0)
        total_loss += loss.item() * batch_size

        _, preds = logits.max(1)
        correct += (preds == label).sum().item()
        total += batch_size

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


@torch.no_grad()
def evaluate(eeg_branch, stim_branch, temp_branch, dual_attn,
             dataloader, ce_criterion, device, n_classes, trial_time,
             cand_freqs, cand_phases):
    eeg_branch.eval()
    stim_branch.eval()
    temp_branch.eval()
    dual_attn.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    # 후보군 텐서 (Pattern Matching용)
    c_freqs = torch.tensor(cand_freqs, dtype=torch.float32).to(device)
    c_phases = torch.tensor(cand_phases, dtype=torch.float32).to(device)
    c_indices = torch.arange(n_classes).to(device)

    for eeg, label, _, _, _ in dataloader:
        eeg, label = eeg.to(device), label.to(device)
        B = eeg.size(0)

        eeg_feat = eeg_branch(eeg, return_sequence=True)
        batch_scores = []

        # Pattern Matching
        for k in c_indices:
            f_val = c_freqs[k].view(1).expand(B)
            p_val = c_phases[k].view(1).expand(B)
            stim_feat = stim_branch(f_val, p_val)

            cls_batch = k.view(1).expand(B)
            temp_feat = temp_branch(eeg, cls_batch)

            logits, _, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)
            batch_scores.append(logits[:, k].unsqueeze(1))

        batch_scores = torch.cat(batch_scores, dim=1)
        loss = ce_criterion(batch_scores, label)
        total_loss += loss.item() * B

        all_preds.append(batch_scores.argmax(dim=1).cpu())
        all_labels.append(label.cpu())

    avg_loss = total_loss / len(all_labels)
    acc = (torch.cat(all_preds) == torch.cat(all_labels)).float().mean().item()
    itr = compute_itr(acc, n_classes, trial_time)

    return avg_loss, acc, itr


# ===== Main (4-Fold Block CV) =====
def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Channel tag
    if args.pick_channels == "all":
        ch_tag = "allch"
    else:
        ch_tag = "".join(args.pick_channels)

    all_accs, all_itrs = [], []

    subjects = parse_subjects(args.subjects, "BETA")

    print(f"========== BETA 4-Fold (Block) Cross-Validation ==========")
    print(f"Subjects: {len(subjects)} | Channels: {args.pick_channels}")

    for subj in subjects:
        print(f"\n[Subject {subj}] Loading Data...")

        # 1. Load Data (All Blocks)
        # stride=0.5 -> Data Augmentation Effect
        dataset = TorchBETADataset([subj], args.beta_data_root, args.pick_channels)

        if len(dataset) == 0: continue

        n_channels = dataset.C
        n_samples = dataset.T
        n_classes = dataset.n_classes
        sfreq = dataset.sfreq
        trial_time = n_samples / sfreq

        cand_freqs = dataset.stim_info['freqs']
        cand_phases = dataset.stim_info['phases']

        # Block IDs (BETA usually has 4 blocks: 0, 1, 2, 3)
        block_ids = np.unique(dataset.blocks)

        subj_accs, subj_itrs = [], []

        # 2. Block-wise CV Loop
        for fold, test_block in enumerate(block_ids):
            print(f"\n  --- Subject {subj:02d} | Fold {fold + 1}/{len(block_ids)} (Test Block {test_block}) ---")

            # Index Splitting
            indices = np.arange(len(dataset))
            train_idx = indices[dataset.blocks != test_block]
            test_idx = indices[dataset.blocks == test_block]

            print(f"[INFO] Train/Test samples: {len(train_idx)}/{len(test_idx)}")
            print(f"[INFO] Input shape: (C={n_channels}, T={n_samples}), Classes={n_classes}, Trial={trial_time}s")

            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

            eeg_branch = EEGBranch(chans=dataset.C,
                                   samples=dataset.T).to(device)
            stim_branch = StimulusBranchWithPhase(T=dataset.T,
                                                  sfreq=dataset.sfreq,
                                                  hidden_dim=args.d_query,
                                                  n_harmonics=5,
                                                  out_dim=args.d_query).to(device)
            temp_branch = TemplateBranch(n_bands=8,
                                         n_features=32,
                                         n_channels=dataset.C,
                                         n_samples=dataset.T,
                                         n_classes=n_classes,
                                         D_temp=args.d_query).to(device)
            dual_attn = DualAttention(d_eeg=eeg_branch.feature_dim,
                                      d_query=args.d_query,
                                      d_model=args.d_model,
                                      num_heads=4,
                                      proj_dim=n_classes).to(device)

            # 첫 번째 Subject의 첫 번째 Fold에서만 모델 사이즈 출력
            if subj == subjects[0] and test_block == block_ids[0]:
                print_total_model_size(eeg_branch, stim_branch, temp_branch, dual_attn)

            params = list(eeg_branch.parameters()) + list(stim_branch.parameters()) + \
                     list(temp_branch.parameters()) + list(dual_attn.parameters())

            optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            # best record
            best_acc, best_itr, best_epoch = 0.0, 0.0, 0

            # Train Loop
            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train_one_epoch(
                    eeg_branch, stim_branch, temp_branch, dual_attn,
                    train_loader, optimizer, criterion, device
                )
                test_loss, test_acc, itr = evaluate(
                    eeg_branch, stim_branch, temp_branch, dual_attn,
                    test_loader, criterion, device, n_classes, trial_time,
                    cand_freqs, cand_phases
                )

                scheduler.step()

                print(f"\n[Epoch {epoch:03d}] "
                      f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} || "
                      f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f} | "
                      f"ITR: {itr:.4f} bits/min")

                # update best record
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_itr = itr
                    best_epoch = epoch

                    # Save Model
                    save_dir = "/home/brainlab/Workspace/jycha/SSVEP/model_path"
                    save_path = os.path.join(save_dir, f"BETA_S{subj}_Block{test_block}_EEGNet_{ch_tag}.pth")

                    torch.save({
                        "epoch": best_epoch,
                        "best_acc": best_acc,
                        "best_itr": best_itr,
                        "eeg_branch": eeg_branch.state_dict(),
                        "stim_branch": stim_branch.state_dict(),
                        "temp_branch": temp_branch.state_dict(),
                        "dual_attn": dual_attn.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }, save_path)

            print(f"    > Fold {fold+1} Result: Acc={best_acc:.4f}, ITR={best_itr:.2f} (Ep {best_epoch})")

            subj_accs.append(best_acc)
            subj_itrs.append(best_itr)

        # Subject Summary
        mean_acc = np.mean(subj_accs)
        mean_itr = np.mean(subj_itrs)
        print(f"\n[Subject {subj} Final] Mean Acc: {mean_acc:.4f}, Mean ITR: {mean_itr:.2f}")

        all_accs.append(mean_acc)
        all_itrs.append(mean_itr)

    print(f"\n========== FINAL RESULT (Subjects: {len(subjects)}) ==========")
    print(f"Mean Acc: {np.mean(all_accs):.5} ± {np.std(all_accs):.5f}")
    print(f"Mean ITR: {np.mean(all_itrs):.4f} ± {np.std(all_itrs):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta_data_root", type=str, default="/home/brainlab/Workspace/jycha/SSVEP/data/BETA")
    parser.add_argument("--subjects", type=str, default="16-70", help="e.g. '16-50' or '16,17,18'")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--d_query", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--pick_channels", type=str, default="PZ,PO3,PO4,PO5,PO6,POZ,O1,O2,OZ", help=" 'all' ")
    args = parser.parse_args()

    # Parse channel selection
    if isinstance(args.pick_channels, str):
        if args.pick_channels.lower() == "all":
            args.pick_channels = "all"
        else:
            cleaned = args.pick_channels.strip("[]")
            args.pick_channels = [ch.strip().strip("'").strip('"') for ch in cleaned.split(",")]

    main(args)