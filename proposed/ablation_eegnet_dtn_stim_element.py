# -*- coding:utf-8 -*-
import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import Lee2019Dataset
from branches import EEGBranch, StimulusBranch, TemplateBranch
from simple_attention import SimpleAttention_EEG_Stimulus, SimpleAttention_EEG_Template


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
def train_one_epoch(eeg_branch, stim_branch, temp_branch,
                    attn_eeg_stim, attn_eeg_temp, fusion_head,
                    dataloader, optimizer, ce_criterion, device):
    eeg_branch.train()
    stim_branch.train()
    temp_branch.train()
    attn_eeg_stim.train()
    attn_eeg_temp.train()
    fusion_head.train()

    all_preds, all_labels = [], []
    total_loss = 0.0

    for eeg, label in dataloader:
        eeg, label = eeg.to(device), label.to(device)

        optimizer.zero_grad()

        # Forward
        eeg_feat = eeg_branch(eeg)                                   # (B, D_eeg)
        stim_feat = stim_branch(label)                               # (B, D_query)
        temp_feat = temp_branch(eeg, label)                          # (B, D_query)

        _, attn_s = attn_eeg_stim(eeg_feat, stim_feat)
        _, attn_t = attn_eeg_temp(eeg_feat, temp_feat)

        attn_s = F.layer_norm(attn_s, attn_s.shape[1:])
        attn_t = F.layer_norm(attn_t, attn_t.shape[1:])

        # fused = torch.cat([attn_s, attn_t], dim=-1)   # Concat
        fused = attn_s + attn_t                         # Element-wise sum
        logits = fusion_head(fused)

        # CE loss
        loss = ce_criterion(logits, label)
        loss.backward()
        optimizer.step()

        batch_size = label.size(0)
        total_loss += loss.item() * batch_size

        _, pred = logits.max(1)
        all_preds.append(pred.detach().cpu())
        all_labels.append(label.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss / len(all_labels)
    acc = (all_preds == all_labels).float().mean().item()

    return avg_loss, acc


@torch.no_grad()
def evaluate(eeg_branch, stim_branch, temp_branch,
             attn_eeg_stim, attn_eeg_temp, fusion_head,
             dataloader, ce_criterion, device, n_classes, trial_time):
    eeg_branch.eval()
    stim_branch.eval()
    temp_branch.eval()
    attn_eeg_stim.eval()
    attn_eeg_temp.eval()
    fusion_head.eval()

    all_preds, all_labels = [], []
    total_loss = 0.0

    for eeg, label in dataloader:
        eeg, label = eeg.to(device), label.to(device)

        eeg_feat = eeg_branch(eeg)
        stim_feat = stim_branch(label)
        temp_feat = temp_branch(eeg, label)

        _, attn_s = attn_eeg_stim(eeg_feat, stim_feat)
        _, attn_t = attn_eeg_temp(eeg_feat, temp_feat)

        attn_s = F.layer_norm(attn_s, attn_s.shape[1:])
        attn_t = F.layer_norm(attn_t, attn_t.shape[1:])

        # fused = torch.cat([attn_s, attn_t], dim=-1)   # Concat
        fused = attn_s + attn_t  # Element-wise sum
        logits = fusion_head(fused)

        loss = ce_criterion(logits, label)
        total_loss += loss.item() * label.size(0)

        _, pred = logits.max(1)
        all_preds.append(pred.cpu())
        all_labels.append(label.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss / len(all_labels)
    acc = (all_preds == all_labels).float().mean().item()

    # ITR
    itr = compute_itr(acc, n_classes, trial_time)

    return avg_loss, acc, itr


# ===== Main (subject-wise session split) =====
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

        # TensorBoard writer
        writer = SummaryWriter(log_dir=f"/home/brainlab/Workspace/jycha/SSVEP/ablation/session_split/eegnet_dtn_stim/runs/Lee2019_Sub{subj}_EEGNet_{ch_tag}")

        train_dataset = Lee2019Dataset(subjects=[subj], train=True, pick_channels=args.pick_channels)
        test_dataset = Lee2019Dataset(subjects=[subj], train=False, pick_channels=args.pick_channels)

        n_channels = train_dataset.C
        n_samples = train_dataset.T
        n_classes = train_dataset.n_classes
        sfreq = train_dataset.sfreq
        trial_time = n_samples / sfreq
        freqs = list(getattr(train_dataset, "freqs", np.linspace(8, 15, n_classes)))

        print(f"[INFO] Dataset: Lee2019")
        print(f"[INFO] Subjects used ({len(subjects)}): {subjects}")
        print(f"[INFO] Train/Test samples: {len(train_dataset)}/{len(test_dataset)}")
        print(f"[INFO] Channels used ({n_channels}): {', '.join(args.pick_channels)}")
        print(f"[INFO] Input shape: (C={n_channels}, T={n_samples}), Classes={n_classes}, Trial={trial_time:.2f}s, Sampling Rate={sfreq}Hz\n")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Model
        eeg_branch = EEGBranch(chans=n_channels, samples=n_samples).to(device)
        stim_branch = StimulusBranch(freqs=freqs,
                                     T=n_samples,
                                     sfreq=sfreq,
                                     hidden_dim=args.d_query,
                                     n_harmonics=3).to(device)
        temp_branch = TemplateBranch(n_bands=8, n_features=32,
                                     n_channels=n_channels,
                                     n_samples=n_samples,
                                     n_classes=n_classes,
                                     D_temp=args.d_query).to(device)

        attn_eeg_stim = SimpleAttention_EEG_Stimulus(
            d_eeg=eeg_branch.out_dim, d_query=args.d_query,
            d_model=args.d_model, n_classes=n_classes
        ).to(device)
        attn_eeg_temp = SimpleAttention_EEG_Template(
            d_eeg=eeg_branch.out_dim, d_query=args.d_query,
            d_model=args.d_model, n_classes=n_classes
        ).to(device)

        # Concat
        # fusion_head = nn.Sequential(
        #     nn.LayerNorm(2 * args.d_model),
        #     nn.Linear(2 * args.d_model, n_classes)
        # ).to(device)

        # Element-wise sum
        fusion_head = nn.Sequential(
            nn.LayerNorm(args.d_model),
            nn.Linear(args.d_model, n_classes)
        ).to(device)

        print_total_model_size(eeg_branch, stim_branch, temp_branch, attn_eeg_stim, attn_eeg_temp, fusion_head)

        params = list(eeg_branch.parameters()) + list(stim_branch.parameters()) + \
                 list(temp_branch.parameters()) + list(attn_eeg_stim.parameters()) + \
                 list(attn_eeg_temp.parameters()) + list(fusion_head.parameters())

        optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
        ce_criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # best record
        best_acc, best_itr, best_epoch = 0.0, 0.0, 0

        # Train Loop
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                eeg_branch, stim_branch, temp_branch,
                attn_eeg_stim, attn_eeg_temp, fusion_head,
                train_loader, optimizer, ce_criterion, device
            )
            test_loss, test_acc, itr = evaluate(
                eeg_branch, stim_branch, temp_branch,
                attn_eeg_stim, attn_eeg_temp, fusion_head,
                test_loader, ce_criterion, device,
                n_classes=n_classes, trial_time=trial_time
            )

            scheduler.step()

            print(f"\n[Epoch {epoch:03d}] "
                  f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} || "
                  f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f} | "
                  f"ITR: {itr:.4f} bits/min")

            # TensorBoard logging
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Test", test_acc, epoch)
            writer.add_scalar("ITR/Test", itr, epoch)

            # update best record
            if test_acc > best_acc:
                best_acc = test_acc
                best_itr = itr
                best_epoch = epoch

                # Save Model
                save_dir = "/home/brainlab/Workspace/jycha/SSVEP/ablation/session_split/eegnet_dtn_stim/model_path"
                save_path = os.path.join(save_dir, f"Lee2019_Sub{subj}_EEGNet_{ch_tag}.pth")

                torch.save({
                    "epoch": best_epoch,
                    "best_acc": best_acc,
                    "best_itr": best_itr,
                    "eeg_branch": eeg_branch.state_dict(),
                    "stim_branch": stim_branch.state_dict(),
                    "temp_branch": temp_branch.state_dict(),
                    "attn_eeg_stim": attn_eeg_stim.state_dict(),
                    "attn_eeg_temp": attn_eeg_temp.state_dict(),
                    "fusion_head": fusion_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, save_path)

                print(f"\n[Save] Epoch {best_epoch} → Best model "
                      f"(Acc={best_acc:.5f}, ITR={best_itr:.4f}) saved to {save_path}")

        writer.close()
        all_accs.append(best_acc)
        all_itrs.append(best_itr)

    # Summary
    print("\n[Final]")
    print(f"Mean Acc: {np.mean(all_accs):.5} ± {np.std(all_accs):.5f}")
    print(f"Mean ITR: {np.mean(all_itrs):.4f} ± {np.std(all_itrs):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, default="all", help=" '1,2,3', '1-10', '1-5,7,9-12', 'all' ")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d_query", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--pick_channels", type=str, default="P3,P4,P7,P8,Pz,PO9,PO10,O1,O2,Oz", help=" 'O1,O2,Oz', 'all' ")
    args = parser.parse_args()

    # Parse channel selection
    if isinstance(args.pick_channels, str):
        if args.pick_channels.lower() == "all":
            args.pick_channels = "all"
        else:
            cleaned = args.pick_channels.strip("[]")
            args.pick_channels = [ch.strip().strip("'").strip('"') for ch in cleaned.split(",")]

    main(args)