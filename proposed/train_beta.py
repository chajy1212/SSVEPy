# -*- coding:utf-8 -*-
import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset

from dual_attention import DualAttention
from data_loader import BETADataset
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
def compute_itr(acc, n_classes, trial_time):
    """
    Compute Information Transfer Rate (ITR) in bits/min.
    acc: accuracy (0~1)
    n_classes: number of target classes
    trial_time: trial length in seconds
    """
    if acc <= 0 or n_classes <= 1:
        return 0.0
    itr = (np.log2(n_classes) +
           acc * np.log2(acc) +
           (1 - acc) * np.log2((1 - acc) / (n_classes - 1)))
    itr = 60.0 / trial_time * itr
    return itr


# ===== Subject parser =====
def parse_subjects(subjects_arg, dataset_name=""):
    """
    subjects_arg: e.g. "1,2,3", "1-10", "1-5,7,9-12", "all"
    dataset_name: "AR", "Nakanishi2015", "Lee2019"
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
                    dataloader, optimizer, ce_criterion, device,
                    accumulation_steps=1):
    eeg_branch.train()
    stim_branch.train()
    temp_branch.train()
    dual_attn.train()

    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch_idx, (eeg, label, freq, phase, _) in enumerate(dataloader):
        eeg, label = eeg.to(device), label.to(device)
        freq, phase = freq.to(device), phase.to(device)

        optimizer.zero_grad()

        eeg_feat = eeg_branch(eeg)
        stim_feat = stim_branch(freq, phase)
        temp_feat = temp_branch(eeg, label)
        logits, pooled, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)

        loss = ce_criterion(logits, label) / accumulation_steps
        loss.backward()

        batch_size = label.size(0)
        total_loss += loss.item() * batch_size * accumulation_steps

        _, pred = logits.max(1)
        all_preds.append(pred.detach().cpu())
        all_labels.append(label.detach().cpu())

        # Update weights after accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss / len(all_labels)
    acc = (all_preds == all_labels).float().mean().item()

    return avg_loss, acc


@torch.no_grad()
def evaluate(eeg_branch, stim_branch, temp_branch, dual_attn,
             dataloader, ce_criterion, device,
             n_classes=None, trial_time=None):
    eeg_branch.eval()
    stim_branch.eval()
    temp_branch.eval()
    dual_attn.eval()

    all_preds, all_labels = [], []
    total_loss = 0.0

    for eeg, label, freq, phase, _ in dataloader:
        eeg, label = eeg.to(device), label.to(device)
        freq, phase = freq.to(device), phase.to(device)

        eeg_feat = eeg_branch(eeg)
        stim_feat = stim_branch(freq, phase)
        temp_feat = temp_branch(eeg, label)
        logits, pooled, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)

        loss = ce_criterion(logits, label)
        total_loss += loss.item() * label.size(0)

        _, pred = logits.max(1)
        all_preds.append(pred.cpu())
        all_labels.append(label.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss / len(all_labels)
    acc = (all_preds == all_labels).float().mean().item()

    itr = None
    if n_classes is not None and trial_time is not None:
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

    writer = SummaryWriter(log_dir=f"/home/brainlab/Workspace/jycha/SSVEP/runs/{args.dataset}_{args.subjects}_{args.encoder}_{ch_tag}")

    if args.dataset == "BETA":
        subjects = parse_subjects(args.subjects, "BETA")

        all_accs, all_itrs = [], []
        for test_subj in subjects:

            # train / test split (Leave-One-Subject-Out)
            train_dataset = ConcatDataset([
                BETADataset(
                    npz_file=os.path.join(args.beta_data_root, f"S{s}.npz"),
                    pick_channels=args.pick_channels,
                )
                for s in subjects if s != test_subj
            ])
            test_dataset = BETADataset(
                npz_file=os.path.join(args.beta_data_root, f"S{test_subj}.npz"),
                pick_channels=args.pick_channels,
            )

            n_channels = train_dataset.datasets[0].C
            n_samples = train_dataset.datasets[0].T
            n_classes = train_dataset.datasets[0].n_classes
            ch_names = train_dataset.datasets[0].ch_names
            sfreq = train_dataset.datasets[0].sfreq
            trial_time = n_samples / sfreq  # 250 / 250 = 1.0s
            # freq = list(train_dataset.datasets[0].freq)
            # phase = list(train_dataset.datasets[0].phase)

            print(f"\n[INFO] LOOCV Test Subject: {test_subj}")
            print(f"[INFO] Dataset: {args.dataset}")
            print(f"[INFO] Subjects: {args.subjects}")
            print(f"[INFO] Train/Test samples: {len(train_dataset)}/{len(test_dataset)}")
            print(f"[INFO] Channels used ({n_channels}): {', '.join(ch_names)}")
            print(f"[INFO] Input shape: (C={n_channels}, T={n_samples}), Classes={n_classes}, Trial={trial_time:.2f}s, Sampling Rate={sfreq}Hz\n")

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            eeg_branch = EEGBranch(chans=n_channels, samples=n_samples).to(device)

            stim_branch = StimulusBranchWithPhase(T=n_samples,
                                                  sfreq=sfreq,
                                                  hidden_dim=args.d_query,
                                                  n_harmonics=3).to(device)
            temp_branch = TemplateBranch(n_bands=8, n_features=32,
                                         n_channels=n_channels,
                                         n_samples=n_samples,
                                         n_classes=n_classes,
                                         D_temp=args.d_query).to(device)
            dual_attn = DualAttention(d_eeg=eeg_branch.out_dim,
                                      d_query=args.d_query,
                                      d_model=args.d_model,
                                      num_heads=4,
                                      proj_dim=n_classes).to(device)

            print_total_model_size(eeg_branch, stim_branch, temp_branch, dual_attn)

            params = list(eeg_branch.parameters()) + list(stim_branch.parameters()) + \
                     list(temp_branch.parameters()) + list(dual_attn.parameters())

            optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
            ce_criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            # Train Loop
            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train_one_epoch(
                    eeg_branch, stim_branch, temp_branch, dual_attn,
                    train_loader, optimizer, ce_criterion, device,
                    accumulation_steps=args.accumulation_steps
                )
                test_loss, test_acc, itr = evaluate(
                    eeg_branch, stim_branch, temp_branch, dual_attn,
                    test_loader, ce_criterion, device,
                    n_classes=n_classes, trial_time=trial_time
                )
                scheduler.step()

                # TensorBoard logging
                writer.add_scalar(f"Subject{test_subj}/Train/Loss", train_loss, epoch)
                writer.add_scalar(f"Subject{test_subj}/Train/Acc", train_acc, epoch)
                writer.add_scalar(f"Subject{test_subj}/Test/Loss", test_loss, epoch)
                writer.add_scalar(f"Subject{test_subj}/Test/Acc", test_acc, epoch)
                if itr is not None:
                    writer.add_scalar(f"Subject{test_subj}/Test/ITR", itr, epoch)

                print(f"\n[Epoch {epoch:03d}] "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
                      f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}", end="")
                if itr is not None:
                    print(f" | ITR: {itr:.2f} bits/min")
                else:
                    print()

            # Save model per subject
            save_dir = "/home/brainlab/Workspace/jycha/SSVEP/model_path"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{args.dataset}_{args.subjects}_{args.encoder}_{ch_tag}.pth")
            torch.save({
                "eeg_branch": eeg_branch.state_dict(),
                "stim_branch": stim_branch.state_dict(),
                "temp_branch": temp_branch.state_dict(),
                "dual_attn": dual_attn.state_dict(),
                "optimizer": optimizer.state_dict()
            }, save_path)
            print(f"[Save] Model saved to {save_path}")

            all_accs.append(test_acc)
            if itr is not None:
                all_itrs.append(itr)

        print("\n========== Final LOOCV Result ==========")
        print(f"Mean Acc: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")
        if len(all_itrs) > 0:
            print(f"Mean ITR: {np.mean(all_itrs):.2f} ± {np.std(all_itrs):.2f}")

    else:
        raise ValueError("Now only BETA + LOOCV mode is supported in this script.")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BETA", choices=["BETA"])
    parser.add_argument("--beta_data_root", type=str, default="/home/brainlab/Workspace/jycha/SSVEP/processed_beta")
    parser.add_argument("--subjects", type=str, default="16-50", help=" '1,2,3', '1-10', 'all' ")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d_query", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--pick_channels", type=str, default="PZ,PO3,PO4,PO5,PO6,POZ,O1,O2,OZ",
                        help=" 'O1,O2,Oz', 'all' ")
    parser.add_argument("--encoder", type=str, default="EEGNet")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients before optimizer step")
    args = parser.parse_args()

    if isinstance(args.pick_channels, str):
        if args.pick_channels.lower() == "all":
            args.pick_channels = "all"
        else:
            cleaned = args.pick_channels.strip("[]")
            args.pick_channels = [ch.strip().strip("'").strip('"') for ch in cleaned.split(",")]

    main(args)