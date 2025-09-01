# -*- coding:utf-8 -*-
import os
import glob
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from data_loader import SSVEPDataset
from dual_attention import DualAttention
from branches import EEGBranch, StimulusBranch, TemplateBranch


# ===== Reproducibility =====
def set_seed(seed=777):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Size]")
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {trainable_params:,}")
    print(f"  Model Memory Estimate: {total_params * 4 / (1024**2):.2f} MB (float32)\n")


# ===== Train / Eval =====
def train_one_epoch(eeg_branch, stim_branch, temp_branch, dual_attn,
                    dataloader, optimizer, criterion, device):
    eeg_branch.train()
    stim_branch.train()
    temp_branch.train()
    dual_attn.train()

    total_loss, correct, total = 0, 0, 0

    for eeg, stim, label, _ in dataloader:
        eeg, stim, label = eeg.to(device), stim.to(device), label.to(device)

        optimizer.zero_grad()

        # Forward
        eeg_feat = eeg_branch(eeg)                                  # (B, D_eeg)
        stim_feat = stim_branch(stim)                               # (B, D_query)
        temp_feat = temp_branch(eeg, label)                         # (B, D_query)
        logits, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)    # (B, n_classes)

        # Loss + backward
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, pred = logits.max(1)
        total += label.size(0)
        correct += pred.eq(label).sum().item()

    acc = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc


def evaluate(eeg_branch, stim_branch, temp_branch, dual_attn,
             dataloader, criterion, device):
    eeg_branch.eval()
    stim_branch.eval()
    temp_branch.eval()
    dual_attn.eval()

    total_loss, correct, total = 0, 0, 0
    task_correct, task_total = {}, {}

    with torch.no_grad():
        for eeg, stim, label, task in dataloader:
            eeg, stim, label = eeg.to(device), stim.to(device), label.to(device)

            eeg_feat = eeg_branch(eeg)
            stim_feat = stim_branch(stim)
            temp_feat = temp_branch(eeg, label)
            logits, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)

            # Loss
            loss = criterion(logits, label)
            total_loss += loss.item()

            # Accuracy
            _, pred = logits.max(1)
            total += label.size(0)
            correct += pred.eq(label).sum().item()

            # Per-task accuracy
            for t, p, l in zip(task, pred.cpu(), label.cpu()):
                t = str(t)
                if t not in task_correct:
                    task_correct[t] = 0
                    task_total[t] = 0
                task_correct[t] += int(p == l)
                task_total[t] += 1

    acc = correct / total
    avg_loss = total_loss / len(dataloader)
    task_acc = {t: task_correct[t] / task_total[t] for t in task_total}

    return avg_loss, acc, task_acc


# ===== Main =====
def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset split: session-01 = train, session-02 = test
    train_files = sorted(glob.glob(os.path.join(args.data_root, "*ses-01.npz")))
    test_files = sorted(glob.glob(os.path.join(args.data_root, "*ses-02.npz")))

    train_dataset = ConcatDataset([SSVEPDataset(f) for f in train_files])
    test_dataset = ConcatDataset([SSVEPDataset(f) for f in test_files])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    n_channels = train_dataset.datasets[0].C
    n_samples = train_dataset.datasets[0].T
    n_classes = train_dataset.datasets[0].n_classes

    # Model components
    eeg_branch = EEGBranch(chans=n_channels, samples=n_samples).to(device)
    stim_branch = StimulusBranch(input_dim=2, hidden_dim=args.d_query).to(device)
    temp_branch = TemplateBranch(n_bands=8, n_features=32,
                                 n_channels=n_channels, n_samples=n_samples,
                                 n_classes=n_classes,
                                 D_temp=args.d_query).to(device)
    dual_attn = DualAttention(d_eeg=1024,
                              d_query=args.d_query,
                              d_model=args.d_model,
                              num_heads=4,
                              proj_dim=n_classes).to(device)

    # Optimizer, loss, scheduler
    params = list(eeg_branch.parameters()) + list(stim_branch.parameters()) + \
             list(temp_branch.parameters()) + list(dual_attn.parameters())

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print_model_size(dual_attn)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            eeg_branch, stim_branch, temp_branch, dual_attn,
            train_loader, optimizer, criterion, device
        )
        test_loss, test_acc, task_acc = evaluate(
            eeg_branch, stim_branch, temp_branch, dual_attn,
            test_loader, criterion, device
        )

        scheduler.step()

        # Log results
        print(f"\n[Epoch {epoch:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        # Per-task accuracy
        for t, acc in task_acc.items():
            print(f"   Task {t}: {acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="your data path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d_query", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    args = parser.parse_args()

    main(args)