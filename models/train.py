# -*- coding:utf-8 -*-
import os, glob
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split

from dual_attention import DualAttention
from branches import EEGBranch, StimulusBranch, TemplateBranch
from data_loader import ARDataset, Nakanishi2015Dataset, Lee2019Dataset


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
    print(f"  Memory Estimate      : {total_params * 4 / (1024**2):.2f} MB\n")


# ===== Subject parser =====
def parse_subjects(subjects_arg, dataset_name=""):
    """
    subjects_arg: e.g. "1,2,3", "1-10", "1-5,7,9-12", "all"
    dataset_name: "AR", "Nakanishi2015", "Lee2019"
    """
    if subjects_arg.lower() == "all":
        if dataset_name == "AR":
            subjects = list(range(1, 25))  # sub-001 ~ sub-024
        elif dataset_name == "Nakanishi2015":
            subjects = list(range(1, 10))  # 1 ~ 9
        elif dataset_name == "Lee2019":
            subjects = list(range(1, 55))  # 1 ~ 54
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        print(f"[INFO] {dataset_name} subjects selected: {subjects}")
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

    print(f"[INFO] {dataset_name} subjects selected: {subjects}")
    return subjects


# ===== Train / Eval =====
def train_one_epoch(eeg_branch, stim_branch, temp_branch, dual_attn,
                    dataloader, optimizer, criterion, device, with_task=False):
    eeg_branch.train()
    stim_branch.train()
    temp_branch.train()
    dual_attn.train()

    total_loss, correct, total = 0, 0, 0

    for batch in dataloader:
        if with_task:
            eeg, stim, label, _ = batch
        else:
            eeg, stim, label = batch
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


@torch.no_grad()
def evaluate(eeg_branch, stim_branch, temp_branch, dual_attn,
             dataloader, criterion, device, with_task=False):
    eeg_branch.eval()
    stim_branch.eval()
    temp_branch.eval()
    dual_attn.eval()

    total_loss, correct, total = 0, 0, 0
    task_correct, task_total = {}, {}

    for batch in dataloader:
        if with_task:
            eeg, stim, label, task = batch
        else:
            eeg, stim, label = batch
            task = None
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
        if with_task:
            for t, p, l in zip(task, pred.cpu(), label.cpu()):
                t = str(t)
                task_correct.setdefault(t, 0)
                task_total.setdefault(t, 0)
                task_correct[t] += int(p == l)
                task_total[t] += 1

    acc = correct / total
    avg_loss = total_loss / len(dataloader)
    task_acc = {t: task_correct[t] / task_total[t] for t in task_total} if with_task else None

    return avg_loss, acc, task_acc


# ===== Main =====
def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset selection
    if args.dataset == "AR":
        all_train_files = sorted(glob.glob(os.path.join(args.data_root, "*ses-01.npz")))
        all_test_files = sorted(glob.glob(os.path.join(args.data_root, "*ses-02.npz")))

        if args.subjects.lower() != "all":
            subjects = parse_subjects(args.subjects, "AR")
            train_files = [f for f in all_train_files if any(f"sub-{s:03d}_" in f for s in subjects)]
            test_files = [f for f in all_test_files if any(f"sub-{s:03d}_" in f for s in subjects)]
        else:
            train_files, test_files = all_train_files, all_test_files

        train_dataset = ConcatDataset([ARDataset(f) for f in train_files])
        test_dataset = ConcatDataset([ARDataset(f) for f in test_files])
        n_channels, n_samples, n_classes = train_dataset.datasets[0].C, train_dataset.datasets[0].T, \
            train_dataset.datasets[0].n_classes
        with_task = True

    elif args.dataset == "Nakanishi2015":
        subjects = parse_subjects(args.subjects, "Nakanishi2015")
        dataset = Nakanishi2015Dataset(subjects=subjects)
        N = len(dataset)
        N_train = int(0.8 * N)
        train_dataset, test_dataset = random_split(dataset, [N_train, N - N_train])
        n_channels, n_samples, n_classes = dataset.C, dataset.T, dataset.n_classes
        with_task = False

    elif args.dataset == "Lee2019":
        subjects = parse_subjects(args.subjects, "Lee2019")
        train_dataset = Lee2019Dataset(subjects=subjects, train=True)  # session 0
        test_dataset = Lee2019Dataset(subjects=subjects, train=False)  # session 1
        n_channels, n_samples, n_classes = train_dataset.C, train_dataset.T, train_dataset.n_classes
        with_task = False

    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    eeg_branch = EEGBranch(chans=n_channels, samples=n_samples).to(device)
    stim_branch = StimulusBranch(hidden_dim=args.d_query, n_harmonics=3).to(device)
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

    # Optimizer, loss, scheduler
    params = list(eeg_branch.parameters()) + list(stim_branch.parameters()) + \
             list(temp_branch.parameters()) + list(dual_attn.parameters())

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(eeg_branch, stim_branch, temp_branch, dual_attn,
                                                train_loader, optimizer, criterion, device, with_task)
        test_loss, test_acc, task_acc = evaluate(eeg_branch, stim_branch, temp_branch, dual_attn,
                                                 test_loader, criterion, device, with_task)

        scheduler.step()

        print(f"\n[Epoch {epoch:03d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        if task_acc is not None:
            for t, acc in task_acc.items():
                print(f"   Task {t}: {acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Lee2019", choices=["AR", "Nakanishi2015", "Lee2019"])
    parser.add_argument("--data_root", type=str, default="/home/jycha/SSVEP/processed_npz")
    parser.add_argument("--subjects", type=str, default="all", help=" '1,2,3', '1-10', '1-5,7,9-12', 'all' ")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d_query", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    args = parser.parse_args()

    main(args)