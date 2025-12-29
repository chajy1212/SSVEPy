# -*- coding:utf-8 -*-
import os, glob
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from data_loader import ARDataset
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
        if dataset_name == "AR":
            subjects = list(range(1, 25))  # 1 ~ 24
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

    all_preds, all_labels = [], []
    total_loss = 0.0

    for eeg, label, freq, phase, task in dataloader:
        eeg, label = eeg.to(device), label.to(device)
        freq, phase = freq.to(device), phase.to(device)

        optimizer.zero_grad()

        # EEG feature extraction
        eeg_feat = eeg_branch(eeg, return_sequence=True)

        # Stimulus feature
        stim_feat = stim_branch(freq, phase)

        # Template feature (label-independent)
        temp_feat = temp_branch(eeg)

        # Dual Attention forward
        logits, _, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)

        # CE loss
        loss = ce_criterion(logits, label) + dual_attn.loss_entropy
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
def evaluate(eeg_branch, stim_branch, temp_branch, dual_attn,
             dataloader, ce_criterion, device, n_classes, trial_time):
    eeg_branch.eval()
    stim_branch.eval()
    temp_branch.eval()
    dual_attn.eval()

    all_preds, all_labels = [], []
    task_correct, task_total = {}, {}

    # 데이터셋에서 후보 주파수와 위상 리스트 가져오기
    # ARDataset에 self.freqs, self.phases가 있다고 가정합니다.
    # 만약 없다면 data_loader.py를 확인하여 해당 리스트를 가져와야 합니다.
    if hasattr(dataloader.dataset, 'freqs') and hasattr(dataloader.dataset, 'phases'):
        cand_freqs = dataloader.dataset.freqs       # list or numpy array
        cand_phases = dataloader.dataset.phases     # list or numpy array
    else:
        # Fallback: 만약 dataset 속성이 없다면 직접 계산하거나 에러 처리
        # 여기서는 안전하게 N_classes만큼의 더미 데이터를 생성하지 않고 에러를 띄웁니다.
        # ARDataset 구현을 확인해주세요. 보통 self.freqs는 존재합니다.
        raise AttributeError("Dataset does not have 'freqs' or 'phases' attribute.")

    candidate_freqs = torch.tensor(cand_freqs, dtype=torch.float32).to(device)
    candidate_phases = torch.tensor(cand_phases, dtype=torch.float32).to(device)
    candidate_indices = torch.arange(n_classes).to(device)

    for eeg, label, _, _, task in dataloader:
        eeg, label = eeg.to(device), label.to(device)
        B = eeg.size(0)

        eeg_feat = eeg_branch(eeg, return_sequence=True)
        temp_feat = temp_branch(eeg)

        batch_scores = []

        # 모든 후보 클래스에 대해 반복 (Pattern Matching)
        for cls_idx, f_val, p_val in zip(candidate_indices, candidate_freqs, candidate_phases):
            # 현재 배치를 cls_idx라고 가정하고 입력 생성
            f_batch = f_val.view(1).expand(B)
            p_batch = p_val.view(1).expand(B)

            # Stimulus Feature 생성
            stim_feat = stim_branch(f_batch, p_batch)

            # Dual Attention Forward
            logits, _, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)

            # 해당 클래스(cls_idx)에 대한 점수만 가져옴
            score = logits[:, cls_idx]
            batch_scores.append(score.unsqueeze(1))

        # 가장 높은 점수 선택
        batch_scores = torch.cat(batch_scores, dim=1)  # (B, n_classes)
        preds = batch_scores.argmax(dim=1)

        # Loss 계산은 생략하거나, 정답 라벨에 대한 스코어로 계산 가능하지만
        # 여기서는 Pattern Matching 방식이므로 정확도는 preds와 labels 비교로 충분함

        all_preds.append(preds.cpu())
        all_labels.append(label.cpu())

        # Task accuracy
        for t, p, l in zip(task, preds.cpu(), label.cpu()):
            t = str(t)
            task_correct[t] = task_correct.get(t, 0) + int(p == l)
            task_total[t] = task_total.get(t, 0) + 1

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # avg_loss = total_loss / len(all_labels)
    acc = (all_preds == all_labels).float().mean().item()

    task_acc = {t: task_correct[t] / task_total[t] for t in task_total if task_total[t] > 0}
    task_itr = {t: compute_itr(a, n_classes, trial_time) for t, a in task_acc.items()}

    # ITR
    itr = compute_itr(acc, n_classes, trial_time)

    return 0.0, acc, task_acc, itr, task_itr


# ===== Main (subject-wise session split) =====
def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Channel tag
    if args.pick_channels == "all":
        ch_tag = "allch"
    else:
        ch_tag = "".join(args.pick_channels)

    subjects = parse_subjects(args.subjects, "AR")

    if args.subjects.lower() != "all":
        subject_partition = {"Custom": subjects}
    else:
        subject_partition = {
            "Exp1": list(range(1, 15)),
            # "Exp2": list(range(1, 14)) + [15],
            # "Exp3": list(range(1, 9)) + list(range(16, 25))
        }

    for exp_name, subj_list in subject_partition.items():
        print(f"\n========== [{exp_name}] Subjects: {subj_list} ==========")
        all_accs, all_itrs = [], []

        for subj in subj_list:
            print(f"\n========== [Subject {subj:02d}] ==========")

            # TensorBoard writer
            writer = SummaryWriter(log_dir=f"/home/brainlab/Workspace/jycha/SSVEP/runs/AR{exp_name}_Sub{subj}_EEGNet_{ch_tag}")

            try:
                train_dataset = ARDataset(args.ar_data_root, subj, exp_name, session="train")
                test_dataset = ARDataset(args.ar_data_root, subj, exp_name, session="test")
            except FileNotFoundError as e:
                print(e)
                continue

            n_channels = train_dataset.C
            n_samples = train_dataset.T
            n_classes = train_dataset.n_classes
            sfreq = train_dataset.sfreq
            trial_time = n_samples / sfreq

            print(f"[INFO] Dataset: AR")
            print(f"[INFO] Train/Test samples: {len(train_dataset)}/{len(test_dataset)}")
            print(f"[INFO] Channels used ({n_channels}): {', '.join(args.pick_channels)}")
            print(f"[INFO] Input shape: (C={n_channels}, T={n_samples}), Classes={n_classes}, Trial={trial_time:.2f}s, Sampling Rate={sfreq}Hz\n")

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            # Model
            eeg_branch = EEGBranch(chans=n_channels, samples=n_samples).to(device)
            stim_branch = StimulusBranchWithPhase(T=n_samples,
                                                  sfreq=sfreq,
                                                  hidden_dim=args.d_query,
                                                  n_harmonics=3,
                                                  out_dim=args.d_query).to(device)
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

            print_total_model_size(eeg_branch, stim_branch, temp_branch, dual_attn)

            params = list(eeg_branch.parameters()) + list(stim_branch.parameters()) + \
                     list(temp_branch.parameters()) + list(dual_attn.parameters())

            optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
            ce_criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            # best record
            best_acc, best_itr, best_epoch = 0.0, 0.0, 0
            best_task_acc, best_task_itr = None, None

            # Train Loop
            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train_one_epoch(
                    eeg_branch, stim_branch, temp_branch, dual_attn,
                    train_loader, optimizer, ce_criterion, device
                )
                test_loss, test_acc, task_acc, itr, task_itr = evaluate(
                    eeg_branch, stim_branch, temp_branch, dual_attn,
                    test_loader, ce_criterion, device,
                    n_classes=n_classes, trial_time=trial_time
                )

                scheduler.step()

                print(f"\n[Epoch {epoch:03d}] "
                      f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} || "
                      f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f} | "
                      f"ITR: {itr:.4f} bits/min")

                for t in task_acc.keys():
                    print(f"   Task {t:<6s} | Acc={task_acc[t]:.5f} | ITR={task_itr[t]:.4f}")

                # TensorBoard logging
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Test", test_loss, epoch)
                writer.add_scalar("Accuracy/Train", train_acc, epoch)
                writer.add_scalar("Accuracy/Test", test_acc, epoch)
                writer.add_scalar("ITR/Test", itr, epoch)
                for t in task_acc.keys():
                    writer.add_scalar(f"TaskAcc/{t}", task_acc[t], epoch)
                    writer.add_scalar(f"TaskITR/{t}", task_itr[t], epoch)

                # update best record
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_itr = itr
                    best_epoch = epoch
                    best_task_acc = task_acc
                    best_task_itr = task_itr

                    # Save Model
                    save_dir = "/home/brainlab/Workspace/jycha/SSVEP/model_path"
                    save_path = os.path.join(save_dir, f"AR{exp_name}_Sub{subj}_EEGNet_{ch_tag}.pth")

                    torch.save({
                        "epoch": best_epoch,
                        "best_acc": best_acc,
                        "best_itr": best_itr,
                        "eeg_branch": eeg_branch.state_dict(),
                        "stim_branch": stim_branch.state_dict(),
                        "temp_branch": temp_branch.state_dict(),
                        "dual_attn": dual_attn.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_task_acc": best_task_acc,
                        "best_task_itr": best_task_itr
                    }, save_path)

                    print(f"\n[Save] Epoch {best_epoch} → Best model "
                          f"(Acc={best_acc:.5f}, ITR={best_itr:.4f}) saved to {save_path}")
                    print(f"Best Task Acc: {best_task_acc}")
                    print(f"Best Task ITR: {best_task_itr}")

            writer.close()

            all_accs.append(best_acc)
            all_itrs.append(best_itr)

        # ---------------- Summary per Experiment ----------------
        if len(all_accs) > 0:
            print(f"\n[{exp_name} Summary] Mean Acc: {np.mean(all_accs):.5f} ± {np.std(all_accs):.5f}")
            print(f"[{exp_name} Summary] Mean ITR: {np.mean(all_itrs):.4f} ± {np.std(all_itrs):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar_data_root", type=str, default="/home/brainlab/Workspace/jycha/SSVEP/processed_npz_occi")
    parser.add_argument("--subjects", type=str, default="all", help=" '1,2,3', '1-10', '1-5,7,9-12', 'all' ")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d_query", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--pick_channels", type=str, default="PO3,PO4,PO5,PO6,PO7,PO8,POz,O1,O2,Oz", help=" 'all' ")
    args = parser.parse_args()

    # Parse channel selection
    if isinstance(args.pick_channels, str):
        if args.pick_channels.lower() == "all":
            args.pick_channels = "all"
        else:
            cleaned = args.pick_channels.strip("[]")
            args.pick_channels = [ch.strip().strip("'").strip('"') for ch in cleaned.split(",")]

    main(args)