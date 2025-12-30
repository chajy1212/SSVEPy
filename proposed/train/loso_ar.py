# -*- coding:utf-8 -*-
import os, glob
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset

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
        if dataset_name == "AR":
            subjects = list(range(1, 25))
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

        # EEG feature
        eeg_feat = eeg_branch(eeg, return_sequence=True)

        stim_feat = stim_branch(freq, phase)

        # Template: 정답 라벨 입력 (업데이트 발생)
        temp_feat = temp_branch(eeg, label)

        # Dual Attention
        logits, _, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)

        # Loss
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
             dataloader, ce_criterion, device, n_classes, trial_time,
             cand_freqs, cand_phases):
    """
    Pattern Matching 방식으로 완전히 변경
    """
    eeg_branch.eval()
    stim_branch.eval()
    temp_branch.eval()
    dual_attn.eval()

    all_preds, all_labels = [], []
    task_correct, task_total = {}, {}

    total_loss = 0.0

    # 후보군 텐서 변환
    candidate_freqs = torch.tensor(cand_freqs, dtype=torch.float32).to(device)
    candidate_phases = torch.tensor(cand_phases, dtype=torch.float32).to(device)
    candidate_indices = torch.arange(n_classes).to(device)

    # dataset size counting
    total_samples = 0

    # 정답(freq, phase)은 사용하지 않음 (_)
    for eeg, label, _, _, task in dataloader:
        eeg, label = eeg.to(device), label.to(device)
        B = eeg.size(0)
        total_samples += B

        # 1. EEG Feature
        eeg_feat = eeg_branch(eeg, return_sequence=True)

        batch_scores = []

        # 2. Pattern Matching Loop
        for cls_idx, f_val, p_val in zip(candidate_indices, candidate_freqs, candidate_phases):
            # (A) Stimulus: 후보 주파수/위상
            f_batch = f_val.view(1).expand(B)
            p_batch = p_val.view(1).expand(B)
            stim_feat = stim_branch(f_batch, p_batch)

            # (B) Template: 후보 클래스 인덱스 -> 해당 템플릿 호출
            cls_batch = cls_idx.view(1).expand(B)
            temp_feat = temp_branch(eeg, cls_batch)

            # (C) Dual Attention
            logits, _, _, _ = dual_attn(eeg_feat, stim_feat, temp_feat)

            # 점수 저장
            score = logits[:, cls_idx]
            batch_scores.append(score.unsqueeze(1))

        # 가장 높은 점수 선택
        batch_scores = torch.cat(batch_scores, dim=1)  # (B, n_classes)
        preds = batch_scores.argmax(dim=1)

        # CrossEntropy Loss 계산
        loss = ce_criterion(batch_scores, label)
        total_loss += loss.item() * B

        all_preds.append(preds.cpu())
        all_labels.append(label.cpu())

        # Task accuracy
        for t, p, l in zip(task, preds.cpu(), label.cpu()):
            t = str(t)
            task_correct[t] = task_correct.get(t, 0) + int(p == l)
            task_total[t] = task_total.get(t, 0) + 1

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss / total_samples
    acc = (all_preds == all_labels).float().mean().item()

    task_acc = {t: task_correct[t] / task_total[t] for t in task_total if task_total[t] > 0}
    task_itr = {t: compute_itr(a, n_classes, trial_time) for t, a in task_acc.items()}
    itr = compute_itr(acc, n_classes, trial_time)

    return avg_loss, acc, task_acc, itr, task_itr


# ===== Main (LOSO) =====
def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        for test_subj in subj_list:
            print(f"\n========== [Test Subject {test_subj:02d}] ==========")

            # Train/Test Split
            train_subj_list = [s for s in subj_list if s != test_subj]

            writer = SummaryWriter(
                log_dir=f"/home/brainlab/Workspace/jycha/SSVEP/runs/LOSOAR{exp_name}_sub{test_subj}_EEGNet_{ch_tag}")

            # Build datasets
            train_datasets = []
            try:
                for s in train_subj_list:
                    train_datasets.append(ARDataset(args.ar_data_root, s, exp_name, session="train"))

                train_dataset = ConcatDataset(train_datasets)
                test_dataset = ARDataset(args.ar_data_root, test_subj, exp_name, session="test")
            except FileNotFoundError as e:
                print(e)
                continue

            n_channels = test_dataset.C
            n_samples = test_dataset.T
            n_classes = test_dataset.n_classes
            sfreq = test_dataset.sfreq
            trial_time = n_samples / sfreq

            # [추가] Class별 고유 주파수/위상 정보 추출 (evaluate에 전달하기 위함)
            cand_freqs = [0.0] * n_classes
            cand_phases = [0.0] * n_classes

            # test_dataset에서 각 클래스별 주파수/위상 정보 수집
            # (모든 샘플이 같은 매핑을 공유한다고 가정)
            unique_labels = np.unique(test_dataset.labels)
            for l in unique_labels:
                idx = np.where(test_dataset.labels == l)[0][0]
                cand_freqs[l] = test_dataset.freqs[idx]
                cand_phases[l] = test_dataset.phases[idx]

            print(f"[INFO] Dataset: AR")
            print(f"[INFO] Train subjects: {train_subj_list}")
            print(f"[INFO] Test subject: {test_subj}")
            print(f"[INFO] Train/Test samples: {len(train_dataset)}/{len(test_dataset)}")
            print(f"[INFO] Input shape: (C={n_channels}, T={n_samples}), Classes={n_classes}")

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

            best_acc, best_itr, best_epoch = 0.0, 0.0, 0
            best_task_acc, best_task_itr = None, None

            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train_one_epoch(
                    eeg_branch, stim_branch, temp_branch, dual_attn,
                    train_loader, optimizer, ce_criterion, device
                )
                test_loss, test_acc, task_acc, itr, task_itr = evaluate(
                    eeg_branch, stim_branch, temp_branch, dual_attn,
                    test_loader, ce_criterion, device,
                    n_classes=n_classes, trial_time=trial_time,
                    cand_freqs=cand_freqs, cand_phases=cand_phases
                )

                scheduler.step()

                print(f"\n[Epoch {epoch:03d}] "
                      f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} || "
                      f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f} | "
                      f"ITR: {itr:.4f} bits/min")

                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Test", test_loss, epoch)
                writer.add_scalar("Accuracy/Train", train_acc, epoch)
                writer.add_scalar("Accuracy/Test", test_acc, epoch)
                writer.add_scalar("ITR/Test", itr, epoch)

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_itr = itr
                    best_epoch = epoch
                    best_task_acc = task_acc
                    best_task_itr = task_itr

                    save_dir = "/home/brainlab/Workspace/jycha/SSVEP/model_path"
                    save_path = os.path.join(save_dir, f"LOSOAR{exp_name}_sub{test_subj}_EEGNet_{ch_tag}.pth")

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)

                    torch.save({
                        "epoch": best_epoch,
                        "best_acc": best_acc,
                        "best_itr": best_itr,
                        "eeg_branch": eeg_branch.state_dict(),
                        "stim_branch": stim_branch.state_dict(),
                        "temp_branch": temp_branch.state_dict(),
                        "dual_attn": dual_attn.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }, save_path)

                    print(f"\n[Save] Epoch {best_epoch} → Best model "
                          f"(Acc={best_acc:.5f}, ITR={best_itr:.4f}) saved to {save_path}")

            writer.close()
            all_accs.append(best_acc)
            all_itrs.append(best_itr)

        if len(all_accs) > 0:
            print(f"\n[{exp_name} Summary] Mean Acc: {np.mean(all_accs):.5f} ± {np.std(all_accs):.5f}")
            print(f"[{exp_name} Summary] Mean ITR: {np.mean(all_itrs):.4f} ± {np.std(all_itrs):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar_data_root", type=str, default="/home/brainlab/Workspace/jycha/SSVEP/processed_npz_occi")
    parser.add_argument("--subjects", type=str, default="all", help=" '1,2,3', '1-10', '1-5,7,9-12', 'all' ")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--d_query", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--pick_channels", type=str, default="PO3,PO4,PO5,PO6,PO7,PO8,POz,O1,O2,Oz", help=" 'all' ")
    args = parser.parse_args()

    if isinstance(args.pick_channels, str):
        if args.pick_channels.lower() == "all":
            args.pick_channels = "all"
        else:
            cleaned = args.pick_channels.strip("[]")
            args.pick_channels = [ch.strip().strip("'").strip('"') for ch in cleaned.split(",")]

    main(args)