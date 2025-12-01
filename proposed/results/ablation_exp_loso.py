# -*- coding:utf-8 -*-
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import ExpLee2019Dataset_LOSO
from branches import EEGBranch, StimulusBranch, TemplateBranch
from stimulus_auto_corrector import StimulusAutoCorrector


# ============================================================
# 1. Reproducibility
# ============================================================
def set_seed(seed=777):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Scaled Dot-Product Attention
# ============================================================
class SoftmaxAttention(nn.Module):
    """
    EEG_feat : (B, T, D_eeg)
    Query_feat : (B, D_query)
    Produces: logits (B, n_classes)
    """
    def __init__(self, d_eeg, d_query, d_model, n_classes):
        super().__init__()
        self.key = nn.Linear(d_eeg, d_model)
        self.value = nn.Linear(d_eeg, d_model)
        self.query = nn.Linear(d_query, d_model)
        self.scale = np.sqrt(d_model)
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, eeg_feat, query_feat):
        K = self.key(eeg_feat)                  # (B, T, D)
        V = self.value(eeg_feat)                # (B, T, D)
        Q = self.query(query_feat).unsqueeze(1) # (B, 1, D)

        att = torch.softmax((Q @ K.transpose(1, 2)) / self.scale, dim=-1)
        out = att @ V                     # (B, 1, D)
        out = out.squeeze(1)

        logits = self.proj(out)
        return logits, out


# ============================================================
# 3. EEG-only Classifier
# ============================================================
class EEGOnlyClassifier(nn.Module):
    """
    Query 없이 EEG만 Attention
    """
    def __init__(self, d_eeg, d_model, n_classes):
        super().__init__()
        self.key = nn.Linear(d_eeg, d_model)
        self.value = nn.Linear(d_eeg, d_model)
        self.query = nn.Parameter(torch.randn(1, d_model))
        self.scale = np.sqrt(d_model)
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, eeg_feat):
        B = eeg_feat.size(0)
        K = self.key(eeg_feat)          # (B, T, D)
        V = self.value(eeg_feat)        # (B, T, D)
        Q = self.query.unsqueeze(0).repeat(B, 1).unsqueeze(1)  # (B,1,D)

        att = torch.softmax((Q @ K.transpose(1, 2)) / self.scale, dim=-1)
        out = att @ V
        out = out.squeeze(1)

        logits = self.proj(out)
        return logits, out


# ============================================================
# 4. ITR function
# ============================================================
def compute_itr(acc, n_classes, trial_time, eps=1e-12):
    if acc <= 0 or n_classes <= 1:
        return 0.0

    acc = min(max(acc, eps), 1 - eps)
    itr = (
        np.log2(n_classes)
        + acc * np.log2(acc)
        + (1 - acc) * np.log2((1 - acc) / (n_classes - 1))
    )
    itr = 60.0 / trial_time * itr
    return itr


# ============================================================
# 5. Train / Eval
# ============================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device, mode):
    model["eeg"].train()
    if "stim" in model: model["stim"].train()
    if "temp" in model: model["temp"].train()
    if "corr" in model: model["corr"].train()
    model["cls"].train()

    all_preds, all_labels = [], []
    total_loss = 0

    for eeg, label, nominal, subj in dataloader:
        eeg, label, nominal = eeg.to(device), label.to(device), nominal.to(device)

        optimizer.zero_grad()

        feat_seq = model["eeg"](eeg, return_sequence=True)
        feat_global = model["eeg"](eeg, return_sequence=False)

        # --- Correction ---
        if mode != "wocorr":
            corrected, _ = model["corr"](eeg, nominal)
        else:
            corrected = nominal

        # --- Stimulus ---
        stim_feat = None
        if mode != "wostim":
            stim_feat = model["stim"](corrected)

        # --- Template ---
        temp_feat = None
        if mode != "wotemp":
            temp_feat = model["temp"](eeg)

        # ===== Classifier =====
        if mode == "eegnet_only":
            logits, _ = model["cls"](feat_seq)

        else:
            if stim_feat is None and temp_feat is None:
                query = feat_global
            elif stim_feat is None:
                query = temp_feat
            elif temp_feat is None:
                query = stim_feat
            else:
                query = torch.cat([stim_feat, temp_feat], dim=-1)

            logits, _ = model["cls"](feat_seq, query)

        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        _, pred = logits.max(1)
        all_preds.append(pred.cpu())
        all_labels.append(label.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return total_loss / len(all_labels), (all_preds == all_labels).float().mean().item()


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, n_classes, trial_time, mode):
    model["eeg"].eval()
    if "stim" in model: model["stim"].eval()
    if "temp" in model: model["temp"].eval()
    if "corr" in model: model["corr"].eval()
    model["cls"].eval()

    all_preds, all_labels = [], []
    total_loss = 0

    for eeg, label, nominal, subj in dataloader:
        eeg, label, nominal = eeg.to(device), label.to(device), nominal.to(device)

        feat_seq = model["eeg"](eeg, return_sequence=True)
        feat_global = model["eeg"](eeg, return_sequence=False)

        # correction
        if mode != "wocorr":
            corrected, _ = model["corr"](eeg, nominal)
        else:
            corrected = nominal

        stim_feat = None
        if mode != "wostim":
            stim_feat = model["stim"](corrected)

        temp_feat = None
        if mode != "wotemp":
            temp_feat = model["temp"](eeg)

        # classifier
        if mode == "eegnet_only":
            logits, _ = model["cls"](feat_seq)
        else:
            if stim_feat is None and temp_feat is None:
                query = feat_global
            elif stim_feat is None:
                query = temp_feat
            elif temp_feat is None:
                query = stim_feat
            else:
                query = torch.cat([stim_feat, temp_feat], dim=-1)

            logits, _ = model["cls"](feat_seq, query)

        loss = criterion(logits, label)
        total_loss += loss.item() * label.size(0)

        _, pred = logits.max(1)
        all_preds.append(pred.cpu())
        all_labels.append(label.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item()
    itr = compute_itr(acc, n_classes, trial_time)
    return total_loss / len(all_labels), acc, itr


# ============================================================
# 6. Main
# ============================================================
def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args.ablation
    save_root = f"/home/brainlab/Workspace/jycha/SSVEP/ablation/{mode}"

    subjects = list(range(1, 55))
    all_accs, all_itrs = [], []

    for test_subj in subjects:
        train_subjs = [s for s in subjects if s != test_subj]

        writer = SummaryWriter(f"{save_root}/runs/sub{test_subj}")

        train_ds = ExpLee2019Dataset_LOSO(train_subjs, pick_channels=args.pick_channels)
        test_ds = ExpLee2019Dataset_LOSO([test_subj], pick_channels=args.pick_channels)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        C = train_ds.C
        T = train_ds.T
        n_classes = train_ds.n_classes
        sfreq = train_ds.sfreq
        trial_time = T / sfreq

        print(f"[INFO] Dataset: Lee2019")
        print(f"[INFO] Subjects used ({len(subjects)}): {subjects}")
        print(f"[INFO] Train subjects: {np.unique(train_ds.subjects)}")
        print(f"[INFO] Test subjects: {np.unique(test_ds.subjects)}")
        print(f"[INFO] Train/Test samples: {len(train_ds)}/{len(test_ds)}")
        print(f"[INFO] Channels used ({C}): {', '.join(args.pick_channels)}")
        print(f"[INFO] Input shape: (C={C}, T={T}), Classes={n_classes}, Trial={trial_time:.2f}s, Sampling Rate={sfreq}Hz\n")

        # -------------------------
        # Model
        # -------------------------
        model = {}
        model["eeg"] = EEGBranch(chans=C, samples=T).to(device)

        if mode != "wostim":
            model["stim"] = StimulusBranch(
                T=T, sfreq=sfreq, hidden_dim=args.d_query, n_harmonics=3
            ).to(device)

        if mode != "wotemp":
            model["temp"] = TemplateBranch(
                n_bands=8, n_features=32, n_channels=C, n_samples=T,
                n_classes=n_classes, D_temp=args.d_query
            ).to(device)

        if mode != "wocorr":
            model["corr"] = StimulusAutoCorrector(
                eeg_channels=C, hidden_dim=args.d_query
            ).to(device)

        # classifier
        if mode == "eegnet_only":
            model["cls"] = EEGOnlyClassifier(
                d_eeg=model["eeg"].feature_dim,
                d_model=args.d_model,
                n_classes=n_classes
            ).to(device)
        else:
            # Query dimension: stim only / temp only / stim+temp
            query_dim = args.d_query
            if mode == "none":
                query_dim = args.d_query * 2

            model["cls"] = SoftmaxAttention(
                d_eeg=model["eeg"].feature_dim,
                d_query=query_dim,
                d_model=args.d_model,
                n_classes=n_classes
            ).to(device)

        # optimizer
        params = []
        for k in model:
            params += list(model[k].parameters())

        optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # -------------------------
        # Training Loop
        # -------------------------
        best_acc, best_itr = 0, 0

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                              optimizer, criterion, device, mode)
            te_loss, te_acc, itr = evaluate(model, test_loader,
                                            criterion, device, n_classes, trial_time, mode)

            scheduler.step()

            writer.add_scalar("Train/Acc", tr_acc, epoch)
            writer.add_scalar("Test/Acc", te_acc, epoch)

            if te_acc > best_acc:
                best_acc = te_acc
                best_itr = itr
                save_path = f"{save_root}/model_path/sub{test_subj}.pth"
                torch.save({k: model[k].state_dict() for k in model}, save_path)

        all_accs.append(best_acc)
        all_itrs.append(best_itr)

    print("\n===== FINAL LOSO =====")
    print(f"Acc: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")
    print(f"ITR: {np.mean(all_itrs):.4f} ± {np.std(all_itrs):.4f}")


# ============================================================
# 7. Arguments
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ablation", type=str,
                        default="eegnet_only",
                        choices=["wocorr", "wostim", "wotemp", "eegnet_only"])

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d_query", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--pick_channels", type=str, default="P3,P4,P7,P8,Pz,PO9,PO10,O1,O2,Oz")
    args = parser.parse_args()

    # format channels
    if isinstance(args.pick_channels, str):
        cleaned = args.pick_channels.strip("[]")
        args.pick_channels = [c.strip().strip("'").strip('"') for c in cleaned.split(",")]

    main(args)