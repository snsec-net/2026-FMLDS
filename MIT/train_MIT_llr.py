"""MIT detector training with three benign-filtering variants."""
import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from dataloader import DGA_Dataset
from MIT_model import MIT
from utility.path import path_artifacts, path_train_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["raw", "llr", "random"])
    parser.add_argument("--llr-threshold", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    target_cols = ["domain", "label"]

    dga_train_df = pd.read_parquet(path_train_data.joinpath("T24_dga_30days_train.parquet"), columns=target_cols)
    dga_val_df = pd.read_parquet(path_train_data.joinpath("T24_dga_30days_val.parquet"), columns=target_cols)
    benign_df = pd.read_parquet(path_train_data.joinpath("T24_benign_30days_with_llr.parquet"))
    n_total = len(benign_df)

    if args.llr_threshold is None:
        tau = float(path_train_data.joinpath("llr_threshold.txt").read_text().strip())
    else:
        tau = args.llr_threshold
    print(f"[{args.variant}] LLR threshold tau = {tau:+.4f}")

    if args.variant == "raw":
        kept = benign_df
    elif args.variant == "llr":
        kept = benign_df[benign_df["llr"] <= tau].reset_index(drop=True)
    elif args.variant == "random":
        drop_frac = (benign_df["llr"] > tau).mean()
        keep_frac = 1.0 - drop_frac
        idx = pd.Series(range(len(benign_df))).sample(int(len(benign_df) * keep_frac), random_state=args.random_seed).values
        kept = benign_df.iloc[idx].reset_index(drop=True)
    print(f"[{args.variant}] Benign kept: {len(kept):,} / {n_total:,} ({len(kept)/n_total:.2%})")

    kept = kept[target_cols]
    benign_train_df, benign_val_df = train_test_split(kept, test_size=0.1, random_state=42)
    print(f"[{args.variant}] Benign train: {len(benign_train_df):,}, Val: {len(benign_val_df):,}")
    print(f"[{args.variant}] DGA    train: {len(dga_train_df):,}, Val: {len(dga_val_df):,}")

    train_df = pd.concat([benign_train_df, dga_train_df]).reset_index(drop=True)
    val_df = pd.concat([benign_val_df, dga_val_df]).reset_index(drop=True)
    train_loader = DataLoader(DGA_Dataset(train_df), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(DGA_Dataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.variant}] Device: {device}")
    model = MIT().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0
    model_path = path_artifacts.joinpath(f"MIT_T24_{args.variant}_30days.pt")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view_as(outputs))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_train_loss = total_loss / n_batches

        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True).float()
                outputs = model(inputs)
                val_loss += criterion(outputs, labels.float().view_as(outputs)).item()
                val_outputs.append(outputs.detach().cpu().view(-1))
                val_labels.append(labels.detach().cpu().view(-1))
        avg_val_loss = val_loss / len(val_loader)

        val_outputs = torch.cat(val_outputs).view(-1)
        val_labels = torch.cat(val_labels).view(-1)
        fpr, tpr, threshold = roc_curve(val_labels.numpy(), val_outputs.numpy())
        theta = threshold[(tpr - fpr).argmax()]
        pred_labels = (val_outputs > theta).int()
        true_labels = val_labels.int()
        accuracy = (pred_labels == true_labels).sum().item() / len(true_labels)
        precision = precision_score(true_labels.numpy(), pred_labels.numpy(), zero_division=0)
        recall = recall_score(true_labels.numpy(), pred_labels.numpy(), zero_division=0)
        f1 = f1_score(true_labels.numpy(), pred_labels.numpy(), zero_division=0)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)

        dt = time.time() - t0
        print(f"[{args.variant}] ep {epoch+1:2d}/{args.epochs} [{dt:.1f}s] train={avg_train_loss:.4f} val={avg_val_loss:.4f} acc={accuracy:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f} theta={theta:.4f}", flush=True)

    print(f"\n[{args.variant}] Best epoch {best_epoch} (val_loss={best_val_loss:.4f}); saved to {model_path}")


if __name__ == "__main__":
    main()
