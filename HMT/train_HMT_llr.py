"""HMT training with three benign-filtering variants."""
import argparse
import pickle
import sys
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from HMT_model import HybridModifiedTransformer
from HMT_dataloader import BigramDictionaryBuilder, HMTDataset
from utility.path import path_artifacts, path_train_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["raw", "llr", "random"])
    parser.add_argument("--llr-threshold", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-layers", type=int, default=4)
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
    print(f"[HMT/{args.variant}] LLR threshold tau = {tau:+.4f}")

    if args.variant == "raw":
        kept = benign_df
    elif args.variant == "llr":
        kept = benign_df[benign_df["llr"] <= tau].reset_index(drop=True)
    elif args.variant == "random":
        drop_frac = (benign_df["llr"] > tau).mean()
        keep_frac = 1.0 - drop_frac
        idx = pd.Series(range(len(benign_df))).sample(int(len(benign_df) * keep_frac), random_state=args.random_seed).values
        kept = benign_df.iloc[idx].reset_index(drop=True)
    print(f"[HMT/{args.variant}] Benign kept: {len(kept):,} / {n_total:,} ({len(kept)/n_total:.2%})")

    kept = kept[target_cols]
    benign_train_df, benign_val_df = train_test_split(kept, test_size=0.1, random_state=42)

    train_df = pd.concat([benign_train_df, dga_train_df]).reset_index(drop=True)
    val_df = pd.concat([benign_val_df, dga_val_df]).reset_index(drop=True)
    print(f"[HMT/{args.variant}] train={len(train_df):,} val={len(val_df):,}")

    # Build (or load shared) bigram dictionary from this variant's training data.
    bigram_pkl = path_artifacts.joinpath(f"bigram_dict_HMT_{args.variant}.pkl")
    print(f"[HMT/{args.variant}] Building bigram dictionary from {len(train_df):,} train domains...", flush=True)
    bd = BigramDictionaryBuilder(train_df["domain"].tolist())
    bigram_dict = bd.build()
    with open(bigram_pkl, "wb") as f:
        pickle.dump(bigram_dict, f)
    print(f"[HMT/{args.variant}] Bigram dict size: {len(bigram_dict)}")

    train_loader = DataLoader(HMTDataset(train_df, bigram_dict), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(HMTDataset(val_df, bigram_dict), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HMT/{args.variant}] Device: {device}")
    model = HybridModifiedTransformer(num_classes=1, n_transformer_layers=args.n_layers).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0
    model_path = path_artifacts.joinpath(f"HMT_T24_{args.variant}_30days.pt")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n = 0
        for cx, bx, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            cx, bx, y = cx.to(device, non_blocking=True), bx.to(device, non_blocking=True), y.to(device, non_blocking=True).float()
            optimizer.zero_grad()
            out = model(cx, bx).squeeze(1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * cx.size(0)
            n += cx.size(0)
        avg_train_loss = total_loss / n

        model.eval()
        val_loss = 0.0
        n_val = 0
        val_out = []
        val_lab = []
        with torch.no_grad():
            for cx, bx, y in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                cx, bx, y = cx.to(device, non_blocking=True), bx.to(device, non_blocking=True), y.to(device, non_blocking=True).float()
                out = model(cx, bx).squeeze(1)
                loss = criterion(out, y)
                val_loss += loss.item() * cx.size(0)
                n_val += cx.size(0)
                val_out.append(torch.sigmoid(out).detach().cpu())
                val_lab.append(y.detach().cpu())
        avg_val_loss = val_loss / n_val
        val_out = torch.cat(val_out)
        val_lab = torch.cat(val_lab)
        fpr, tpr, thr = roc_curve(val_lab.numpy(), val_out.numpy())
        theta = thr[(tpr - fpr).argmax()]
        pred = (val_out > theta).int()
        true = val_lab.int()
        accuracy = (pred == true).sum().item() / len(true)
        precision = precision_score(true.numpy(), pred.numpy(), zero_division=0)
        recall = recall_score(true.numpy(), pred.numpy(), zero_division=0)
        f1 = f1_score(true.numpy(), pred.numpy(), zero_division=0)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)

        dt = time.time() - t0
        print(f"[HMT/{args.variant}] ep {epoch+1:2d}/{args.epochs} [{dt:.1f}s] train={avg_train_loss:.4f} val={avg_val_loss:.4f} acc={accuracy:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f} theta={theta:.4f}", flush=True)

    print(f"\n[HMT/{args.variant}] Best epoch {best_epoch} (val_loss={best_val_loss:.4f}); saved to {model_path}")


if __name__ == "__main__":
    main()
