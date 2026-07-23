"""Evaluate a trained NYU variant on T25-26 test."""
import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from dataloader import DGA_Dataset
from NYU_model import NYU_DGA
from utility.path import path_artifacts, path_results, path_test_data, path_train_data
sys.path.append(str(Path(__file__).parent.parent / "MIT"))
import MIT.compute_llr as llr_mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["raw", "llr", "random"])
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_cols = ["domain", "label"]

    model = NYU_DGA(n_classes=1).to(device)
    ckpt = path_artifacts.joinpath(f"NYU_T24_{args.variant}_30days.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    benign_full = pd.read_parquet(path_train_data.joinpath("T24_benign_30days_with_llr.parquet"))
    dga_val_df = pd.read_parquet(path_train_data.joinpath("T24_dga_30days_val.parquet"), columns=target_cols)
    tau = float(path_train_data.joinpath("llr_threshold.txt").read_text().strip())

    if args.variant == "raw":
        benign_for_val = benign_full
    elif args.variant == "llr":
        benign_for_val = benign_full[benign_full["llr"] <= tau].reset_index(drop=True)
    elif args.variant == "random":
        drop_frac = (benign_full["llr"] > tau).mean()
        keep_frac = 1.0 - drop_frac
        idx = pd.Series(range(len(benign_full))).sample(int(len(benign_full) * keep_frac), random_state=42).values
        benign_for_val = benign_full.iloc[idx].reset_index(drop=True)

    _, benign_val_df = train_test_split(benign_for_val[target_cols], test_size=0.1, random_state=42)
    val_df = pd.concat([benign_val_df, dga_val_df]).reset_index(drop=True)
    val_loader = DataLoader(DGA_Dataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    val_outputs, val_labels = [], []
    with torch.no_grad():
        for X, y in tqdm(val_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(X)
            val_outputs.append(out.detach().cpu().view(-1))
            val_labels.append(y.detach().cpu().view(-1))
    val_outputs = torch.cat(val_outputs)
    val_labels = torch.cat(val_labels)
    fpr_v, tpr_v, thr_v = roc_curve(val_labels.numpy(), val_outputs.numpy())
    theta = float(thr_v[(tpr_v - fpr_v).argmax()])
    print(f"[NYU/{args.variant}] theta = {theta:.4f}", flush=True)

    dga_test_df = pd.read_parquet(path_test_data.joinpath("T25-26_dga_byYear.parquet"), columns=target_cols)
    benign_test_df = pd.read_parquet(path_test_data.joinpath("T25-26_benign_30days.parquet"))
    print(f"[NYU/{args.variant}] DGA test: {len(dga_test_df):,}, benign test: {len(benign_test_df):,}", flush=True)

    print(f"[NYU/{args.variant}] Computing LLR on test benigns...", flush=True)
    benign_train_full = pd.read_parquet(path_train_data.joinpath("T24_benign_30days.parquet"))
    ref_size = int(len(benign_train_full) * llr_mod.REF_BENIGN_FRAC)
    ref_benign = benign_train_full.nsmallest(ref_size, "avg_rank")["domain"].tolist()
    M_B = llr_mod.build_bigram(ref_benign)
    dga_train_full = pd.read_parquet(path_train_data.joinpath("T24_dga_30days_train.parquet"))
    M_D = llr_mod.build_bigram(dga_train_full["domain"].sample(min(200_000, len(dga_train_full)), random_state=42).tolist())
    benign_test_df["llr"] = benign_test_df["domain"].map(lambda d: llr_mod.score(d, M_D, M_B))
    print(f"[NYU/{args.variant}] Test benign LLR > tau: {(benign_test_df['llr'] > tau).mean():.4%}", flush=True)

    test_df = pd.concat([benign_test_df[target_cols], dga_test_df]).reset_index(drop=True)
    test_loader = DataLoader(DGA_Dataset(test_df), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    t0 = time.time()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc=f"[Test] {args.variant}", leave=False):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(X)
            all_preds.append(out.detach().cpu().view(-1))
            all_labels.append(y.detach().cpu().view(-1))
    print(f"[NYU/{args.variant}] Inference: {time.time()-t0:.1f}s", flush=True)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(int)
    pred_labels = (all_preds > theta).astype(int)

    def metrics(y_true, y_pred, name):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        acc = (y_pred == y_true).mean()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        print(f"  [{name}] n={len(y_true):>8,} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} FPR={fpr:.4f} FNR={fnr:.4f}  (TP={tp} FP={fp} TN={tn} FN={fn})", flush=True)
        return dict(name=name, n=len(y_true), acc=acc, prec=prec, rec=rec, f1=f1, fpr=fpr, fnr=fnr, tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))

    print(f"\n[NYU/{args.variant}] FULL TEST:", flush=True)
    full_row = metrics(all_labels, pred_labels, name="full")

    print(f"\n[NYU/{args.variant}] LLR-CLEANED TEST:", flush=True)
    test_df["llr"] = pd.concat([benign_test_df["llr"], pd.Series([np.nan] * len(dga_test_df))], ignore_index=True)
    mask = (test_df["label"] == 1) | (test_df["llr"] <= tau)
    cln_row = metrics(all_labels[mask.values], pred_labels[mask.values], name="llr_cleaned")

    out = pd.DataFrame([dict(model="NYU", variant=args.variant, theta=theta, **full_row), dict(model="NYU", variant=args.variant, theta=theta, **cln_row)])
    out_path = path_results.joinpath(f"results_NYU_{args.variant}_byYear.csv")
    out.to_csv(out_path, index=False)
    print(f"[NYU/{args.variant}] Saved metrics to {out_path}", flush=True)


if __name__ == "__main__":
    main()
