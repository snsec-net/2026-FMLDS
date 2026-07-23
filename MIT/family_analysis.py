"""Family-level FNR breakdown for MIT models (raw, llr, random) using DGArchive family meta.
Also computes threshold sensitivity sweep on the LLR-cleaned test view."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).parent.parent))
from MIT.dataloader import DGA_Dataset
from MIT.MIT_model import MIT
from utility.path import path_artifacts, path_results, path_test_data, path_train_data
import compute_llr as llr_mod


def threshold_for_variant(model, device, variant, tau):
    benign_full = pd.read_parquet(path_train_data.joinpath("T24_benign_30days_with_llr.parquet"))
    dga_val_df = pd.read_parquet(path_train_data.joinpath("T24_dga_30days_val.parquet"), columns=["domain", "label"])
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve

    if variant == "raw":
        benign_for_val = benign_full
    elif variant == "llr":
        benign_for_val = benign_full[benign_full["llr"] <= tau].reset_index(drop=True)
    elif variant == "random":
        drop_frac = (benign_full["llr"] > tau).mean()
        keep_frac = 1.0 - drop_frac
        idx = pd.Series(range(len(benign_full))).sample(int(len(benign_full) * keep_frac), random_state=42).values
        benign_for_val = benign_full.iloc[idx].reset_index(drop=True)

    _, benign_val_df = train_test_split(benign_for_val[["domain", "label"]], test_size=0.1, random_state=42)
    val_df = pd.concat([benign_val_df, dga_val_df]).reset_index(drop=True)
    val_loader = DataLoader(DGA_Dataset(val_df), batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    val_o, val_l = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device, non_blocking=True)
            out = model(X).cpu()
            val_o.append(out.view(-1))
            val_l.append(y.view(-1))
    val_o = torch.cat(val_o).numpy()
    val_l = torch.cat(val_l).numpy()
    fpr, tpr, thr = roc_curve(val_l, val_o)
    return float(thr[(tpr - fpr).argmax()])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau = float(path_train_data.joinpath("llr_threshold.txt").read_text().strip())

    # Load DGArchive test with family meta
    dga_test = pd.read_parquet(path_test_data.joinpath("T25-26_dga_byYear.parquet"))
    cols = set(dga_test.columns)
    print(f"DGA test columns: {sorted(cols)}")
    print(f"DGA test rows: {len(dga_test):,}")
    has_family = "family" in cols
    if has_family:
        print(f"families: {dga_test['family'].nunique()}; top 10:\n{dga_test['family'].value_counts().head(10).to_string()}")

    # Load benign test with LLR computed (for FPR-cleaned view)
    benign_test = pd.read_parquet(path_test_data.joinpath("T25-26_benign_30days.parquet"))
    print("Computing LLR on test benigns (one-shot for both analyses)...")
    benign_train_full = pd.read_parquet(path_train_data.joinpath("T24_benign_30days.parquet"))
    ref_size = int(len(benign_train_full) * llr_mod.REF_BENIGN_FRAC)
    M_B = llr_mod.build_bigram(benign_train_full.nsmallest(ref_size, "avg_rank")["domain"].tolist())
    dga_train = pd.read_parquet(path_train_data.joinpath("T24_dga_30days_train.parquet"))
    M_D = llr_mod.build_bigram(dga_train["domain"].sample(min(200_000, len(dga_train)), random_state=42).tolist())
    benign_test["llr"] = benign_test["domain"].map(lambda d: llr_mod.score(d, M_D, M_B))

    # ===== Family-level inference per variant =====
    family_rows = []
    threshold_rows = []
    for variant in ["raw", "llr", "random"]:
        ckpt = path_artifacts.joinpath(f"MIT_T24_{variant}_30days.pt")
        if not ckpt.exists():
            print(f"WARN: missing {ckpt}; skipping")
            continue
        model = MIT().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        theta = threshold_for_variant(model, device, variant, tau)
        print(f"\n=== {variant}: theta={theta:.4f} ===")

        # Inference on DGA test
        dga_loader = DataLoader(DGA_Dataset(dga_test[["domain", "label"]]), batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
        dga_preds = []
        with torch.no_grad():
            for X, _ in dga_loader:
                X = X.to(device, non_blocking=True)
                p = model(X).cpu().view(-1).numpy()
                dga_preds.append(p)
        dga_preds = np.concatenate(dga_preds)
        dga_test[f"pred_{variant}"] = (dga_preds > theta).astype(int)

        # Family-level FNR
        if has_family:
            fam_stat = dga_test.groupby("family").apply(lambda g: pd.Series({
                "n": len(g),
                "missed": int((g[f"pred_{variant}"] == 0).sum()),
                "fnr": float((g[f"pred_{variant}"] == 0).mean()),
            })).reset_index()
            fam_stat["variant"] = variant
            family_rows.append(fam_stat)

        # Threshold sensitivity sweep (re-evaluate FPR/FNR under different LLR-cleaned test views)
        benign_loader = DataLoader(DGA_Dataset(benign_test[["domain", "label"]]), batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
        ben_preds = []
        with torch.no_grad():
            for X, _ in benign_loader:
                X = X.to(device, non_blocking=True)
                p = model(X).cpu().view(-1).numpy()
                ben_preds.append(p)
        ben_preds = np.concatenate(ben_preds)
        ben_pred_labels = (ben_preds > theta).astype(int)

        dga_pred_labels = dga_test[f"pred_{variant}"].values
        dga_labels = np.ones(len(dga_test), dtype=int)
        ben_labels = np.zeros(len(benign_test), dtype=int)

        # FNR is just on DGA (no threshold of LLR matters here)
        fnr = (dga_pred_labels == 0).mean()
        # FPR sensitivity to LLR cutoff on test benigns
        for tau_test in [None, -0.2, -0.1, 0.0, tau, +0.2, +0.5]:
            if tau_test is None:
                mask = np.ones(len(benign_test), dtype=bool)
                label_str = "no_clean"
            else:
                mask = (benign_test["llr"].values <= tau_test)
                label_str = f"tau_{tau_test:+.3f}"
            sub_ben = ben_pred_labels[mask]
            sub_lbl = ben_labels[mask]
            tn = int(((sub_lbl == 0) & (sub_ben == 0)).sum())
            fp = int(((sub_lbl == 0) & (sub_ben == 1)).sum())
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            threshold_rows.append({"model":"MIT","variant": variant, "test_view": label_str, "n_benign_kept": int(mask.sum()), "fpr": fpr, "fnr": fnr})

    if family_rows:
        fam_df = pd.concat(family_rows, ignore_index=True)
        fam_df.to_csv(path_results.joinpath("family_FNR_MIT_byYear.csv"), index=False)
        print(f"\nSaved per-family FNR to {path_results.joinpath('family_FNR_MIT_byYear.csv')}")

        # Highlight word-based DGAs (if present)
        word_families = ["suppobox", "pushdo", "gazavat", "matsnu", "rovnix"]
        wf_present = fam_df[fam_df["family"].str.lower().isin(word_families)]
        if len(wf_present) > 0:
            print("\n--- Word-based DGA families ---")
            print(wf_present.to_string(index=False))

        # Pivot for headline view
        pivot = fam_df.pivot(index="family", columns="variant", values="fnr").reset_index()
        pivot["n"] = fam_df.groupby("family")["n"].first().reindex(pivot["family"]).values
        if all(v in pivot.columns for v in ["raw", "llr", "random"]):
            pivot["llr_minus_raw"] = pivot["llr"] - pivot["raw"]
            pivot = pivot.sort_values("n", ascending=False).head(25)
            pivot.to_csv(path_results.joinpath("family_FNR_pivot_top25_byYear.csv"), index=False)
            print("\n--- Top 25 families by sample size (FNR per variant) ---")
            print(pivot.to_string(index=False))

    if threshold_rows:
        thr_df = pd.DataFrame(threshold_rows)
        thr_df.to_csv(path_results.joinpath("threshold_sweep_MIT_byYear.csv"), index=False)
        print(f"\n--- Threshold sweep ---")
        print(thr_df.to_string(index=False))

    print("\nfamily_analysis.py done.")


if __name__ == "__main__":
    main()
