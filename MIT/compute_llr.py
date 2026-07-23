"""Compute character-bigram LLR for T24 benign domains.
Reference benign = top 10% by avg_rank (real ground truth).
"""
import string
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).parent.parent))
from utility.path import path_train_data

ALPHA = "<" + string.ascii_lowercase + string.digits + "-" + ">"
A2I = {c: i for i, c in enumerate(ALPHA)}
N = len(ALPHA)
REF_BENIGN_FRAC = 0.10
DGA_REF_SIZE = 200_000


def sld(d: str) -> str:
    return d.split(".")[0].lower() if d else ""


def bigrams(s: str):
    s = "<" + "".join(c if c in ALPHA[1:-1] else "" for c in s.lower()) + ">"
    return [(s[i], s[i + 1]) for i in range(len(s) - 1)]


def build_bigram(domains, alpha: float = 1.0) -> np.ndarray:
    cnt = np.zeros((N, N), dtype=np.float64)
    for d in domains:
        for a, b in bigrams(sld(d)):
            if a in A2I and b in A2I:
                cnt[A2I[a], A2I[b]] += 1
    cnt += alpha
    return np.log(cnt / cnt.sum(axis=1, keepdims=True))


def score(domain: str, M_D: np.ndarray, M_B: np.ndarray) -> float:
    bg = bigrams(sld(domain))
    if not bg:
        return 0.0
    s = 0.0
    n = 0
    for a, b in bg:
        if a in A2I and b in A2I:
            s += M_D[A2I[a], A2I[b]] - M_B[A2I[a], A2I[b]]
            n += 1
    return s / max(n, 1)


def main() -> None:
    t0 = time.time()
    benign_path = path_train_data.joinpath("T24_benign_30days.parquet")
    dga_path = path_train_data.joinpath("T24_dga_30days_train.parquet")

    print(f"Loading benign: {benign_path}")
    benign = pq.read_table(benign_path).to_pandas()
    print(f"  benign rows: {len(benign):,}")
    print(f"Loading DGA: {dga_path}")
    dga = pq.read_table(dga_path).to_pandas()
    print(f"  DGA rows: {len(dga):,}")

    ref_size = int(len(benign) * REF_BENIGN_FRAC)
    print(f"Building benign reference (top {REF_BENIGN_FRAC:.0%} = {ref_size:,} by avg_rank)...")
    ref_benign = benign.nsmallest(ref_size, "avg_rank")["domain"].tolist()
    M_B = build_bigram(ref_benign)

    print(f"Building DGA reference (sample of {DGA_REF_SIZE:,})...")
    dga_ref = dga["domain"].sample(min(DGA_REF_SIZE, len(dga)), random_state=42).tolist()
    M_D = build_bigram(dga_ref)

    eval_b = pd.Series(ref_benign).sample(min(5000, len(ref_benign)), random_state=1).map(lambda d: score(d, M_D, M_B))
    eval_d = pd.Series(dga_ref).sample(min(5000, len(dga_ref)), random_state=1).map(lambda d: score(d, M_D, M_B))
    tau = float((eval_b.quantile(0.95) + eval_d.quantile(0.05)) / 2)
    print(f"Calibration: benign p95={eval_b.quantile(0.95):+.3f}, DGA p5={eval_d.quantile(0.05):+.3f}, tau={tau:+.3f}")

    print("Scoring full benign set...")
    benign["llr"] = benign["domain"].map(lambda d: score(d, M_D, M_B))
    cont_rate = (benign["llr"] > tau).mean()
    print(f"Contamination rate (LLR > tau): {cont_rate:.4%}")
    print(f"LLR distribution: median={benign['llr'].median():+.3f}, p95={benign['llr'].quantile(0.95):+.3f}")

    # out_path = path_train_data.joinpath("T24_benign_30days_with_llr.parquet")
    # benign.to_parquet(out_path, index=False)
    # print(f"Saved {len(benign):,} rows with LLR to {out_path}")

    # thr_path = path_train_data.joinpath("llr_threshold.txt")
    # thr_path.write_text(f"{tau}\n")
    # print(f"Saved threshold to {thr_path}")
    # print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
