#!/usr/bin/env python3
"""
multi_rater_agreement.py

Computes multi-rater agreement metrics from your Phase-1 CSV:

- Fleiss' kappa + bootstrap CI
- Multi-rater Gwet AC1 + bootstrap CI
- Multi-rater Gwet AC2 (ordinal weighted) + bootstrap CI (optional)

Designed for your schema:
  prompt, human_decision, pred_temp_0.1, pred_temp_0.4, pred_temp_0.8

Default input:
  /mnt/data/phase1_predictions_20260127_213206.csv
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_INPUT = "results/phase1_results_lfm2.5/phase1_predictions_20260127_213206.csv"
DEFAULT_RATER_COLS = ["human_decision", "pred_temp_0.1", "pred_temp_0.4", "pred_temp_0.8"]


def _safe_div(a: float, b: float) -> float:
    return float("nan") if b == 0 else a / b


def fleiss_kappa_from_counts(counts: np.ndarray):
    """
    Fleiss' kappa from N x k count matrix.
    counts[i, j] = number of raters who assigned item i to category j.
    """
    counts = np.asarray(counts, dtype=float)
    N, k = counts.shape

    n_i = counts.sum(axis=1)  # raters per item
    # per-item agreement
    P_i = (counts * (counts - 1)).sum(axis=1) / (n_i * (n_i - 1))
    Pbar = float(np.mean(P_i))

    # overall category proportions
    p_j = counts.sum(axis=0) / np.sum(n_i)
    Pe = float(np.sum(p_j ** 2))

    kappa = _safe_div(Pbar - Pe, 1.0 - Pe)
    return kappa, Pbar, Pe, p_j


def bootstrap_ci(stat_fn, data, B=2000, alpha=0.05, seed=13):
    """
    Nonparametric bootstrap CI over items.
    """
    rng = np.random.default_rng(seed)
    N = len(data)
    stats = np.empty(B, dtype=float)

    for b in range(B):
        idx = rng.integers(0, N, size=N)
        stats[b] = stat_fn(data[idx])

    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def gwet_ac1_nominal_multi(labels: np.ndarray):
    """
    Multi-rater Gwet AC1 (nominal categories).

    labels: N x n_raters, integer categories 0..k-1
    """
    A = np.asarray(labels, dtype=int)
    N, n = A.shape
    k = int(A.max()) + 1

    # Observed agreement Ao (mean per-item pairwise agreement)
    Ao_items = np.empty(N, dtype=float)
    for i in range(N):
        li = A[i]
        counts = np.bincount(li, minlength=k)
        Ao_items[i] = (counts * (counts - 1)).sum() / (n * (n - 1))
    Ao = float(Ao_items.mean())

    # category proportions across all ratings
    p = np.bincount(A.ravel(), minlength=k) / (N * n)

    # Chance agreement for AC1 nominal
    # Ae = sum_j p_j(1-p_j)/(k-1)
    Ae = float(np.sum(p * (1 - p)) / (k - 1)) if k > 1 else 0.0

    AC1 = _safe_div(Ao - Ae, 1 - Ae)
    return AC1, Ao, Ae, p


def _weight_matrix(k: int, kind="quadratic"):
    if k <= 1:
        return np.ones((k, k))
    W = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if kind == "linear":
                W[i, j] = 1.0 - abs(i - j) / (k - 1)
            elif kind == "quadratic":
                W[i, j] = 1.0 - ((i - j) ** 2) / ((k - 1) ** 2)
            else:
                raise ValueError("weights must be 'linear' or 'quadratic'")
    return W


def gwet_ac2_multi(labels: np.ndarray, weights="quadratic"):
    """
    Multi-rater Gwet AC2 (ordinal weighted agreement).

    NOTE: AC2 is only meaningful for ordinal labels with k>=3.
    For binary, AC2 becomes less informative; still computed for completeness.

    labels: N x n_raters integer categories
    """
    A = np.asarray(labels, dtype=int)
    N, n = A.shape
    k = int(A.max()) + 1

    W = _weight_matrix(k, kind=weights)

    # Observed weighted agreement Ao
    Ao_items = np.empty(N, dtype=float)
    for i in range(N):
        li = A[i]
        counts = np.bincount(li, minlength=k)
        p_i = counts / n
        Ao_items[i] = float(np.sum(W * np.outer(p_i, p_i)))
    Ao = float(Ao_items.mean())

    # Expected agreement (unadjusted weighted)
    p = np.bincount(A.ravel(), minlength=k) / (N * n)
    Ae = float(np.sum(W * np.outer(p, p)))

    AC2 = _safe_div(Ao - Ae, 1 - Ae)
    return AC2, Ao, Ae, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON report path.")
    parser.add_argument("--raters", nargs="+", default=DEFAULT_RATER_COLS, help="Rater columns.")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap samples for CI.")
    parser.add_argument("--seed", type=int, default=13, help="RNG seed for bootstrap.")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_name(inp.stem + "_multi_rater_agreement.json")

    df = pd.read_csv(inp)

    # Validate rater columns
    for c in args.raters:
        if c not in df.columns:
            raise KeyError(f"Missing rater column '{c}'. CSV cols={list(df.columns)}")

    labels = df[args.raters].to_numpy(dtype=int)
    N, n_raters = labels.shape
    k = int(labels.max()) + 1

    # Fleiss counts matrix for Fleiss kappa
    # For each item: counts per category
    counts = np.zeros((N, k), dtype=int)
    for i in range(N):
        counts[i] = np.bincount(labels[i], minlength=k)

    fleiss_k, Pbar, Pe, pj = fleiss_kappa_from_counts(counts)

    # Bootstrap CI Fleiss
    fleiss_ci = bootstrap_ci(lambda x: fleiss_kappa_from_counts(x)[0], counts, B=args.bootstrap, seed=args.seed)

    # Gwet AC1 + CI
    ac1, Ao1, Ae1, p_ac1 = gwet_ac1_nominal_multi(labels)
    ac1_ci = bootstrap_ci(lambda x: gwet_ac1_nominal_multi(x)[0], labels, B=args.bootstrap, seed=args.seed)

    # Gwet AC2 + CI (quadratic weights by default)
    ac2, Ao2, Ae2, p_ac2 = gwet_ac2_multi(labels, weights="quadratic")
    ac2_ci = bootstrap_ci(lambda x: gwet_ac2_multi(x, weights="quadratic")[0], labels, B=args.bootstrap, seed=args.seed)

    report = {
        "input_csv": str(inp),
        "raters": args.raters,
        "N_items": int(N),
        "n_raters": int(n_raters),
        "n_categories": int(k),
        "fleiss_kappa": {
            "value": float(fleiss_k),
            "bootstrap_ci_95": [float(fleiss_ci[0]), float(fleiss_ci[1])],
            "Pbar": float(Pbar),
            "Pe": float(Pe),
            "category_proportions": pj.tolist(),
        },
        "gwet_ac1": {
            "value": float(ac1),
            "bootstrap_ci_95": [float(ac1_ci[0]), float(ac1_ci[1])],
            "Ao": float(Ao1),
            "Ae": float(Ae1),
            "category_proportions": p_ac1.tolist(),
        },
        "gwet_ac2_quadratic": {
            "value": float(ac2),
            "bootstrap_ci_95": [float(ac2_ci[0]), float(ac2_ci[1])],
            "Ao": float(Ao2),
            "Ae": float(Ae2),
            "category_proportions": p_ac2.tolist(),
            "note": "AC2 is only meaningful for ordinal labels (k>=3). For binary labels, interpret cautiously."
        }
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("=== Multi-rater Agreement Report ===")
    print(f"Input : {inp}")
    print(f"Output: {out}")
    print()
    print(f"Fleiss kappa : {report['fleiss_kappa']['value']:.6f}  CI95={report['fleiss_kappa']['bootstrap_ci_95']}")
    print(f"Gwet AC1     : {report['gwet_ac1']['value']:.6f}  CI95={report['gwet_ac1']['bootstrap_ci_95']}")
    print(f"Gwet AC2     : {report['gwet_ac2_quadratic']['value']:.6f}  CI95={report['gwet_ac2_quadratic']['bootstrap_ci_95']}")


if __name__ == "__main__":
    main()
