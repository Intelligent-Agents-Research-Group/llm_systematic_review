#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, Any


DEFAULT_INPUT = "results/eval_results_lfm2.5_1.2b_instruct.json"


def safe_div(a: float, b: float) -> float:
    return float("nan") if b == 0 else a / b


def compute_agreement_metrics(conf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute agreement metrics from a binary confusion matrix.
    Expected keys:
      tp_count, fp_count, tn_count, fn_count, total
    """
    tp = conf["tp"]
    fp = conf["fp"]
    tn = conf["tn"]
    fn = conf["fn"]
    N = conf["total"]

    # Observed agreement
    Po = safe_div(tp + tn, N)

    # Marginals:
    p_true_pos = safe_div(tp + fn, N)
    p_true_neg = safe_div(tn + fp, N)

    p_pred_pos = safe_div(tp + fp, N)
    p_pred_neg = safe_div(tn + fn, N)

    # Cohen's kappa expected agreement
    Pe_kappa = p_true_pos * p_pred_pos + p_true_neg * p_pred_neg
    kappa = safe_div(Po - Pe_kappa, 1 - Pe_kappa)

    # PABAK
    pabak = 2 * Po - 1

    # Gwet AC1 (binary)
    pi = 0.5 * (p_true_pos + p_pred_pos)
    Pe_ac1 = 2 * pi * (1 - pi)
    ac1 = safe_div(Po - Pe_ac1, 1 - Pe_ac1)

    return {
        "N": N,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        },
        "observed_agreement_Po": Po,
        "expected_agreement_kappa_Pe": Pe_kappa,
        "cohen_kappa": kappa,
        "pabak": pabak,
        "gwet_ac1_expected_agreement_Pe": Pe_ac1,
        "gwet_ac1": ac1,
        "marginals": {
            "p_true_pos": p_true_pos,
            "p_true_neg": p_true_neg,
            "p_pred_pos": p_pred_pos,
            "p_pred_neg": p_pred_neg,
            "pi_bar": pi
        }
    }


def build_default_output_path(input_path: str) -> str:
    p = Path(input_path)
    return str(p.with_name(p.stem + "_agreement_metrics.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Compute Cohen's kappa, PABAK, and Gwet AC1 from TP/FP/TN/FN JSON."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Path to input JSON (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON (default: <input_stem>_agreement_metrics.json)"
    )
    parser.add_argument(
        "--metrics-key",
        type=str,
        default="metrics",
        help="Top-level key where confusion matrix fields live (default: metrics)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print output JSON"
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or build_default_output_path(input_path)

    with open(input_path, "r") as f:
        payload = json.load(f)

    if args.metrics_key not in payload:
        raise KeyError(
            f"metrics_key='{args.metrics_key}' not found in JSON. "
            f"Available keys: {list(payload.keys())}"
        )

    conf = payload[args.metrics_key]
    required = ["total", "tn", "tp", "fn", "fp"] # "tp_count", "fp_count", "tn_count", "fn_count", 
    missing = [k for k in required if k not in conf]
    if missing:
        raise KeyError(f"Missing required fields in JSON['{args.metrics_key}']: {missing}")

    results = compute_agreement_metrics(conf)

    with open(output_path, "w") as f:
        if args.pretty:
            json.dump(results, f, indent=2)
        else:
            json.dump(results, f, indent=2)

    # Console summary
    print("=== Agreement Metrics Computed ===")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print()
    print(f"Cohen's kappa : {results['cohen_kappa']:.6f}")
    print(f"PABAK         : {results['pabak']:.6f}")
    print(f"Gwet AC1      : {results['gwet_ac1']:.6f}")


if __name__ == "__main__":
    main()