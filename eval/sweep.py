"""
eval/sweep.py - Threshold sweep evaluation for MCPGuard block threshold.
Sweeps block threshold from 0.1 to 0.9 on the DEV split using saved scores,
prints recall and false-positive rate at each value, writes results/sweep.csv,
and recommends the threshold with the highest recall at false-positive rate <= 5%.
Provides an option (--update-config / --write-config) to persist the recommended
threshold into config.yaml.
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Configure utf-8 console output for Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compute_metrics_at_threshold(
    scored_items: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    """Compute TPR (Recall), FPR, Precision, and F1 at a specific threshold."""
    y_true = [1 if item.get("label") == "poisoned" else 0 for item in scored_items]
    y_scores = [float(item.get("score", 0.0)) for item in scored_items]
    y_pred = [1 if s >= threshold else 0 for s in y_scores]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": round(threshold, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def run_sweep(
    scored_items: List[Dict[str, Any]],
    start: float = 0.1,
    stop: float = 0.9,
    step: float = 0.05,
) -> List[Dict[str, Any]]:
    """Generate threshold sweep rows from start to stop."""
    sweep_rows: List[Dict[str, Any]] = []

    # Generate thresholds rounded to avoid floating point drift
    num_steps = int(round((stop - start) / step)) + 1
    thresholds = [round(start + i * step, 4) for i in range(num_steps)]

    for t in thresholds:
        metrics = compute_metrics_at_threshold(scored_items, t)
        sweep_rows.append(metrics)

    return sweep_rows


def select_recommended_threshold(
    sweep_rows: List[Dict[str, Any]],
    max_fpr: float = 0.05,
) -> Dict[str, Any]:
    """
    Recommend the threshold with the highest recall at false-positive rate <= max_fpr.
    Tie-breaking: choose the higher threshold for conservative safety margin.
    """
    eligible = [r for r in sweep_rows if r["fpr"] <= max_fpr]
    if not eligible:
        # Fallback to row with lowest FPR if none meet criteria
        return min(sweep_rows, key=lambda r: r["fpr"])

    # Highest recall first; higher threshold second
    eligible.sort(key=lambda r: (r["recall"], r["threshold"]), reverse=True)
    return eligible[0]


def write_sweep_csv(
    sweep_rows: List[Dict[str, Any]],
    csv_path: str,
) -> None:
    """Write sweep metrics to CSV file."""
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    fieldnames = ["threshold", "recall", "fpr", "precision", "f1", "tp", "fp", "tn", "fn"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_rows)


def update_config_file(config_path: str, t_block: float) -> None:
    """Update t_block in config.yaml while preserving other settings."""
    import yaml

    data = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}

    data["t_block"] = round(float(t_block), 4)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    """CLI Entrypoint for threshold sweep."""
    parser = argparse.ArgumentParser(description="Sweep block threshold on DEV split.")
    parser.add_argument(
        "--scores-path",
        type=str,
        default=None,
        help="Path to saved scores JSON (defaults to results/scores_baseline_dev.json)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output sweep CSV (defaults to results/sweep.csv)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.1,
        help="Start threshold (default: 0.1)",
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=0.9,
        help="Stop threshold (default: 0.9)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Threshold step increment (default: 0.05)",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.05,
        help="Maximum false-positive rate constraint (default: 0.05)",
    )
    parser.add_argument(
        "--update-config",
        "--write-config",
        action="store_true",
        dest="update_config",
        help="Write recommended t_block value into config.yaml",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config.yaml (defaults to config.yaml in repo root)",
    )

    args = parser.parse_args()

    # 1. Resolve paths
    scores_path = args.scores_path or os.path.join(REPO_ROOT, "results", "scores_baseline_dev.json")
    output_csv = args.output_csv or os.path.join(REPO_ROOT, "results", "sweep.csv")
    config_path = args.config_path or os.path.join(REPO_ROOT, "config.yaml")

    if not os.path.exists(scores_path):
        print(f"Scores file not found at {scores_path}.")
        print("Running eval/eval_scanner.py to generate baseline DEV scores first...")
        from eval.eval_scanner import run_evaluation
        run_evaluation(detector="baseline", split="dev", threshold=0.5)

    with open(scores_path, "r", encoding="utf-8") as f:
        scored_items = json.load(f)

    # 2. Run sweep
    sweep_rows = run_sweep(
        scored_items=scored_items,
        start=args.start,
        stop=args.stop,
        step=args.step,
    )

    # 3. Print sweep table
    print("=" * 72)
    print("MCPGuard Threshold Sweep on DEV Split")
    print("=" * 72)
    print(f"Source Scores:    {scores_path}")
    print(f"Threshold Range:  {args.start:.2f} to {args.stop:.2f} (step={args.step:.2f})")
    print("-" * 72)
    print(f"{'Threshold':<12} {'Recall (TPR)':<16} {'FPR':<16} {'Precision':<14} {'F1':<10}")
    print("-" * 72)
    for r in sweep_rows:
        print(
            f"{r['threshold']:>9.2f}   "
            f"{r['recall'] * 100:>11.1f}%   "
            f"{r['fpr'] * 100:>11.1f}%   "
            f"{r['precision'] * 100:>10.1f}%   "
            f"{r['f1']:>8.4f}"
        )
    print("=" * 72)

    # 4. Save results/sweep.csv
    write_sweep_csv(sweep_rows, output_csv)
    print(f"Sweep results written to: {output_csv}\n")

    # 5. Determine recommended threshold
    recommended = select_recommended_threshold(sweep_rows, max_fpr=args.max_fpr)
    rec_t = recommended["threshold"]
    rec_recall = recommended["recall"] * 100
    rec_fpr = recommended["fpr"] * 100

    print("Recommendation:")
    print(f"  Recommended Block Threshold: t_block = {rec_t:.2f}")
    print(f"  Performance at t_block:       Recall = {rec_recall:.1f}%, FPR = {rec_fpr:.1f}%\n")

    # 6. Optionally write to config.yaml
    if args.update_config:
        update_config_file(config_path, rec_t)
        print(f"Successfully updated {config_path} with t_block: {rec_t:.2f}\n")


if __name__ == "__main__":
    main()
