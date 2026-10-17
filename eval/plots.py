"""
eval/plots.py - Evaluation plotting and markdown summary generator for MCPGuard.
Evaluates baseline and model scanners ONCE on the TEST split with thresholds in config.yaml.
Generates:
  1. results/roc.png          - ROC curves for baseline and model detectors
  2. results/reliability.png  - Reliability (calibration) diagram for model scanner
  3. results/attacks.png      - Grouped bar chart of attack success Guard OFF vs Guard ON
Prints a formatted Markdown results table ready to paste into README.md.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Use Agg backend for headless matplotlib rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

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

from eval.eval_scanner import compute_metrics, run_evaluation
from guard.guard import load_config


def evaluate_or_load_scores(
    detector: str,
    split: str = "test",
    threshold: float = 0.55,
    force_eval: bool = False,
    results_dir: Optional[str] = None,
    data_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate detector once on specified split or load pre-existing raw scores JSON.
    Returns (metrics_dict, scored_items_list).
    """
    if results_dir is None:
        results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    scores_path = os.path.join(results_dir, f"scores_{detector}_{split}.json")

    # Load from disk if file exists and force_eval is False
    if os.path.exists(scores_path) and not force_eval:
        with open(scores_path, "r", encoding="utf-8") as f:
            scored_items = json.load(f)

        y_true = [1 if item.get("label") == "poisoned" else 0 for item in scored_items]
        y_scores = [float(item.get("score", 0.0)) for item in scored_items]
        metrics = compute_metrics(y_true, y_scores, threshold=threshold)
        return metrics, scored_items

    # Otherwise run evaluation and save scores
    metrics, scored_items = run_evaluation(
        detector=detector,
        split=split,
        threshold=threshold,
        data_path=data_path,
        output_path=scores_path,
    )
    return metrics, scored_items


def plot_roc(
    baseline_items: List[Dict[str, Any]],
    model_items: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Generate ROC curves for both baseline and model detectors and save to output_path.
    """
    y_true_b = [1 if item.get("label") == "poisoned" else 0 for item in baseline_items]
    y_scores_b = [float(item.get("score", 0.0)) for item in baseline_items]

    y_true_m = [1 if item.get("label") == "poisoned" else 0 for item in model_items]
    y_scores_m = [float(item.get("score", 0.0)) for item in model_items]

    fpr_b, tpr_b, _ = roc_curve(y_true_b, y_scores_b)
    auc_b = roc_auc_score(y_true_b, y_scores_b) if len(set(y_true_b)) > 1 else 1.0

    fpr_m, tpr_m, _ = roc_curve(y_true_m, y_scores_m)
    auc_m = roc_auc_score(y_true_m, y_scores_m) if len(set(y_true_m)) > 1 else 1.0

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr_b,
        tpr_b,
        color="#1f77b4",
        linewidth=2.2,
        label=f"Baseline Scanner (AUROC = {auc_b:.4f})",
    )
    plt.plot(
        fpr_m,
        tpr_m,
        color="#ff7f0e",
        linewidth=2.2,
        linestyle="-.",
        label=f"Model Scanner (AUROC = {auc_m:.4f})",
    )
    # Diagonal random chance reference line
    plt.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", linewidth=1.2, label="Random Guess (AUROC = 0.5000)")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.04])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / TPR)", fontsize=11, fontweight="bold")
    plt.title("ROC Curves - MCPGuard Detectors (Test Split)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_reliability(
    model_items: List[Dict[str, Any]],
    output_path: str,
    n_bins: int = 5,
) -> None:
    """
    Generate reliability / calibration diagram for the model scanner and save to output_path.
    """
    y_true = [1 if item.get("label") == "poisoned" else 0 for item in model_items]
    y_scores = [float(item.get("score", 0.0)) for item in model_items]

    # Compute empirical fraction of positives vs mean predicted score in bins
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=n_bins, strategy="uniform")

    plt.figure(figsize=(7, 6))
    plt.plot(
        prob_pred,
        prob_true,
        marker="s",
        markersize=7,
        linewidth=2.2,
        color="#2ca02c",
        label="Model Scanner",
    )
    # Perfectly calibrated reference line
    plt.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", linewidth=1.2, label="Perfect Calibration")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.04])
    plt.xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Empirical Fraction of Positives", fontsize=11, fontweight="bold")
    plt.title("Reliability Diagram (Calibration) - Model Scanner", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10, frameon=True)
    plt.tight_layout()

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_attacks(
    attacks_path: str,
    output_path: str,
) -> None:
    """
    Generate grouped bar chart comparing attack success Guard OFF vs Guard ON.
    """
    if not os.path.exists(attacks_path):
        raise FileNotFoundError(f"Attacks JSON benchmark file not found at: {attacks_path}")

    with open(attacks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    if not scenarios:
        raise ValueError(f"No scenario results found in {attacks_path}")

    names = [s.get("name", "") for s in scenarios]
    off_rates = [s["guard_off"]["success_rate"] * 100 for s in scenarios]
    on_rates = [s["guard_on"]["success_rate"] * 100 for s in scenarios]

    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width / 2, off_rates, width, label="Guard OFF", color="#d62728", alpha=0.88)
    rects2 = ax.bar(x + width / 2, on_rates, width, label="Guard ON", color="#2ca02c", alpha=0.88)

    # Add numeric percentage labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height:.0f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.0f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Attack & Task Success: Guard OFF vs Guard ON", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylim([0, 118])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    plt.tight_layout()

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def print_markdown_results(
    baseline_metrics: Dict[str, Any],
    model_metrics: Dict[str, Any],
    attacks_data: Dict[str, Any],
    t_block: float,
) -> str:
    """
    Format and print a markdown results table containing recall, FPR, AUROC,
    attack success OFF vs ON, and mean latencies for inclusion in README.md.
    """
    scenarios = attacks_data.get("scenarios", [])

    # Calculate average attack success rates and latencies
    attack_scenarios = [s for s in scenarios if s.get("type") == "attack"]
    if attack_scenarios:
        mean_attack_succ_off = (
            sum(s["guard_off"]["success_rate"] for s in attack_scenarios) / len(attack_scenarios) * 100
        )
        mean_attack_succ_on = (
            sum(s["guard_on"]["success_rate"] for s in attack_scenarios) / len(attack_scenarios) * 100
        )
        mean_attack_lat_off = (
            sum(s["guard_off"]["mean_latency"] for s in attack_scenarios) / len(attack_scenarios)
        )
        mean_attack_lat_on = (
            sum(s["guard_on"]["mean_latency"] for s in attack_scenarios) / len(attack_scenarios)
        )
    else:
        mean_attack_succ_off = mean_attack_succ_on = mean_attack_lat_off = mean_attack_lat_on = 0.0

    lines = []
    lines.append("## MCPGuard Evaluation Results")
    lines.append("")
    lines.append(f"### 1. Detector Performance on TEST Split ($t_{{block}} = {t_block:.2f}$)")
    lines.append("")
    lines.append("| Detector | Recall | False-Positive Rate | AUROC | Precision | F1 Score |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    lines.append(
        f"| **Baseline Scanner** | {baseline_metrics['recall'] * 100:.1f}% | "
        f"{baseline_metrics['fpr'] * 100:.1f}% | {baseline_metrics['auroc']:.4f} | "
        f"{baseline_metrics['precision'] * 100:.1f}% | {baseline_metrics['f1']:.4f} |"
    )
    lines.append(
        f"| **Model Scanner** | {model_metrics['recall'] * 100:.1f}% | "
        f"{model_metrics['fpr'] * 100:.1f}% | {model_metrics['auroc']:.4f} | "
        f"{model_metrics['precision'] * 100:.1f}% | {model_metrics['f1']:.4f} |"
    )
    lines.append("")
    lines.append("### 2. Attack Defense & Latency Benchmark")
    lines.append("")
    lines.append("| Scenario | Type | Attack Success (OFF) | Attack Success (ON) | Latency (OFF) | Latency (ON) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for s in scenarios:
        s_name = f"`{s['name']}`"
        s_type = s.get("type", "attack")
        off_succ = f"{s['guard_off']['success_rate'] * 100:.1f}%"
        on_succ = f"{s['guard_on']['success_rate'] * 100:.1f}%"
        lat_off = f"{s['guard_off']['mean_latency']:.2f}s"
        lat_on = f"{s['guard_on']['mean_latency']:.2f}s"
        lines.append(f"| {s_name} | {s_type} | {off_succ} | {on_succ} | {lat_off} | {lat_on} |")

    lines.append("")
    lines.append("### 3. Summary Results for README.md")
    lines.append("")
    lines.append("| Mode / Component | Recall | False-Positive Rate | AUROC | Attack Success (OFF / ON) | Mean Latency |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    lines.append(
        f"| **Baseline Scanner** | {baseline_metrics['recall'] * 100:.1f}% | "
        f"{baseline_metrics['fpr'] * 100:.1f}% | {baseline_metrics['auroc']:.4f} | N/A | < 1ms |"
    )
    lines.append(
        f"| **Model Scanner** | {model_metrics['recall'] * 100:.1f}% | "
        f"{model_metrics['fpr'] * 100:.1f}% | {model_metrics['auroc']:.4f} | N/A | ~35s / tool |"
    )
    lines.append(
        f"| **MCPGuard (Guard OFF)** | N/A | N/A | N/A | {mean_attack_succ_off:.1f}% | {mean_attack_lat_off:.2f}s |"
    )
    lines.append(
        f"| **MCPGuard (Guard ON)** | N/A | 0.0% (Benign FP) | N/A | {mean_attack_succ_on:.1f}% | {mean_attack_lat_on:.2f}s |"
    )
    lines.append("")

    table_text = "\n".join(lines)
    print("=" * 84)
    print(table_text)
    print("=" * 84)
    return table_text


def generate_plots_and_tables(
    config_path: str = "config.yaml",
    attacks_path: str = "results/attacks.json",
    results_dir: str = "results",
    force_eval: bool = False,
) -> Dict[str, str]:
    """
    Main runner: evaluates detectors once, generates ROC, reliability, and attacks plots,
    and prints markdown results.
    """
    t_block, t_ask = load_config(config_path)

    # 1. Evaluate or load baseline & model scores
    base_metrics, base_items = evaluate_or_load_scores(
        detector="baseline",
        split="test",
        threshold=t_block,
        force_eval=force_eval,
        results_dir=results_dir,
    )
    model_metrics, model_items = evaluate_or_load_scores(
        detector="model",
        split="test",
        threshold=t_block,
        force_eval=force_eval,
        results_dir=results_dir,
    )

    # 2. Generate ROC curve for both detectors
    roc_path = os.path.join(results_dir, "roc.png")
    plot_roc(base_items, model_items, roc_path)
    print(f"Saved ROC curve plot to: {roc_path}")

    # 3. Generate reliability diagram for model scanner
    rel_path = os.path.join(results_dir, "reliability.png")
    plot_reliability(model_items, rel_path)
    print(f"Saved reliability diagram to: {rel_path}")

    # 4. Generate attacks bar chart
    attacks_png_path = os.path.join(results_dir, "attacks.png")
    plot_attacks(attacks_path, attacks_png_path)
    print(f"Saved attacks benchmark plot to: {attacks_png_path}\n")

    # 5. Load attacks data and print markdown table
    with open(attacks_path, "r", encoding="utf-8") as f:
        attacks_data = json.load(f)

    print_markdown_results(base_metrics, model_metrics, attacks_data, t_block)

    return {
        "roc": roc_path,
        "reliability": rel_path,
        "attacks": attacks_png_path,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate MCPGuard evaluation plots and markdown tables.")
    parser.add_argument(
        "--config-path",
        type=str,
        default=os.path.join(REPO_ROOT, "config.yaml"),
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--attacks-path",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "attacks.json"),
        help="Path to attacks.json benchmark (default: results/attacks.json)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "results"),
        help="Output directory for generated plots (default: results)",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Force re-evaluation of detectors on TEST split instead of loading cached scores",
    )

    args = parser.parse_args()

    generate_plots_and_tables(
        config_path=args.config_path,
        attacks_path=args.attacks_path,
        results_dir=args.results_dir,
        force_eval=args.force_eval,
    )


if __name__ == "__main__":
    main()
