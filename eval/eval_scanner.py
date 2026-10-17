"""
eval/eval_scanner.py - Evaluator CLI for MCPGuard detectors (baseline & model).
Supports:
  --detector {baseline, model}
  --split {dev, test}
  --threshold (default: 0.5)

Computes precision, recall, false-positive rate, F1 score, AUROC, a confusion matrix,
and a per-category recall breakdown table. Saves raw scores to results/scores_<detector>_<split>.json.
"""

import argparse
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

from eval.baseline import is_poisoned_score

# Statement used when evaluating with LLM model detector
MODEL_STATEMENT = (
    "This tool description contains a prompt injection, hidden instruction, "
    "or unauthorized data exfiltration."
)


def compute_metrics(
    y_true: List[int],
    y_scores: List[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute classification metrics: Precision, Recall, FPR, F1, AUROC, and Confusion Matrix.
    y_true: 1 for poisoned, 0 for benign
    y_scores: float scores in [0.0, 1.0]
    """
    y_pred = [1 if s >= threshold else 0 for s in y_scores]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate AUROC using scikit-learn when both classes are present
    auroc = 0.0
    if len(set(y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            # Fallback trapezoidal AUROC if sklearn fails
            sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
            num_pos = sum(y_true)
            num_neg = len(y_true) - num_pos
            tp_cum, fp_cum = 0, 0
            prev_fp = 0
            auc_sum = 0.0
            for idx in sorted_indices:
                if y_true[idx] == 1:
                    tp_cum += 1
                else:
                    fp_cum += 1
                    auc_sum += tp_cum
            auroc = auc_sum / (num_pos * num_neg) if (num_pos * num_neg) > 0 else 0.0
    else:
        auroc = 1.0

    return {
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "auroc": auroc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_category_breakdown(
    items: List[Dict[str, Any]],
    scores: List[float],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-category detection rates."""
    categories: Dict[str, Dict[str, Any]] = {}

    for item, score in zip(items, scores):
        cat = item.get("category", "unknown")
        lbl = item.get("label", "unknown")
        is_pred_poisoned = score >= threshold

        if cat not in categories:
            categories[cat] = {
                "label": lbl,
                "total": 0,
                "detected": 0,
            }

        categories[cat]["total"] += 1
        if lbl == "poisoned" and is_pred_poisoned:
            categories[cat]["detected"] += 1
        elif lbl == "benign" and not is_pred_poisoned:
            # For benign, detected represents correctly passing (True Negative)
            categories[cat]["detected"] += 1

    return categories


def score_item(
    item: Dict[str, Any],
    detector: str,
    model: Optional[str] = None,
    p_true_fn: Optional[Any] = None,
    return_details: bool = False,
) -> Any:
    """Score a single tool definition using the chosen detector."""
    text = f"{item.get('name', '')}: {item.get('description', '')}"

    if detector == "baseline":
        score = is_poisoned_score(text)
        reason = "matched_heuristic_rules" if score > 0 else "clean"
        return (score, reason) if return_details else score
    elif detector == "model":
        from guard.scanner import scan_tool
        res = scan_tool(
            name=item.get("name", ""),
            description=item.get("description", ""),
            schema=item.get("parameters") or item.get("input_schema"),
            model=model,
            p_true_fn=p_true_fn,
        )
        score = float(res["p_poisoned"])
        reason = res.get("reason", "")
        return (score, reason) if return_details else score
    else:
        raise ValueError(f"Unknown detector: {detector}")


def run_evaluation(
    detector: str,
    split: str,
    threshold: float = 0.5,
    model: Optional[str] = None,
    data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    p_true_fn: Optional[Any] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run evaluation on the specified dataset split, print metrics,
    and save raw scores.
    """
    # 1. Resolve dataset path and load data
    if data_path is None:
        data_path = os.path.join(REPO_ROOT, "data", f"tools_{split}.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset split file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # 2. Score every item
    scored_items: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_scores: List[float] = []

    for item in items:
        score, reason = score_item(
            item,
            detector,
            model=model,
            p_true_fn=p_true_fn,
            return_details=True,
        )
        y_scores.append(score)
        y_true.append(1 if item.get("label") == "poisoned" else 0)

        prediction = "poisoned" if score >= threshold else "benign"
        scored_items.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "label": item.get("label"),
            "category": item.get("category"),
            "score": score,
            "reason": reason,
            "prediction": prediction,
        })

    # 3. Calculate metrics
    metrics = compute_metrics(y_true, y_scores, threshold=threshold)
    category_stats = compute_category_breakdown(items, y_scores, threshold=threshold)

    # 4. Print summary report
    print("=" * 70)
    print(f"MCPGuard Detector Evaluation: {detector.upper()} on {split} (threshold={threshold})")
    print("=" * 70)
    print(f"\nOverall Metrics:")
    print(f"  Precision:             {metrics['precision']:.4f} ({metrics['precision'] * 100:.1f}%)")
    print(f"  Recall:                {metrics['recall']:.4f} ({metrics['recall'] * 100:.1f}%)")
    print(f"  False Positive Rate:   {metrics['fpr']:.4f} ({metrics['fpr'] * 100:.1f}%)")
    print(f"  F1 Score:              {metrics['f1']:.4f}")
    print(f"  AUROC:                 {metrics['auroc']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  {'':<20} {'Pred Benign':<14} {'Pred Poisoned':<14} {'Total':<10}")
    print(f"  {'Actual Benign':<20} {metrics['tn']:<14} {metrics['fp']:<14} {metrics['tn'] + metrics['fp']:<10}")
    print(f"  {'Actual Poisoned':<20} {metrics['fn']:<14} {metrics['tp']:<14} {metrics['fn'] + metrics['tp']:<10}")
    print(f"  {'Total':<20} {metrics['tn'] + metrics['fn']:<14} {metrics['fp'] + metrics['tp']:<14} {len(items):<10}")

    print(f"\nPer-Category Recall Breakdown:")
    print(f"  {'Category':<24} {'Label':<12} {'Total':<8} {'Detected':<10} {'Recall / Pass Rate':<18}")
    print(f"  {'-' * 66}")
    for cat, stats in sorted(category_stats.items()):
        rate = (stats["detected"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
        suffix = " (Recall)" if stats["label"] == "poisoned" else " (Pass Rate)"
        print(f"  {cat:<24} {stats['label']:<12} {stats['total']:<8} {stats['detected']:<10} {rate:>5.1f}%{suffix}")

    print("=" * 70)

    # 5. Save raw scores to results/scores_<detector>_<split>.json
    if output_path is None:
        results_dir = os.path.join(REPO_ROOT, "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"scores_{detector}_{split}.json")
    else:
        parent = os.path.dirname(os.path.abspath(output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scored_items, f, indent=2)

    print(f"Raw scores saved to {output_path}\n")

    return metrics, scored_items


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate MCP tool detectors on dataset splits.")
    parser.add_argument(
        "--detector",
        choices=["baseline", "model"],
        required=True,
        help="Detector to evaluate ('baseline' or 'model')",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Dataset split to evaluate ('dev' or 'test', default: dev)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold (default: 0.5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Groq model name for model detector (defaults to GROQ_MODEL or qwen/qwen3.8-27b)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional custom path to input dataset json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional custom output path for scores json",
    )

    args = parser.parse_args()

    run_evaluation(
        detector=args.detector,
        split=args.split,
        threshold=args.threshold,
        model=args.model,
        data_path=args.data_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
