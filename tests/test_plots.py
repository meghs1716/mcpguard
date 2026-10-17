"""
tests/test_plots.py - Unit tests for eval/plots.py.
Verifies:
  1. Score loading and evaluation caching
  2. ROC curve generation and file output
  3. Reliability diagram generation and calibration curve computation
  4. Attack benchmark bar chart plotting
  5. Markdown results table formatting
  6. End-to-end execution of generate_plots_and_tables
"""

import json
import os
import pytest
from eval.plots import (
    evaluate_or_load_scores,
    generate_plots_and_tables,
    plot_attacks,
    plot_reliability,
    plot_roc,
    print_markdown_results,
)


@pytest.fixture
def sample_detector_items():
    """Fixture providing dummy scored detector items."""
    return [
        {"id": "p1", "label": "poisoned", "score": 0.95},
        {"id": "p2", "label": "poisoned", "score": 0.85},
        {"id": "p3", "label": "poisoned", "score": 0.40},
        {"id": "b1", "label": "benign", "score": 0.10},
        {"id": "b2", "label": "benign", "score": 0.05},
    ]


@pytest.fixture
def sample_attacks_data(tmp_path):
    """Fixture providing a mock attacks.json file."""
    attacks_file = tmp_path / "attacks.json"
    data = {
        "date": "2026-10-17T00:00:00Z",
        "model": "qwen/qwen3.8-27b",
        "thresholds": {"t_block": 0.55, "t_ask": 0.3},
        "num_runs": 5,
        "scenarios": [
            {
                "name": "mock_attack",
                "type": "attack",
                "description": "Mock attack scenario",
                "guard_off": {"success_rate": 1.0, "mean_latency": 4.5},
                "guard_on": {"success_rate": 0.0, "mean_latency": 5.2},
            },
            {
                "name": "mock_benign",
                "type": "benign",
                "description": "Mock benign task",
                "guard_off": {"success_rate": 1.0, "mean_latency": 2.1},
                "guard_on": {"success_rate": 1.0, "mean_latency": 3.8},
            },
        ],
    }
    with open(attacks_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(attacks_file)


def test_evaluate_or_load_scores_existing(tmp_path):
    """Verify evaluate_or_load_scores loads from disk when file exists."""
    results_dir = str(tmp_path / "results")
    os.makedirs(results_dir, exist_ok=True)
    scores_path = os.path.join(results_dir, "scores_baseline_test.json")

    mock_items = [
        {"id": "p1", "label": "poisoned", "score": 0.8},
        {"id": "b1", "label": "benign", "score": 0.1},
    ]
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(mock_items, f)

    metrics, loaded_items = evaluate_or_load_scores(
        detector="baseline",
        split="test",
        threshold=0.55,
        force_eval=False,
        results_dir=results_dir,
    )

    assert len(loaded_items) == 2
    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["auroc"] == 1.0


def test_plot_roc(tmp_path, sample_detector_items):
    """Verify plot_roc creates a valid PNG file."""
    output_png = str(tmp_path / "roc.png")
    plot_roc(sample_detector_items, sample_detector_items, output_png)

    assert os.path.exists(output_png)
    assert os.path.getsize(output_png) > 1000


def test_plot_reliability(tmp_path, sample_detector_items):
    """Verify plot_reliability creates a valid PNG file."""
    output_png = str(tmp_path / "reliability.png")
    plot_reliability(sample_detector_items, output_png, n_bins=3)

    assert os.path.exists(output_png)
    assert os.path.getsize(output_png) > 1000


def test_plot_attacks(tmp_path, sample_attacks_data):
    """Verify plot_attacks creates a valid bar chart PNG file."""
    output_png = str(tmp_path / "attacks.png")
    plot_attacks(sample_attacks_data, output_png)

    assert os.path.exists(output_png)
    assert os.path.getsize(output_png) > 1000


def test_print_markdown_results():
    """Verify print_markdown_results produces expected headers and metric values."""
    base_m = {"recall": 1.0, "fpr": 0.0, "auroc": 1.0, "precision": 1.0, "f1": 1.0}
    mod_m = {"recall": 0.95, "fpr": 0.02, "auroc": 0.99, "precision": 0.98, "f1": 0.965}
    attacks = {
        "scenarios": [
            {
                "name": "evil_test",
                "type": "attack",
                "guard_off": {"success_rate": 1.0, "mean_latency": 3.0},
                "guard_on": {"success_rate": 0.0, "mean_latency": 3.5},
            }
        ]
    }

    table = print_markdown_results(base_m, mod_m, attacks, t_block=0.55)

    assert "## MCPGuard Evaluation Results" in table
    assert "| **Baseline Scanner** | 100.0% | 0.0% | 1.0000 |" in table
    assert "| **Model Scanner** | 95.0% | 2.0% | 0.9900 |" in table
    assert "`evil_test`" in table
    assert "Attack Success (OFF / ON)" in table


def test_generate_plots_and_tables_end_to_end(tmp_path, sample_attacks_data):
    """Verify generate_plots_and_tables end-to-end execution."""
    cfg_file = str(tmp_path / "config.yaml")
    with open(cfg_file, "w", encoding="utf-8") as f:
        f.write("t_block: 0.55\nt_ask: 0.3\n")

    res_dir = str(tmp_path / "results")
    os.makedirs(res_dir, exist_ok=True)

    # Seed baseline and model scores
    dummy = [
        {"id": "p1", "label": "poisoned", "score": 0.9},
        {"id": "b1", "label": "benign", "score": 0.1},
    ]
    with open(os.path.join(res_dir, "scores_baseline_test.json"), "w", encoding="utf-8") as f:
        json.dump(dummy, f)
    with open(os.path.join(res_dir, "scores_model_test.json"), "w", encoding="utf-8") as f:
        json.dump(dummy, f)

    paths = generate_plots_and_tables(
        config_path=cfg_file,
        attacks_path=sample_attacks_data,
        results_dir=res_dir,
        force_eval=False,
    )

    assert os.path.exists(paths["roc"])
    assert os.path.exists(paths["reliability"])
    assert os.path.exists(paths["attacks"])
