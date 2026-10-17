"""
tests/test_sweep.py - Unit tests for eval/sweep.py.
Verifies threshold sweep generation, metrics computation, CSV output,
recommended threshold selection (highest recall at FPR <= 5%),
and config.yaml updating.
"""

import csv
import json
import os
import pytest
from eval.sweep import (
    compute_metrics_at_threshold,
    run_sweep,
    select_recommended_threshold,
    write_sweep_csv,
    update_config_file,
)


def test_compute_metrics_at_threshold():
    """Verify metrics calculation at a specific threshold."""
    items = [
        {"score": 0.8, "label": "poisoned"},
        {"score": 0.4, "label": "poisoned"},
        {"score": 0.1, "label": "benign"},
        {"score": 0.6, "label": "benign"},
    ]
    # At t = 0.5:
    # Pred pos: idx 0 (TP), idx 3 (FP)
    # Pred neg: idx 1 (FN), idx 2 (TN)
    metrics = compute_metrics_at_threshold(items, threshold=0.5)
    assert metrics["threshold"] == 0.5
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["fpr"] == 0.5


def test_run_sweep_range_and_steps():
    """Verify run_sweep covers the expected threshold range."""
    items = [
        {"score": 0.7, "label": "poisoned"},
        {"score": 0.2, "label": "benign"},
    ]
    rows = run_sweep(items, start=0.1, stop=0.9, step=0.1)
    thresholds = [r["threshold"] for r in rows]

    assert len(rows) == 9
    assert thresholds[0] == 0.1
    assert thresholds[-1] == 0.9


def test_select_recommended_threshold():
    """Verify recommendation picks highest recall with FPR <= 0.05."""
    rows = [
        {"threshold": 0.2, "recall": 0.95, "fpr": 0.10},  # FPR too high (> 0.05)
        {"threshold": 0.4, "recall": 0.90, "fpr": 0.04},  # Valid
        {"threshold": 0.5, "recall": 0.80, "fpr": 0.02},  # Valid, but recall lower
        {"threshold": 0.6, "recall": 0.90, "fpr": 0.03},  # Valid, equal recall to 0.4 but higher threshold
    ]
    rec = select_recommended_threshold(rows, max_fpr=0.05)
    # Should pick threshold 0.6 (highest recall 0.90 with higher threshold tie-break)
    assert rec["threshold"] == 0.6
    assert rec["recall"] == 0.90
    assert rec["fpr"] <= 0.05


def test_write_sweep_csv(tmp_path):
    """Verify write_sweep_csv creates a valid CSV file with expected columns."""
    csv_file = str(tmp_path / "sweep.csv")
    sample_rows = [
        {"threshold": 0.5, "recall": 0.95, "fpr": 0.0, "precision": 1.0, "f1": 0.97, "tp": 19, "fp": 0, "tn": 10, "fn": 1},
    ]

    write_sweep_csv(sample_rows, csv_file)
    assert os.path.exists(csv_file)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert float(rows[0]["threshold"]) == 0.5
    assert float(rows[0]["recall"]) == 0.95
    assert float(rows[0]["fpr"]) == 0.0


def test_update_config_file(tmp_path):
    """Verify update_config_file updates t_block while preserving existing settings."""
    cfg_file = str(tmp_path / "config.yaml")
    with open(cfg_file, "w", encoding="utf-8") as f:
        f.write("t_block: 0.7\nt_ask: 0.35\ncustom_key: test\n")

    update_config_file(cfg_file, 0.55)

    import yaml
    with open(cfg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["t_block"] == 0.55
    assert data["t_ask"] == 0.35
    assert data["custom_key"] == "test"
