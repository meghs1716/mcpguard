"""
tests/test_build_dataset.py - Unit tests for scripts/build_dataset.py.
"""

import json
import os
import pytest
from scripts.build_dataset import (
    validate_item,
    deduplicate_items,
    stratified_split_50_50,
    build_dataset,
)


def test_validate_item():
    """Verify validation of required fields."""
    valid = {
        "id": "1",
        "name": "tool_a",
        "description": "A valid tool",
        "label": "benign",
        "category": "math",
    }
    assert validate_item(valid)

    # Missing field
    invalid_missing = {
        "id": "2",
        "name": "tool_b",
        "description": "Missing category",
        "label": "poisoned",
    }
    assert not validate_item(invalid_missing)

    # Empty value
    invalid_empty = {
        "id": "3",
        "name": "",
        "description": "Empty name",
        "label": "benign",
        "category": "math",
    }
    assert not validate_item(invalid_empty)


def test_deduplicate_items():
    """Verify exact and near-duplicate removal."""
    items = [
        {"id": "1", "name": "a", "description": "Calculate square root of number.", "label": "benign", "category": "math"},
        {"id": "1", "name": "a_dup_id", "description": "Completely different text.", "label": "benign", "category": "math"},
        {"id": "2", "name": "b", "description": "Calculate square root of number!", "label": "benign", "category": "math"},
        {"id": "3", "name": "c", "description": "Send network ping request.", "label": "benign", "category": "network"},
    ]
    unique, removed = deduplicate_items(items, threshold=0.85)
    # id 1 dup removed, description near-dup removed -> 2 removed, 2 unique
    assert len(unique) == 2
    assert removed == 2
    unique_ids = [it["id"] for it in unique]
    assert unique_ids == ["1", "3"]


def test_stratified_split_50_50():
    """Verify 50/50 split stratified by (label, category)."""
    items = []
    # 4 items per (label, category) combination across 2 labels and 2 categories
    for lbl in ["benign", "poisoned"]:
        for cat in ["math", "utility"]:
            for i in range(4):
                items.append({
                    "id": f"{lbl}_{cat}_{i}",
                    "name": f"tool_{lbl}_{cat}_{i}",
                    "description": f"Description for {lbl} {cat} {i}",
                    "label": lbl,
                    "category": cat,
                })

    assert len(items) == 16
    dev, test = stratified_split_50_50(items, seed=42)

    assert len(dev) == 8
    assert len(test) == 8

    # Check that each (label, category) group has exactly 2 in dev and 2 in test
    for lbl in ["benign", "poisoned"]:
        for cat in ["math", "utility"]:
            dev_group = [it for it in dev if it["label"] == lbl and it["category"] == cat]
            test_group = [it for it in test if it["label"] == lbl and it["category"] == cat]
            assert len(dev_group) == 2
            assert len(test_group) == 2


def test_build_dataset_end_to_end(tmp_path):
    """Verify build_dataset creates tools_dev.json and tools_test.json with correct counts."""
    benign_file = str(tmp_path / "raw_benign.json")
    poisoned_file = str(tmp_path / "raw_poisoned.json")
    dev_file = str(tmp_path / "tools_dev.json")
    test_file = str(tmp_path / "tools_test.json")

    benign_data = [
        {"id": f"b{i}", "name": f"benign_{i}", "description": f"Benign tool description {i}", "label": "benign", "category": "math" if i < 2 else "utility"}
        for i in range(4)
    ]
    poisoned_data = [
        {"id": f"p{i}", "name": f"poison_{i}", "description": f"Poisoned tool description {i}", "label": "poisoned", "category": "math" if i < 2 else "utility"}
        for i in range(4)
    ]

    with open(benign_file, "w", encoding="utf-8") as f:
        json.dump(benign_data, f)
    with open(poisoned_file, "w", encoding="utf-8") as f:
        json.dump(poisoned_data, f)

    dev, test = build_dataset(
        benign_path=benign_file,
        poisoned_path=poisoned_file,
        dev_path=dev_file,
        test_path=test_file,
        seed=42,
    )

    assert len(dev) == 4
    assert len(test) == 4
    assert os.path.exists(dev_file)
    assert os.path.exists(test_file)

    with open(dev_file, "r", encoding="utf-8") as f:
        loaded_dev = json.load(f)
    assert len(loaded_dev) == 4
