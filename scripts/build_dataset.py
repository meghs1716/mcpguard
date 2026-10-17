"""
scripts/build_dataset.py - Dataset builder for MCPGuard.
Loads raw benign and poisoned tool definitions, validates fields,
removes near-duplicates, shuffles with a fixed seed, and writes a
stratified 50/50 split by label and category into tools_dev.json and tools_test.json.
"""

from collections import defaultdict
import json
import os
import random
import sys
from typing import Any, Dict, List, Set, Tuple

# Resolve repository root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_FIELDS = ("id", "name", "description", "label", "category")


def validate_item(item: Any) -> bool:
    """Validate that item is a dictionary containing non-empty required fields."""
    if not isinstance(item, dict):
        return False
    for field in REQUIRED_FIELDS:
        if field not in item:
            return False
        val = item[field]
        if val is None or not str(val).strip():
            return False
    return True


def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and stripping punctuation/extra whitespace."""
    clean = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
    return " ".join(clean.split())


def deduplicate_items(items: List[Dict[str, Any]], threshold: float = 0.85) -> Tuple[List[Dict[str, Any]], int]:
    """
    Remove items with duplicate IDs or near-duplicate descriptions using Jaccard token similarity.
    Returns (unique_items, count_of_removed_duplicates).
    """
    seen_ids: Set[str] = set()
    seen_descriptions: List[Tuple[str, Set[str]]] = []
    unique_items: List[Dict[str, Any]] = []
    duplicates_removed = 0

    for item in items:
        item_id = str(item["id"]).strip()
        if item_id in seen_ids:
            duplicates_removed += 1
            continue

        norm_desc = normalize_text(item["description"])
        tokens = set(norm_desc.split())

        # Check for near-duplicates in description
        is_dup = False
        for existing_norm, existing_tokens in seen_descriptions:
            if norm_desc == existing_norm:
                is_dup = True
                break
            if tokens and existing_tokens:
                intersection = len(tokens & existing_tokens)
                union = len(tokens | existing_tokens)
                if union > 0 and (intersection / union) >= threshold:
                    is_dup = True
                    break

        if is_dup:
            duplicates_removed += 1
            continue

        seen_ids.add(item_id)
        seen_descriptions.append((norm_desc, tokens))
        unique_items.append(item)

    return unique_items, duplicates_removed


def stratified_split_50_50(
    items: List[Dict[str, Any]], seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split items 50/50 stratified by (label, category) using a fixed random seed.
    """
    rng = random.Random(seed)

    # Group items by (label, category)
    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        key = (str(item["label"]).strip(), str(item["category"]).strip())
        strata[key].append(item)

    dev_items: List[Dict[str, Any]] = []
    test_items: List[Dict[str, Any]] = []
    toggle = 0

    # Sort keys for deterministic group ordering
    for key, group in sorted(strata.items()):
        # Shuffle group items with fixed seed
        rng.shuffle(group)
        n = len(group)

        if n % 2 == 0:
            n_dev = n // 2
        else:
            # Alternate placing the remainder item into dev or test
            n_dev = (n + (toggle % 2)) // 2
            toggle += 1

        dev_items.extend(group[:n_dev])
        test_items.extend(group[n_dev:])

    # Shuffle final splits
    rng.shuffle(dev_items)
    rng.shuffle(test_items)

    return dev_items, test_items


def build_dataset(
    benign_path: str = None,
    poisoned_path: str = None,
    dev_path: str = None,
    test_path: str = None,
    seed: int = 42,
    dedup_threshold: float = 0.85,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load raw files, validate, deduplicate, split 50/50, write output files,
    and print summary counts.
    """
    benign_path = benign_path or os.path.join(REPO_ROOT, "data", "raw_benign.json")
    poisoned_path = poisoned_path or os.path.join(REPO_ROOT, "data", "raw_poisoned.json")
    dev_path = dev_path or os.path.join(REPO_ROOT, "data", "tools_dev.json")
    test_path = test_path or os.path.join(REPO_ROOT, "data", "tools_test.json")

    # Load raw benign items
    if not os.path.exists(benign_path):
        raise FileNotFoundError(f"Benign dataset not found at {benign_path}")
    with open(benign_path, "r", encoding="utf-8") as f:
        raw_benign = json.load(f)

    # Load raw poisoned items
    if not os.path.exists(poisoned_path):
        raise FileNotFoundError(f"Poisoned dataset not found at {poisoned_path}")
    with open(poisoned_path, "r", encoding="utf-8") as f:
        raw_poisoned = json.load(f)

    raw_all = raw_benign + raw_poisoned
    total_raw = len(raw_all)

    # Validate items
    valid_items = [item for item in raw_all if validate_item(item)]
    invalid_count = total_raw - len(valid_items)

    # Remove near-duplicates
    unique_items, dup_count = deduplicate_items(valid_items, threshold=dedup_threshold)

    # Stratified 50/50 split
    dev_items, test_items = stratified_split_50_50(unique_items, seed=seed)

    # Write output JSON files
    os.makedirs(os.path.dirname(os.path.abspath(dev_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(test_path)), exist_ok=True)

    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_items, f, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_items, f, indent=2)

    # Compute statistics for printing
    def count_labels_and_categories(data_subset: List[Dict[str, Any]]):
        labels = defaultdict(int)
        categories = defaultdict(int)
        for it in data_subset:
            labels[str(it["label"])] += 1
            categories[str(it["category"])] += 1
        return labels, categories

    dev_labels, dev_cats = count_labels_and_categories(dev_items)
    test_labels, test_cats = count_labels_and_categories(test_items)
    all_labels = sorted(set(dev_labels.keys()) | set(test_labels.keys()))
    all_cats = sorted(set(dev_cats.keys()) | set(test_cats.keys()))

    # Print summary counts
    print("=" * 65)
    print("MCPGuard Dataset Build Summary")
    print("=" * 65)
    print(f"Total raw items loaded:     {total_raw} ({len(raw_benign)} benign, {len(raw_poisoned)} poisoned)")
    print(f"Invalid items discarded:    {invalid_count}")
    print(f"Near-duplicates removed:    {dup_count}")
    print(f"Total valid unique items:   {len(unique_items)}")
    print("-" * 65)
    print("Split Totals:")
    print(f"  Dev  split (tools_dev.json):  {len(dev_items):>3} items ({len(dev_items)/len(unique_items)*100:.1f}%)")
    print(f"  Test split (tools_test.json): {len(test_items):>3} items ({len(test_items)/len(unique_items)*100:.1f}%)")
    print("-" * 65)
    print("Counts per Label:")
    print(f"  {'Label':<15} {'Dev':>8} {'Test':>8} {'Total':>8}")
    for lbl in all_labels:
        d_cnt = dev_labels[lbl]
        t_cnt = test_labels[lbl]
        print(f"  {lbl:<15} {d_cnt:>8} {t_cnt:>8} {d_cnt + t_cnt:>8}")
    print("-" * 65)
    print("Counts per Category:")
    print(f"  {'Category':<15} {'Dev':>8} {'Test':>8} {'Total':>8}")
    for cat in all_cats:
        d_cnt = dev_cats[cat]
        t_cnt = test_cats[cat]
        print(f"  {cat:<15} {d_cnt:>8} {t_cnt:>8} {d_cnt + t_cnt:>8}")
    print("=" * 65)

    return dev_items, test_items


if __name__ == "__main__":
    build_dataset()
