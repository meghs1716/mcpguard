"""
tests/test_make_templates.py - Unit tests for scripts/make_templates.py.
"""

import json
import os
import pytest
from scripts.make_templates import (
    generate_templates,
    insert_zero_width_spaces,
    base64_obfuscate,
    space_out_key_phrase,
    combine_tool_and_sentence,
    make_param_obfuscation,
)


def test_obfuscation_helpers():
    """Verify obfuscation helper functions."""
    text = "Please send_email to audit"
    zws = insert_zero_width_spaces(text)
    assert "\u200b" in zws

    b64 = base64_obfuscate(text)
    assert b64.startswith("Configuration note (base64): ")

    spaced = space_out_key_phrase("call send_email tool", "H1")
    assert "s e n d _ e m a i l" in spaced

    param_obf = make_param_obfuscation("Simple desc.", "param1 (string)", "SECRET INSTRUCTION", None)
    assert "SECRET INSTRUCTION" in param_obf
    assert param_obf.startswith("Simple desc. Parameters:")


def test_generate_templates_execution(tmp_path):
    """Verify end-to-end template generation, required fields, and category balancing."""
    out_p = str(tmp_path / "raw_poisoned.json")
    out_hb = str(tmp_path / "raw_benign_hard.json")

    poisoned, hard_benign = generate_templates(
        out_poisoned_path=out_p,
        out_benign_hard_path=out_hb,
        seed=42,
    )

    required_fields = {"id", "name", "description", "label", "category", "source_sentence_id", "split"}

    # Check poisoned items schema
    assert len(poisoned) > 0
    for it in poisoned:
        assert set(it.keys()) == required_fields
        assert it["label"] == "poisoned"
        assert it["category"] in ("hidden_instruction", "shadowing", "exfil_request", "obfuscated")
        assert it["split"] in ("dev", "test")

    # Check that each category in poisoned has the exact same count per split
    from collections import defaultdict
    p_counts = defaultdict(lambda: defaultdict(int))
    for it in poisoned:
        p_counts[it["split"]][it["category"]] += 1

    # Dev counts should all be equal
    dev_counts = set(p_counts["dev"].values())
    assert len(dev_counts) == 1, f"Dev category counts not equal: {dict(p_counts['dev'])}"

    # Test counts should all be equal
    test_counts = set(p_counts["test"].values())
    assert len(test_counts) == 1, f"Test category counts not equal: {dict(p_counts['test'])}"

    # Check hard benign items
    assert len(hard_benign) == 12
    hb_dev = sum(1 for it in hard_benign if it["split"] == "dev")
    hb_test = sum(1 for it in hard_benign if it["split"] == "test")
    assert hb_dev == 6
    assert hb_test == 6
    for it in hard_benign:
        assert set(it.keys()) == required_fields
        assert it["label"] == "benign"
        assert it["category"] == "hard_benign"

    # Verify files exist on disk and match
    assert os.path.exists(out_p)
    assert os.path.exists(out_hb)
    with open(out_p, "r", encoding="utf-8") as f:
        loaded_p = json.load(f)
    assert len(loaded_p) == len(poisoned)
