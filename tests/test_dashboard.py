"""
tests/test_dashboard.py - Unit tests for app/dashboard.py.
Verifies event loading, normalization, filtering, styling, and sample_events.jsonl.
"""

import json
import os
import pytest
import pandas as pd
from app.dashboard import (
    DEFAULT_EVENTS_PATH,
    SAMPLE_EVENTS_PATH,
    filter_events,
    load_events,
    style_action_cells,
)


def test_sample_events_file_exists():
    """Verify sample_events.jsonl exists and contains valid records."""
    assert os.path.exists(SAMPLE_EVENTS_PATH), f"Missing {SAMPLE_EVENTS_PATH}"

    with open(SAMPLE_EVENTS_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 5, f"Expected at least 5 sample events, got {len(lines)}"

    actions = set()
    stages = set()
    for line in lines:
        ev = json.loads(line)
        assert "timestamp" in ev
        assert "stage" in ev
        assert "tool" in ev
        assert "probability" in ev
        assert "action" in ev
        assert "reason" in ev
        actions.add(ev["action"])
        stages.add(ev["stage"])

    # Ensure allow, ask, and block actions are represented
    assert "allow" in actions
    assert "ask" in actions
    assert "block" in actions

    # Ensure scan, args, and result stages are represented
    assert "scan" in stages
    assert "args" in stages
    assert "result" in stages


def test_load_events_valid(tmp_path):
    """Verify load_events parses JSONL correctly into a DataFrame."""
    events_file = tmp_path / "test_events.jsonl"
    data = [
        {
            "timestamp": "2026-10-17T00:00:00Z",
            "stage": "scan",
            "tool": "add",
            "probability": 0.05,
            "action": "allow",
            "reason": "clean description",
        },
        {
            "timestamp": "2026-10-17T00:01:00Z",
            "stage": "args",
            "tool": "evil_exfil",
            "probability": 0.95,
            "action": "block",
            "reason": "exfiltration detected",
        },
    ]
    with open(events_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    df = load_events(str(events_file))
    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "stage", "tool", "probability", "action", "reason"]
    assert df.iloc[0]["tool"] == "add"
    assert df.iloc[0]["action"] == "allow"
    assert df.iloc[1]["action"] == "block"
    assert df.iloc[1]["probability"] == 0.95


def test_load_events_missing_file(tmp_path):
    """Verify load_events handles missing file gracefully."""
    non_existent = str(tmp_path / "does_not_exist.jsonl")
    df = load_events(non_existent)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "action" in df.columns


def test_load_events_empty_file(tmp_path):
    """Verify load_events handles empty file gracefully."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    df = load_events(str(empty_file))
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_filter_events():
    """Verify filter_events correctly filters by tool, stage, and action."""
    records = [
        {"timestamp": "t1", "stage": "scan", "tool": "calc", "probability": 0.1, "action": "allow", "reason": "ok"},
        {"timestamp": "t2", "stage": "args", "tool": "calc", "probability": 0.4, "action": "ask", "reason": "warn"},
        {"timestamp": "t3", "stage": "result", "tool": "notes", "probability": 0.9, "action": "block", "reason": "bad"},
    ]
    df = pd.DataFrame(records)

    # Filter by tool
    f_tool = filter_events(df, selected_tools=["calc"])
    assert len(f_tool) == 2
    assert set(f_tool["tool"]) == {"calc"}

    # Filter by stage
    f_stage = filter_events(df, selected_stages=["result"])
    assert len(f_stage) == 1
    assert f_stage.iloc[0]["tool"] == "notes"

    # Filter by action
    f_action = filter_events(df, selected_actions=["block"])
    assert len(f_action) == 1
    assert f_action.iloc[0]["action"] == "block"

    # Combined filter
    f_combo = filter_events(df, selected_tools=["calc"], selected_actions=["allow"])
    assert len(f_combo) == 1
    assert f_combo.iloc[0]["stage"] == "scan"


def test_style_action_cells():
    """Verify style_action_cells returns appropriate CSS rules."""
    assert "28a745" in style_action_cells("allow")  # Greenish
    assert "ff9800" in style_action_cells("ask") or "e65100" in style_action_cells("ask")  # Orange/Yellowish
    assert "dc3545" in style_action_cells("block")  # Reddish
    assert style_action_cells("unknown") == ""
