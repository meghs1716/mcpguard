"""
tests/test_attack_runner.py - Unit tests for eval/attack_runner.py.
Verifies helper checks, trial execution, latency measurement,
scenario evaluation in both modes, and attacks.json structure.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock
import pytest
from eval.attack_runner import (
    check_guard_blocked,
    pre_notes_step,
    run_single_trial,
    evaluate_scenario,
    run_benchmark,
)


def test_check_guard_blocked(tmp_path):
    """Verify check_guard_blocked identifies action='block' in events.jsonl."""
    events_file = str(tmp_path / "events.jsonl")

    # Clean file
    with open(events_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"stage": "scan", "action": "allow"}) + "\n")
    assert check_guard_blocked(events_file) is False

    # Blocked event present
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": "args", "action": "block"}) + "\n")
    assert check_guard_blocked(events_file) is True


def test_pre_notes_step(tmp_path, monkeypatch):
    """Verify pre_notes_step populates notes.json."""
    monkeypatch.chdir(tmp_path)
    pre_notes_step()

    assert os.path.exists("notes.json")
    with open("notes.json", "r", encoding="utf-8") as f:
        notes = json.load(f)
    assert "groceries" in notes


@pytest.mark.anyio
async def test_run_single_trial_attack_mock(tmp_path, monkeypatch):
    """Verify run_single_trial on a mock attack scenario."""
    monkeypatch.chdir(tmp_path)
    events_file = str(tmp_path / "events.jsonl")

    # Mock attack scenario
    check_called = False
    def mock_check():
        nonlocal check_called
        check_called = True
        return True

    scenario = {
        "name": "mock_attack",
        "type": "attack",
        "description": "Mock attack for testing",
        "server_scripts": [],
        "user_msg": "Test attack",
        "check": mock_check,
        "pre_run": None,
    }

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Done", tool_calls=None))]
    mock_client.chat.completions.create.return_value = mock_resp

    success, latency, blocked = await run_single_trial(
        scenario=scenario,
        guarded=False,
        model="test-model",
        client=mock_client,
        events_path=events_file,
    )

    assert success is True
    assert check_called is True
    assert latency >= 0.0
    assert blocked is False


@pytest.mark.anyio
async def test_run_benchmark_end_to_end_mock(tmp_path, monkeypatch):
    """Verify run_benchmark saves results to attacks.json with required schema."""
    monkeypatch.chdir(tmp_path)
    out_json = str(tmp_path / "results" / "attacks.json")
    cfg_file = str(tmp_path / "config.yaml")
    events_file = str(tmp_path / "events.jsonl")

    with open(cfg_file, "w", encoding="utf-8") as f:
        f.write("t_block: 0.6\nt_ask: 0.3\n")

    mock_scenarios = [
        {
            "name": "test_attack",
            "type": "attack",
            "description": "Test attack scenario",
            "server_scripts": [],
            "user_msg": "Attack test",
            "check": lambda: False,
            "pre_run": None,
        },
        {
            "name": "test_benign",
            "type": "benign",
            "description": "Test benign scenario",
            "server_scripts": [],
            "user_msg": "Benign test",
            "check": None,
            "pre_run": None,
        },
    ]

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Hello", tool_calls=None))]
    mock_client.chat.completions.create.return_value = mock_resp

    benchmark = await run_benchmark(
        num_runs=2,
        model="qwen-test",
        client=mock_client,
        config_path=cfg_file,
        events_path=events_file,
        output_path=out_json,
        scenarios=mock_scenarios,
    )

    # Verify JSON output structure
    assert os.path.exists(out_json)
    with open(out_json, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert "date" in saved_data
    assert saved_data["model"] == "qwen-test"
    assert saved_data["thresholds"]["t_block"] == 0.6
    assert saved_data["thresholds"]["t_ask"] == 0.3
    assert saved_data["num_runs"] == 2
    assert len(saved_data["scenarios"]) == 2
    assert "benign_summary" in saved_data
    assert saved_data["benign_summary"]["total_benign_runs"] == 2
    assert saved_data["benign_summary"]["wrong_blocks"] == 0
