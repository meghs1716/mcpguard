"""
tests/test_check_attacks.py - Unit tests for eval/check_attacks.py.
"""

import os
import json
import pytest
from eval.check_attacks import (
    reset_artifacts,
    check_stolen_log,
    check_forbidden_tool,
    evaluate_attack,
    ATTACKS,
)


def test_reset_artifacts(tmp_path, monkeypatch):
    """Verify reset_artifacts removes target files."""
    monkeypatch.chdir(tmp_path)
    for fname in ["stolen.log", "events.jsonl", ".rugpull_counter", "notes.json"]:
        with open(fname, "w", encoding="utf-8") as f:
            f.write("test content")

    reset_artifacts()

    for fname in ["stolen.log", "events.jsonl", ".rugpull_counter", "notes.json"]:
        assert not os.path.exists(fname)


def test_check_stolen_log(tmp_path):
    """Verify check_stolen_log correctly detects empty vs non-empty stolen.log."""
    log_path = str(tmp_path / "stolen.log")
    assert not check_stolen_log(log_path)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n   \n")
    assert not check_stolen_log(log_path)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("stolen private key\n")
    assert check_stolen_log(log_path)


def test_check_forbidden_tool(tmp_path):
    """Verify check_forbidden_tool detects presence of forbidden tool call in events.jsonl."""
    events_path = str(tmp_path / "events.jsonl")
    check_fn = check_forbidden_tool("save_note", events_path=events_path)

    assert not check_fn()

    with open(events_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"tool_name": "get_weather"}) + "\n")
    assert not check_fn()

    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"tool_name": "save_note"}) + "\n")
    assert check_fn()


@pytest.mark.anyio
async def test_evaluate_attack_mock(monkeypatch):
    """Verify evaluate_attack counts successes accurately over iterations."""
    counter = 0

    def mock_check():
        nonlocal counter
        counter += 1
        return counter % 2 == 1  # True on 1st, 3rd, 5th run -> 3 successes

    attack = {
        "name": "mock_attack",
        "description": "Mock attack for testing",
        "server_scripts": [],
        "user_msg": "test",
        "check": mock_check,
        "pre_run": None,
    }

    async def mock_run_victim(*args, **kwargs):
        pass

    monkeypatch.setattr("eval.check_attacks.run_victim", mock_run_victim)

    successes = await evaluate_attack(attack, num_runs=5)
    assert successes == 3
