"""
tests/test_servers.py - Unit tests for calculator.py and notes.py.
"""

import os
import sys
import pytest
from mcp import Client, StdioServerParameters


@pytest.mark.anyio
async def test_calculator_server():
    """Verify calculator server starts over stdio and executes add(2, 3)."""
    calc_path = os.path.abspath("servers/calculator.py")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    params = StdioServerParameters(command=sys.executable, args=[calc_path], env=env)

    async with Client(params) as client:
        tools_res = await client.list_tools()
        tool_names = [t.name for t in tools_res.tools]
        assert "add" in tool_names

        res = await client.call_tool("add", {"a": 2, "b": 3})
        assert res.content[0].text == "5"


@pytest.mark.anyio
async def test_notes_server(tmp_path):
    """Verify notes server saves and reads notes from local notes.json."""
    notes_file = str(tmp_path / "notes.json")
    notes_path = os.path.abspath("servers/notes.py")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NOTES_PATH"] = notes_file
    params = StdioServerParameters(command=sys.executable, args=[notes_path], env=env)

    async with Client(params) as client:
        tools_res = await client.list_tools()
        tool_names = [t.name for t in tools_res.tools]
        assert "save_note" in tool_names
        assert "read_note" in tool_names

        # Save note
        save_res = await client.call_tool(
            "save_note", {"title": "meeting", "content": "discuss project roadmap"}
        )
        assert "saved successfully" in save_res.content[0].text

        # Verify notes.json was written
        assert os.path.exists(notes_file)

        # Read note
        read_res = await client.call_tool("read_note", {"title": "meeting"})
        assert read_res.content[0].text == "discuss project roadmap"

        # Read nonexistent note
        missing_res = await client.call_tool("read_note", {"title": "unknown"})
        assert "not found" in missing_res.content[0].text
