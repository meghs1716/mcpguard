"""
servers/notes.py - FastMCP notes server over stdio.
Stores notes locally in notes.json.
"""

import json
import os
try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from mcp.server import MCPServer as FastMCP

mcp = FastMCP("Notes")

# Local storage file for notes
NOTES_FILE = os.environ.get("NOTES_PATH", "notes.json")


def _load_notes() -> dict:
    """Load notes dictionary from notes.json."""
    if os.path.exists(NOTES_FILE):
        try:
            with open(NOTES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_notes(notes: dict) -> None:
    """Save notes dictionary to notes.json."""
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)


@mcp.tool(description="Save a note with title and content.")
def save_note(title: str, content: str) -> str:
    """Save or update a note by title."""
    notes = _load_notes()
    notes[title] = content
    _save_notes(notes)
    return f"Note '{title}' saved successfully."


@mcp.tool(description="Read a note by its title.")
def read_note(title: str) -> str:
    """Read a stored note by its title."""
    notes = _load_notes()
    if title in notes:
        return notes[title]
    return f"Note '{title}' not found."


if __name__ == "__main__":
    mcp.run()
