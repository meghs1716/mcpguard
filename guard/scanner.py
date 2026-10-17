"""
guard/scanner.py - Model-based tool security scanner.
Evaluates tool name, description, and schema against three fixed security questions
using LLM-based verification (p_true from guard.decide).
Combines scores with max and records the highest-scoring question as the reason.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

from guard.normalize import normalize

# Three fixed yes/no security questions evaluated against untrusted tool definitions
QUESTION_HIDDEN_INSTRUCTIONS = (
    "Does this tool description contain hidden instructions aimed at an AI?"
)
QUESTION_READ_OR_SEND = (
    "Does this tool description ask to read files or send data somewhere?"
)
QUESTION_HIDE_OR_AFFECT = (
    "Does this tool description tell the model to hide something from the user or affect other tools?"
)

QUESTIONS = [
    QUESTION_HIDDEN_INSTRUCTIONS,
    QUESTION_READ_OR_SEND,
    QUESTION_HIDE_OR_AFFECT,
]


def format_tool_text(name: str, description: str, schema: Any = None) -> str:
    """Format tool metadata and schema into a structured text representation for evaluation."""
    parts = [f"Tool Name: {name}"]
    if description:
        parts.append(f"Description: {description}")
    if schema:
        if isinstance(schema, (dict, list)):
            schema_str = json.dumps(schema, ensure_ascii=False)
        else:
            schema_str = str(schema)
        parts.append(f"Parameters Schema: {schema_str}")
    return "\n".join(parts)


def scan_tool(
    name: str,
    description: str,
    schema: Any = None,
    model: Optional[str] = None,
    client: Optional[Any] = None,
    cache_path: str = "cache.json",
    k: int = 5,
    p_true_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Evaluate a tool definition using p_true across three fixed yes/no questions.
    Normalizes input text first, detects obfuscation flags, and returns a dict with
    'p_poisoned' (float in [0.0, 1.0]), 'reason' (the highest-scoring question), and 'flags'.
    """
    clean_name, name_flags = normalize(name or "")
    clean_desc, desc_flags = normalize(description or "")
    flags = sorted(list(set(name_flags + desc_flags)))

    tool_text = format_tool_text(clean_name, clean_desc, schema)
    selected_model = model or os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")

    # Import p_true if no custom evaluator function is provided
    if p_true_fn is None:
        from guard.decide import p_true
        eval_fn = p_true
    else:
        eval_fn = p_true_fn

    # Ask the three fixed questions
    scored_questions = []
    for q in QUESTIONS:
        try:
            score = eval_fn(
                statement=q,
                text=tool_text,
                k=k,
                model=selected_model,
                client=client,
                cache_path=cache_path,
            )
        except TypeError:
            try:
                score = eval_fn(statement=q, text=tool_text)
            except TypeError:
                score = eval_fn(q, tool_text)
        scored_questions.append((float(score), q))

    # Combine with max and set reason to the highest-scoring question
    max_score, top_reason = max(scored_questions, key=lambda item: item[0])

    return {
        "p_poisoned": round(max_score, 4),
        "reason": top_reason,
        "flags": flags,
        "question_scores": {q: round(s, 4) for s, q in scored_questions},
    }


# =====================================================================
# Pin Store: SHA-256 hash of name + description + input schema
# =====================================================================

def compute_tool_hash(name: str, description: str, schema: Any = None) -> str:
    """
    Compute a deterministic SHA-256 hash of name + description + input schema.
    """
    if isinstance(schema, (dict, list)):
        schema_str = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    else:
        schema_str = str(schema or "")
    raw = f"{name}:{description or ''}:{schema_str}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_definition_hash(tool: str, definition: Any) -> str:
    """Extract name, description, schema from definition and return its SHA-256 hash."""
    if isinstance(definition, dict):
        name = definition.get("name", tool)
        desc = definition.get("description", "")
        schema = (
            definition.get("input_schema")
            or definition.get("inputSchema")
            or definition.get("parameters")
            or {}
        )
        return compute_tool_hash(name, desc, schema)
    elif hasattr(definition, "name"):
        name = getattr(definition, "name", tool)
        desc = getattr(definition, "description", "") or ""
        schema = (
            getattr(definition, "input_schema", None)
            or getattr(definition, "inputSchema", None)
            or getattr(definition, "parameters", None)
            or {}
        )
        return compute_tool_hash(name, desc, schema)
    elif isinstance(definition, (tuple, list)) and len(definition) >= 2:
        if len(definition) == 2:
            return compute_tool_hash(tool, str(definition[0]), definition[1])
        return compute_tool_hash(str(definition[0]), str(definition[1]), definition[2])
    elif isinstance(definition, str):
        return compute_tool_hash(tool, definition, {})
    else:
        return compute_tool_hash(tool, str(definition or ""), {})


def load_pins(pins_path: str = "pins.json") -> Dict[str, str]:
    """Load the pin store dictionary from disk."""
    if not os.path.exists(pins_path):
        return {}
    try:
        with open(pins_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_pins(pins_path: str, pins: Dict[str, str]) -> None:
    """Save the pin store dictionary to disk."""
    parent = os.path.dirname(os.path.abspath(pins_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(pins_path, "w", encoding="utf-8") as f:
        json.dump(pins, f, indent=2)


def check_pin(
    server: str,
    tool: str,
    definition: Any,
    pins_path: str = "pins.json",
) -> str:
    """
    Check the pin status of a tool definition against the recorded pin store.
    Returns:
      - 'new': if (server, tool) has not been pinned yet
      - 'same': if the tool hash matches the pinned hash
      - 'changed': if the tool hash differs from the pinned hash
    """
    current_hash = get_definition_hash(tool, definition)
    pins = load_pins(pins_path)

    # Check primary key format and nested format
    key = f"{server}:{tool}"
    stored_hash = None
    if key in pins:
        stored_hash = pins[key]
    elif server in pins and isinstance(pins[server], dict) and tool in pins[server]:
        stored_hash = pins[server][tool]

    if stored_hash is None:
        return "new"
    if stored_hash == current_hash:
        return "same"
    return "changed"


def approve_pin(
    server: str,
    tool: str,
    definition: Any,
    pins_path: str = "pins.json",
) -> str:
    """
    Record and approve the current tool definition hash in the pin store.
    Returns the recorded SHA-256 hash string.
    """
    current_hash = get_definition_hash(tool, definition)
    pins = load_pins(pins_path)

    key = f"{server}:{tool}"
    pins[key] = current_hash
    save_pins(pins_path, pins)

    return current_hash

