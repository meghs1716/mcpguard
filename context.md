Project: MCPGuard, a security layer for MCP (Model Context Protocol) clients.
It defends an LLM client against malicious MCP servers: tool-description
poisoning, rug pulls (description changes after approval), result injection,
and argument exfiltration.

Stack: Python 3.11+, `mcp` SDK (FastMCP for servers, ClientSession for client,
stdio transport), Ollama with qwen2.5:3b for the model, pydantic, pytest, Streamlit.

Folder layout:
  servers/   calculator.py notes.py evil_poisoned.py evil_result.py evil_exfil.py evil_rugpull.py
  guard/     decide.py normalize.py scanner.py guard.py
  agent/     victim.py
  data/      tools_dev.json tools_test.json
  eval/      baseline.py eval_scanner.py attack_runner.py sweep.py plots.py
  app/       dashboard.py
  tests/
  fake_secrets/id_rsa   (dummy text only)
  events.jsonl          (one JSON event per line)

Rules:
- I am a beginner at MCP. Keep code simple, with short comments on non-obvious lines.
- Before writing MCP code, check the installed `mcp` version (`pip show mcp`) and
  use APIs that exist in that version. Never guess.
- All attacks are local, harmless sandbox demos. Never read real user files; the
  "secret" is always fake_secrets/id_rsa. Nothing touches the network.
- Keep files small. Don't modify finished files unless I ask.
- For every task: show a short plan first, then code, then the exact command to
  test it and the output I should expect.