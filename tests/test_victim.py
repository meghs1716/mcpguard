"""
tests/test_victim.py - Unit and integration tests for agent/victim.py.
Tests:
1. Tool conversion from MCP Tool format to Groq/OpenAI tool format.
2. Direct response (no tool calls).
3. Tool execution, role 'tool' message creation with tool_call_id, and final answer.
4. events.jsonl logging.
5. Max steps limit (5 steps).
6. End-to-end integration test with a real MCPServer subprocess.
"""

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock
import pytest
import mcp.types as mcp_types
from agent.victim import run, mcp_to_openai_tool


def _make_mock_model_response(content: str = "", tool_calls = None):
    """Build a mock ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_tool_call(call_id: str, fn_name: str, fn_args: str):
    """Build a mock ToolCall object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = fn_name
    tc.function.arguments = fn_args
    return tc


def test_mcp_to_openai_tool():
    """Verify conversion from MCP Tool to Groq/OpenAI tool format."""
    tool = mcp_types.Tool(
        name="calculator",
        description="Calculate math expression",
        inputSchema={
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
    )
    converted = mcp_to_openai_tool(tool)
    assert converted["type"] == "function"
    assert converted["function"]["name"] == "calculator"
    assert converted["function"]["description"] == "Calculate math expression"
    assert converted["function"]["parameters"] == {
        "type": "object",
        "properties": {"expr": {"type": "string"}},
        "required": ["expr"],
    }


@pytest.mark.anyio
async def test_run_direct_response(tmp_path):
    """Verify run() returns the answer directly when no tool calls are needed."""
    events_file = str(tmp_path / "events.jsonl")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_model_response(
        content="Hello, how can I help you?"
    )

    result = await run(
        user_msg="Hello",
        server_scripts=[],
        client=mock_client,
        events_path=events_file,
    )

    assert result == "Hello, how can I help you?"
    # No tool calls should be logged
    assert not os.path.exists(events_file)


@pytest.mark.anyio
async def test_run_with_tool_call(tmp_path):
    """Verify tool call execution, parameter parsing, role 'tool' append, and logging."""
    events_file = str(tmp_path / "events.jsonl")

    # Mock MCP session
    mock_session = AsyncMock()
    mock_session.list_tools.return_value = mcp_types.ListToolsResult(
        tools=[
            mcp_types.Tool(
                name="add",
                description="Add two numbers",
                inputSchema={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
            )
        ]
    )
    mock_session.call_tool.return_value = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="42")]
    )

    # Mock Model: Step 1 returns tool call, Step 2 returns final answer
    tc = _make_tool_call("call_abc123", "add", '{"a": 20, "b": 22}')
    resp_step1 = _make_mock_model_response(content="", tool_calls=[tc])
    resp_step2 = _make_mock_model_response(content="The sum is 42", tool_calls=None)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [resp_step1, resp_step2]

    final_answer = await run(
        user_msg="What is 20 + 22?",
        server_scripts=[mock_session],
        guarded=False,
        client=mock_client,
        events_path=events_file,
    )

    assert final_answer == "The sum is 42"
    assert mock_session.call_tool.call_count == 1
    mock_session.call_tool.assert_called_with("add", arguments={"a": 20, "b": 22})

    # Verify messages sent to model in Step 2 include the assistant tool call and tool response
    call_args_step2 = mock_client.chat.completions.create.call_args_list[1].kwargs
    messages = call_args_step2["messages"]
    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "What is 20 + 22?"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_abc123"
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": "42",
    }

    # Verify event logged to events.jsonl
    assert os.path.exists(events_file)
    with open(events_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["step"] == 0
    assert event["tool_call_id"] == "call_abc123"
    assert event["tool_name"] == "add"
    assert event["arguments"] == {"a": 20, "b": 22}
    assert event["result"] == "42"
    assert event["guarded"] is False


@pytest.mark.anyio
async def test_run_max_steps_limit(tmp_path):
    """Verify run() stops after at most 5 steps when model keeps requesting tool calls."""
    events_file = str(tmp_path / "events.jsonl")

    mock_session = AsyncMock()
    mock_session.list_tools.return_value = mcp_types.ListToolsResult(
        tools=[
            mcp_types.Tool(
                name="ping",
                description="Ping tool",
                inputSchema={"type": "object", "properties": {}},
            )
        ]
    )
    mock_session.call_tool.return_value = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="pong")]
    )

    # Endless tool call responses
    def make_ping_response(*args, **kwargs):
        tc = _make_tool_call("call_loop", "ping", "{}")
        return _make_mock_model_response(content="Calling ping again", tool_calls=[tc])

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = make_ping_response

    await run(
        user_msg="Ping forever",
        server_scripts=[mock_session],
        client=mock_client,
        events_path=events_file,
        max_steps=5,
    )

    # Exactly 5 calls to the model (max_steps)
    assert mock_client.chat.completions.create.call_count == 5

    # Exactly 5 events logged
    with open(events_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 5


@pytest.mark.anyio
async def test_end_to_end_with_real_mcp_server(tmp_path):
    """Verify integration connecting to a real MCPServer subprocess via stdio."""
    events_file = str(tmp_path / "events.jsonl")

    # Create a small MCP server script
    server_script = str(tmp_path / "calc_server.py")
    server_code = """
from mcp.server.mcpserver import MCPServer

server = MCPServer("test_calc")

@server.tool(description="Multiply two integers")
def multiply(x: int, y: int) -> int:
    return x * y

if __name__ == "__main__":
    server.run()
"""
    with open(server_script, "w", encoding="utf-8") as f:
        f.write(server_code)

    # Mock Model to call multiply tool, then finish
    tc = _make_tool_call("call_mult_1", "multiply", '{"x": 6, "y": 7}')
    resp1 = _make_mock_model_response(content="", tool_calls=[tc])
    resp2 = _make_mock_model_response(content="Result is 42", tool_calls=None)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    result = await run(
        user_msg="Multiply 6 by 7",
        server_scripts=[server_script],
        client=mock_client,
        events_path=events_file,
    )

    assert result == "Result is 42"
    with open(events_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["tool_name"] == "multiply"
    assert event["arguments"] == {"x": 6, "y": 7}
    assert "42" in event["result"]
