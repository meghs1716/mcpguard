"""
agent/victim.py - MCP client agent connecting to MCP servers.
Queries the LLM with tool calling, executes tool calls via MCP sessions,
logs each tool call to events.jsonl, and returns the final answer.
"""

import asyncio
from contextlib import AsyncExitStack
import datetime
import json
import os
import sys
from typing import Any, List, Optional
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()


def mcp_to_openai_tool(tool: Any) -> dict:
    """Convert an MCP Tool definition to the Groq/OpenAI tool format."""
    input_schema = getattr(tool, "input_schema", None)
    if input_schema is None and hasattr(tool, "inputSchema"):
        input_schema = getattr(tool, "inputSchema")
    if input_schema is None:
        input_schema = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": input_schema,
        },
    }


def _format_tool_result(result: Any) -> str:
    """Extract text content from an MCP CallToolResult or return as string."""
    if hasattr(result, "content") and result.content:
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            else:
                texts.append(str(item))
        return "\n".join(texts)
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        return json.dumps(result)
    return str(result)


def _log_event(events_path: str, event: dict) -> None:
    """Append a single JSON line event to events.jsonl."""
    dir_name = os.path.dirname(os.path.abspath(events_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


async def run(
    user_msg: str,
    server_scripts: List[Any],
    guarded: bool = False,
    model: str = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
    client: Optional[Any] = None,
    events_path: str = "events.jsonl",
    max_steps: int = 5,
) -> str:
    """
    Connect to MCP servers, convert tools to Groq/OpenAI format,
    run up to max_steps (default 5) LLM iterations, execute tool calls,
    append results as role 'tool' messages with tool_call_id,
    log events to events.jsonl, and return the final response text.
    """
    # Initialize Groq client if not provided
    if client is None:
        from groq import AsyncGroq
        client = AsyncGroq()

    async with AsyncExitStack() as stack:
        tool_to_session = {}
        openai_tools = []

        # Connect to each MCP server script
        for script in server_scripts:
            # Allow mock or pre-initialized sessions in tests
            if hasattr(script, "list_tools") and hasattr(script, "call_tool"):
                session = script
                server_id = getattr(script, "server_id", "mock_server")
            else:
                script_path = os.path.abspath(str(script))
                server_id = script_path
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                params = StdioServerParameters(
                    command=sys.executable,
                    args=[script_path],
                    env=env,
                )
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(params)
                )
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()

            # If guarded mode is enabled, wrap the session with GuardedSession
            if guarded:
                from guard.guard import GuardedSession
                session = GuardedSession(
                    session=session,
                    server_id=server_id,
                    events_path=events_path,
                    model=model,
                )

            # Retrieve available tools from this session
            tools_res = await session.list_tools()
            for t in tools_res.tools:
                openai_tools.append(mcp_to_openai_tool(t))
                tool_to_session[t.name] = session

        messages = [{"role": "user", "content": user_msg}]
        final_answer = ""

        # Loop of at most max_steps (5)
        for step in range(max_steps):
            call_kwargs = {
                "model": model,
                "messages": messages,
            }
            if openai_tools:
                call_kwargs["tools"] = openai_tools

            # Call the model (supports async client or sync mock)
            raw_res = client.chat.completions.create(**call_kwargs)
            if asyncio.iscoroutine(raw_res):
                response = await raw_res
            else:
                response = raw_res

            choice = response.choices[0]
            assistant_msg = choice.message

            # If no tool calls were requested, return the model's final response
            if not getattr(assistant_msg, "tool_calls", None):
                final_answer = assistant_msg.content or ""
                break

            # Append the assistant's message with tool calls to conversation history
            messages.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ],
            })

            # Execute tool calls through their respective MCP session
            for tc in assistant_msg.tool_calls:
                # Parse call.function.arguments with json.loads
                raw_args = tc.function.arguments
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}

                tool_name = tc.function.name
                session = tool_to_session.get(tool_name)
                if session:
                    tool_result = await session.call_tool(tool_name, arguments=args)
                    result_text = _format_tool_result(tool_result)
                else:
                    result_text = f"Error: Tool '{tool_name}' not found."

                # Log each tool call to events.jsonl
                _log_event(
                    events_path=events_path,
                    event={
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "step": step,
                        "tool_call_id": tc.id,
                        "tool_name": tool_name,
                        "arguments": args,
                        "result": result_text,
                        "guarded": guarded,
                    },
                )

                # Append results as role "tool" messages with the tool_call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })
        else:
            final_answer = assistant_msg.content or ""

        return final_answer
