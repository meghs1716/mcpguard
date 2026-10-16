"""
agent/plain_client.py - Plain MCP client connecting to calculator.py.
Lists available tools, prints their name, description, and input schema,
and calls add(2, 3).
"""

import asyncio
import os
import sys
from mcp import Client, StdioServerParameters


async def main() -> None:
    # Resolve absolute path to servers/calculator.py
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    calc_path = os.path.join(base_dir, "servers", "calculator.py")

    # Set up stdio server parameters for the subprocess
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[calc_path],
        env=env,
    )

    # Connect to the MCP server using the high-level Client
    async with Client(server_params) as client:
        # List tools exposed by the server
        tools_result = await client.list_tools()
        print("Available tools:")
        for tool in tools_result.tools:
            print(f"- Name: {tool.name}")
            print(f"  Description: {tool.description}")
            print(f"  Input Schema: {tool.input_schema}")

        # Call the add tool with arguments 2 and 3
        print("\nCalling add(2, 3)...")
        result = await client.call_tool("add", {"a": 2, "b": 3})

        # Extract and print result content
        if result.content:
            for item in result.content:
                text = getattr(item, "text", str(item))
                print(f"Result: {text}")
        elif hasattr(result, "structured_content"):
            print(f"Structured Result: {result.structured_content}")
        else:
            print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
