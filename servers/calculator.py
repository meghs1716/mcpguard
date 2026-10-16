"""
servers/calculator.py - FastMCP calculator server over stdio.
Exposes arithmetic tools (add).
"""

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from mcp.server import MCPServer as FastMCP

mcp = FastMCP("Calculator")


@mcp.tool(description="Add two numbers together.")
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    mcp.run()
