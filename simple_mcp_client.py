"""
Simple MCP Client
================

A clean, minimal MCP client for easy integration into any application.
This client handles MCP server connections, tool discovery, and execution
without any CLI dependencies.

Usage:
    from simple_mcp_client import SimpleMCPClient

    client = SimpleMCPClient()
    await client.connect_to_server("filesystem")
    result = await client.execute_tool("read_file", {"path": "example.txt"})
"""

from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import MCPConfig, load_mcp_config


class SimpleMCPClient:
    """
    A minimal MCP client for easy integration into applications.

    This client focuses solely on MCP functionality:
    - Connect to MCP servers
    - Discover available tools
    - Execute tools
    - Handle server lifecycle
    """

    def __init__(self, config_path: str = "mcp_config.json", config: MCPConfig | None = None):
        """
        Initialize the MCP client.

        Args:
            config_path: Path to MCP configuration file
            config: Optional MCPConfig object (overrides config_path)
        """
        self.config = config or load_mcp_config(config_path)
        self.session: ClientSession | None = None
        self.available_tools: list[dict[str, Any]] = []
        self.exit_stack = AsyncExitStack()
        self.current_server: str | None = None

    def list_servers(self) -> list[str]:
        """List all configured MCP servers."""
        return list(self.config.servers.keys())

    def get_server_info(self, server_name: str) -> dict[str, Any]:
        """Get configuration info for a specific server."""
        return self.config.servers.get(server_name, {})

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Get list of available tools from connected server."""
        return self.available_tools.copy()

    def is_connected(self) -> bool:
        """Check if client is connected to a server."""
        return self.session is not None

    def get_current_server(self) -> str | None:
        """Get the name of the currently connected server."""
        return self.current_server

    async def connect_to_server(
        self,
        server_name: str | None = None,
        server_command: str | None = None,
        server_args: list[str] | None = None,
        server_env: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Connect to an MCP server and return available tools.

        Args:
            server_name: Name of server from config
            server_command: Direct command to run server
            server_args: Arguments for server command
            server_env: Environment variables for server

        Returns:
            List of available tools

        Raises:
            ValueError: If server not found or no connection params
            RuntimeError: If connection fails
        """
        # Disconnect from current server if connected
        if self.session:
            await self.cleanup()

        # Determine connection parameters
        if server_name:
            if server_name not in self.config.servers:
                raise ValueError(f"Server '{server_name}' not found in config")

            server_config = self.config.servers[server_name]
            command = server_config["command"]
            args = server_config["args"]
            env = server_config.get("env", {})
            if server_env:
                env.update(server_env)
            self.current_server = server_name

        elif server_command and server_args:
            command = server_command
            args = server_args
            env = server_env or {}
            self.current_server = "direct"

        else:
            # Auto-connect to first available server
            if not self.config.servers:
                raise ValueError(
                    "No servers configured and no direct connection parameters provided"
                )

            first_server = list(self.config.servers.keys())[0]
            return await self.connect_to_server(server_name=first_server)

        try:
            # Create server parameters
            server_params = StdioServerParameters(command=command, args=args, env=env)

            # Connect to server
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read_stream, write_stream = stdio_transport

            # Create session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialize session
            await self.session.initialize()

            # Discover available tools
            tools_response = await self.session.list_tools()
            self.available_tools = []

            for tool in tools_response.tools:
                tool_def = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                self.available_tools.append(tool_def)

            return self.available_tools

        except Exception as e:
            # Clean up on failure
            await self.cleanup()
            raise RuntimeError(f"Failed to connect to MCP server: {e}") from e

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool on the connected MCP server.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result as string

        Raises:
            RuntimeError: If not connected or tool execution fails
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server. Call connect_to_server() first.")

        # Validate tool exists
        tool_names = [tool["name"] for tool in self.available_tools]
        if tool_name not in tool_names:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {tool_names}")

        try:
            # Execute tool
            result = await self.session.call_tool(tool_name, arguments)

            # Extract and return result
            output_parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    output_parts.append(content.text)
                else:
                    output_parts.append(str(content))

            return "\n".join(output_parts)

        except Exception as e:
            raise RuntimeError(f"Tool execution failed: {e}") from e

    async def cleanup(self) -> None:
        """Clean up resources and disconnect from server."""
        if self.exit_stack:
            await self.exit_stack.aclose()
        self.session = None
        self.available_tools = []
        self.current_server = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# Convenience functions for common use cases
async def execute_mcp_tool(
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    config_path: str = "mcp_config.json",
) -> str:
    """
    Convenience function to execute a single MCP tool.

    Args:
        server_name: Name of MCP server
        tool_name: Name of tool to execute
        arguments: Tool arguments
        config_path: Path to MCP config file

    Returns:
        Tool execution result
    """
    async with SimpleMCPClient(config_path) as client:
        await client.connect_to_server(server_name)
        return await client.execute_tool(tool_name, arguments)


async def list_mcp_tools(
    server_name: str, config_path: str = "mcp_config.json"
) -> list[dict[str, Any]]:
    """
    Convenience function to list available tools from an MCP server.

    Args:
        server_name: Name of MCP server
        config_path: Path to MCP config file

    Returns:
        List of available tools
    """
    async with SimpleMCPClient(config_path) as client:
        return await client.connect_to_server(server_name)
