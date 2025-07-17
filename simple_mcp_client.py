"""
Simple MCP Client
================

A clean, minimal MCP client for easy integration into any application.
This client handles MCP server connections, tool discovery, and execution
without any CLI dependencies, built on FastMCP v2.

Usage:
    from simple_mcp_client import SimpleMCPClient

    client = SimpleMCPClient()
    await client.connect_to_server("filesystem")
    result = await client.execute_tool("read_file", {"path": "example.txt"})
"""

from typing import Any

from fastmcp import Client

from config import MCPConfig, load_mcp_config


class SimpleMCPClient:
    """
    A minimal MCP client for easy integration into applications using FastMCP v2.

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
        self.client: Client | None = None
        self.available_tools: list[dict[str, Any]] = []
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
        return self.client is not None and self.client.is_connected()

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
        if self.client:
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
            # Create FastMCP v2 client 
            # For single server connections, connect directly
            if len(args) == 1 and args[0].endswith('.py'):
                # Script file - FastMCP can infer stdio transport
                self.client = Client(args[0], env=env)
            else:
                # Use transport config for complex setups
                from fastmcp.client.transports import StdioTransport
                transport = StdioTransport(command=command, args=args, env=env or {})
                self.client = Client(transport)

            # Connect and initialize
            await self.client.__aenter__()

            # Discover available tools
            tools = await self.client.list_tools()
            self.available_tools = []

            for tool in tools:
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
        if not self.client or not self.is_connected():
            raise RuntimeError("Not connected to MCP server. Call connect_to_server() first.")

        # Validate tool exists
        tool_names = [tool["name"] for tool in self.available_tools]
        if tool_name not in tool_names:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {tool_names}")

        try:
            # Execute tool using FastMCP v2 API
            result = await self.client.call_tool(tool_name, arguments)

            # FastMCP v2 returns result with .text or .data attributes
            if hasattr(result, "text"):
                return result.text
            elif hasattr(result, "data"):
                return str(result.data)
            else:
                return str(result)

        except Exception as e:
            raise RuntimeError(f"Tool execution failed: {e}") from e

    async def cleanup(self) -> None:
        """Clean up resources and disconnect from server."""
        if self.client:
            try:
                await self.client.__aexit__(None, None, None)
            except Exception:
                pass  # Ignore cleanup errors
            self.client = None
        self.available_tools = []
        self.current_server = None

    async def __aenter__(self) -> "SimpleMCPClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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
