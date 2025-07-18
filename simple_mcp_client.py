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

        # Multi-server support
        self.clients: dict[str, Client] = {}
        self.tools_by_server: dict[str, list[dict[str, Any]]] = {}
        self.connected_servers: set[str] = set()

        # Backward compatibility - points to first connected server
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
        """Get all available tools from all connected servers."""
        all_tools = []
        for server_name, tools in self.tools_by_server.items():
            all_tools.extend(tools)
        return all_tools

    def get_tools_by_server(self, server_name: str) -> list[dict[str, Any]]:
        """Get tools for a specific server."""
        return self.tools_by_server.get(server_name, [])

    def get_connected_servers(self) -> list[str]:
        """Get list of connected server names."""
        return list(self.connected_servers)

    def is_connected(self) -> bool:
        """Check if client is connected to a server."""
        return self.client is not None and self.client.is_connected()

    def get_current_server(self) -> str | None:
        """Get the name of the currently connected server."""
        return self.current_server

    async def connect_to_all_servers(self) -> dict[str, list[dict[str, Any]]]:
        """
        Connect to all configured MCP servers.

        Returns:
            Dictionary mapping server names to their available tools

        Raises:
            RuntimeError: If no servers are configured or all connections fail
        """
        if not self.config.servers:
            raise RuntimeError("No servers configured")

        results = {}
        successful_connections = 0

        for server_name in self.config.servers.keys():
            try:
                tools = await self._connect_to_single_server(server_name)
                results[server_name] = tools
                successful_connections += 1

                # Set backward compatibility properties to first successful connection
                if successful_connections == 1:
                    self.client = self.clients[server_name]
                    self.current_server = server_name
                    self.available_tools = tools

            except Exception as e:
                print(f"Warning: Failed to connect to server '{server_name}': {e}")
                results[server_name] = []

        if successful_connections == 0:
            raise RuntimeError("Failed to connect to any MCP servers")

        return results

    async def _connect_to_single_server(self, server_name: str) -> list[dict[str, Any]]:
        """
        Connect to a single MCP server.

        Args:
            server_name: Name of server to connect to

        Returns:
            List of available tools from the server

        Raises:
            ValueError: If server not found in config
            RuntimeError: If connection fails
        """
        if server_name not in self.config.servers:
            raise ValueError(f"Server '{server_name}' not found in config")

        # Disconnect if already connected to this server
        if server_name in self.clients:
            await self._disconnect_from_server(server_name)

        server_config = self.config.servers[server_name]
        command = server_config["command"]
        args = server_config["args"]
        env = server_config.get("env", {})

        try:
            # Create FastMCP v2 client
            if len(args) == 1 and args[0].endswith(".py"):
                # Script file - FastMCP can infer stdio transport
                client = Client(args[0], env=env)
            else:
                # Use transport config for complex setups
                from fastmcp.client.transports import StdioTransport

                transport = StdioTransport(command=command, args=args, env=env or {})
                client = Client(transport)

            # Connect and initialize
            await client.__aenter__()

            # Discover available tools
            tools = await client.list_tools()
            available_tools = []

            for tool in tools:
                tool_def = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                    "server": server_name,  # Add server info
                }
                available_tools.append(tool_def)

            # Store connection info
            self.clients[server_name] = client
            self.tools_by_server[server_name] = available_tools
            self.connected_servers.add(server_name)

            return available_tools

        except Exception as e:
            # Clean up on failure
            if server_name in self.clients:
                await self._disconnect_from_server(server_name)
            raise RuntimeError(f"Failed to connect to MCP server '{server_name}': {e}") from e

    async def _disconnect_from_server(self, server_name: str) -> None:
        """Disconnect from a specific server."""
        if server_name in self.clients:
            try:
                await self.clients[server_name].__aexit__(None, None, None)
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                del self.clients[server_name]
                self.tools_by_server.pop(server_name, None)
                self.connected_servers.discard(server_name)

    async def connect_to_server(
        self,
        server_name: str | None = None,
        server_command: str | None = None,
        server_args: list[str] | None = None,
        server_env: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Connect to an MCP server and return available tools.

        For multi-server support, use connect_to_all_servers() instead.

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
        # Disconnect from all current servers first
        await self.cleanup()

        # Determine connection parameters
        if server_name:
            if server_name not in self.config.servers:
                raise ValueError(f"Server '{server_name}' not found in config")

            tools = await self._connect_to_single_server(server_name)

            # Set backward compatibility properties
            self.client = self.clients[server_name]
            self.current_server = server_name
            self.available_tools = tools

            return tools

        elif server_command and server_args:
            # Handle direct connection (create temporary server config)
            # This is a more complex case that would need custom handling
            # For now, raise an error to encourage using the config-based approach
            raise ValueError(
                "Direct server connections not yet supported with multi-server architecture. Please add server to mcp_config.json"
            )

        else:
            # Auto-connect to first available server (backward compatibility)
            if not self.config.servers:
                raise ValueError(
                    "No servers configured and no direct connection parameters provided"
                )

            first_server = list(self.config.servers.keys())[0]
            return await self.connect_to_server(server_name=first_server)

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool on the connected MCP servers.

        Searches for the tool across all connected servers and executes it on the first server
        that has the tool. For more control, use execute_tool_on_server().

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result as string

        Raises:
            RuntimeError: If not connected to any servers or tool execution fails
            ValueError: If tool not found on any server
        """
        if not self.connected_servers:
            raise RuntimeError(
                "Not connected to any MCP servers. Call connect_to_all_servers() first."
            )

        # Find which server has this tool
        server_with_tool = None
        for server_name, tools in self.tools_by_server.items():
            if any(tool["name"] == tool_name for tool in tools):
                server_with_tool = server_name
                break

        if not server_with_tool:
            available_tools = [tool["name"] for tool in self.get_available_tools()]
            raise ValueError(
                f"Tool '{tool_name}' not found on any connected server. Available tools: {available_tools}"
            )

        return await self.execute_tool_on_server(server_with_tool, tool_name, arguments)

    async def execute_tool_on_server(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """
        Execute a tool on a specific MCP server.

        Args:
            server_name: Name of the server to execute the tool on
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result as string

        Raises:
            ValueError: If server not connected or tool not found
            RuntimeError: If tool execution fails
        """
        if server_name not in self.clients:
            raise ValueError(
                f"Not connected to server '{server_name}'. Connected servers: {list(self.connected_servers)}"
            )

        # Validate tool exists on this server
        server_tools = self.tools_by_server.get(server_name, [])
        tool_names = [tool["name"] for tool in server_tools]
        if tool_name not in tool_names:
            raise ValueError(
                f"Tool '{tool_name}' not found on server '{server_name}'. Available tools: {tool_names}"
            )

        try:
            # Execute tool using FastMCP v2 API
            client = self.clients[server_name]
            result = await client.call_tool(tool_name, arguments)

            # FastMCP v2 returns CallToolResult with content attribute
            if hasattr(result, "content") and result.content:
                # Extract text from content items
                text_parts = []
                for content_item in result.content:
                    if hasattr(content_item, "text"):
                        text_parts.append(content_item.text)
                    else:
                        text_parts.append(str(content_item))
                return "\n".join(text_parts)
            elif hasattr(result, "text"):
                return result.text
            elif hasattr(result, "data"):
                return str(result.data)
            else:
                return str(result)

        except Exception as e:
            raise RuntimeError(f"Tool execution failed on server '{server_name}': {e}") from e

    async def cleanup(self) -> None:
        """Clean up all server connections and reset state."""
        # Disconnect from all servers
        for server_name in list(self.connected_servers):
            await self._disconnect_from_server(server_name)

        # Reset state
        self.clients.clear()
        self.tools_by_server.clear()
        self.connected_servers.clear()

        # Reset backward compatibility properties
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
