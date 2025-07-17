"""Tests for SimpleMCPClient."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from simple_mcp_client import SimpleMCPClient, execute_mcp_tool, list_mcp_tools
from config import MCPConfig


class TestSimpleMCPClient:
    """Test cases for SimpleMCPClient."""

    def test_init_with_config_path(self, temp_config_file: str):
        """Test initialization with config file path."""
        client = SimpleMCPClient(config_path=temp_config_file)
        assert client.config is not None
        assert "test-server" in client.config.servers
        assert "filesystem" in client.config.servers

    def test_init_with_config_object(self, sample_mcp_config: MCPConfig):
        """Test initialization with config object."""
        client = SimpleMCPClient(config=sample_mcp_config)
        assert client.config == sample_mcp_config
        assert client.session is None
        assert client.available_tools == []
        assert client.current_server is None

    def test_init_with_missing_config_file(self):
        """Test initialization with missing config file."""
        client = SimpleMCPClient(config_path="nonexistent.json")
        assert client.config is not None
        assert client.config.servers == {}

    def test_list_servers(self, simple_mcp_client: SimpleMCPClient):
        """Test listing configured servers."""
        servers = simple_mcp_client.list_servers()
        assert "test-server" in servers
        assert "filesystem" in servers
        assert len(servers) == 2

    def test_get_server_info(self, simple_mcp_client: SimpleMCPClient):
        """Test getting server information."""
        info = simple_mcp_client.get_server_info("test-server")
        assert info["command"] == "python"
        assert info["args"] == ["-m", "test_server"]
        assert info["description"] == "Test server"

    def test_get_server_info_nonexistent(self, simple_mcp_client: SimpleMCPClient):
        """Test getting info for nonexistent server."""
        info = simple_mcp_client.get_server_info("nonexistent")
        assert info == {}

    def test_get_available_tools_empty(self, simple_mcp_client: SimpleMCPClient):
        """Test getting available tools when not connected."""
        tools = simple_mcp_client.get_available_tools()
        assert tools == []

    def test_is_connected_false(self, simple_mcp_client: SimpleMCPClient):
        """Test connection status when not connected."""
        assert not simple_mcp_client.is_connected()

    def test_get_current_server_none(self, simple_mcp_client: SimpleMCPClient):
        """Test getting current server when not connected."""
        assert simple_mcp_client.get_current_server() is None

    async def test_connect_to_server_by_name(self, connected_client: SimpleMCPClient):
        """Test connecting to server by name."""
        assert connected_client.is_connected()
        assert connected_client.get_current_server() == "test-server"

        tools = connected_client.get_available_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "test_tool"
        assert tools[1]["name"] == "another_tool"

    async def test_connect_to_server_nonexistent(self, simple_mcp_client: SimpleMCPClient):
        """Test connecting to nonexistent server."""
        with pytest.raises(ValueError, match="Server 'nonexistent' not found"):
            await simple_mcp_client.connect_to_server("nonexistent")

    async def test_connect_to_server_direct_params(self, simple_mcp_client: SimpleMCPClient):
        """Test connecting with direct parameters."""
        with (
            patch("simple_mcp_client.stdio_client") as mock_stdio,
            patch("simple_mcp_client.ClientSession") as mock_session_class,
        ):
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            # Mock stdio client to return proper async context manager
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_stdio.return_value = mock_context

            # Mock tools response
            mock_tools_response = MagicMock()
            mock_tools_response.tools = []
            mock_session.list_tools.return_value = mock_tools_response

            tools = await simple_mcp_client.connect_to_server(
                server_command="python",
                server_args=["-m", "test_server"],
                server_env={"TEST": "value"},
            )

            assert simple_mcp_client.get_current_server() == "direct"
            assert tools == []

    async def test_connect_to_server_no_params(self, simple_mcp_client: SimpleMCPClient):
        """Test connecting with no parameters (auto-connect)."""
        with (
            patch("simple_mcp_client.stdio_client") as mock_stdio,
            patch("simple_mcp_client.ClientSession") as mock_session_class,
        ):
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            # Mock stdio client to return proper async context manager
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_stdio.return_value = mock_context

            # Mock tools response
            mock_tools_response = MagicMock()
            mock_tools_response.tools = []
            mock_session.list_tools.return_value = mock_tools_response

            tools = await simple_mcp_client.connect_to_server()

            # Should auto-connect to first server
            assert simple_mcp_client.get_current_server() == "test-server"

    async def test_connect_to_server_empty_config(self):
        """Test connecting with empty config."""
        client = SimpleMCPClient(config=MCPConfig())

        with pytest.raises(ValueError, match="No servers configured"):
            await client.connect_to_server()

    async def test_connect_to_server_connection_failure(self, simple_mcp_client: SimpleMCPClient):
        """Test connection failure handling."""
        with patch("simple_mcp_client.stdio_client") as mock_stdio:
            mock_stdio.side_effect = Exception("Connection failed")

            with pytest.raises(RuntimeError, match="Failed to connect to MCP server"):
                await simple_mcp_client.connect_to_server("test-server")

    async def test_execute_tool_success(self, connected_client: SimpleMCPClient, mock_tool_result):
        """Test successful tool execution."""
        connected_client.session.call_tool.return_value = mock_tool_result

        result = await connected_client.execute_tool("test_tool", {"param": "value"})

        assert result == "Test result"
        connected_client.session.call_tool.assert_called_once_with("test_tool", {"param": "value"})

    async def test_execute_tool_not_connected(self, simple_mcp_client: SimpleMCPClient):
        """Test tool execution when not connected."""
        with pytest.raises(RuntimeError, match="Not connected to MCP server"):
            await simple_mcp_client.execute_tool("test_tool", {})

    async def test_execute_tool_not_found(self, connected_client: SimpleMCPClient):
        """Test executing nonexistent tool."""
        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            await connected_client.execute_tool("nonexistent", {})

    async def test_execute_tool_execution_failure(self, connected_client: SimpleMCPClient):
        """Test tool execution failure."""
        connected_client.session.call_tool.side_effect = Exception("Tool failed")

        with pytest.raises(RuntimeError, match="Tool execution failed"):
            await connected_client.execute_tool("test_tool", {})

    async def test_execute_tool_multiple_content(self, connected_client: SimpleMCPClient):
        """Test tool execution with multiple content parts."""
        mock_result = MagicMock()
        mock_content1 = MagicMock()
        mock_content1.text = "Part 1"
        mock_content2 = MagicMock()
        mock_content2.text = "Part 2"
        mock_result.content = [mock_content1, mock_content2]

        connected_client.session.call_tool.return_value = mock_result

        result = await connected_client.execute_tool("test_tool", {})
        assert result == "Part 1\nPart 2"

    async def test_execute_tool_non_text_content(self, connected_client: SimpleMCPClient):
        """Test tool execution with non-text content."""
        mock_result = MagicMock()
        mock_content = MagicMock()
        mock_content.text = None
        mock_content.__str__ = lambda: "String content"
        mock_result.content = [mock_content]

        connected_client.session.call_tool.return_value = mock_result

        result = await connected_client.execute_tool("test_tool", {})
        assert "String content" in result

    async def test_cleanup(self, connected_client: SimpleMCPClient):
        """Test cleanup functionality."""
        assert connected_client.is_connected()

        await connected_client.cleanup()

        assert not connected_client.is_connected()
        assert connected_client.get_current_server() is None
        assert connected_client.get_available_tools() == []

    async def test_context_manager(self, sample_mcp_config: MCPConfig):
        """Test async context manager functionality."""
        async with SimpleMCPClient(config=sample_mcp_config) as client:
            assert client is not None
            assert isinstance(client, SimpleMCPClient)

        # Client should be cleaned up after context manager exits
        assert not client.is_connected()

    async def test_reconnect_to_different_server(self, simple_mcp_client: SimpleMCPClient):
        """Test reconnecting to a different server."""
        with (
            patch("simple_mcp_client.stdio_client") as mock_stdio,
            patch("simple_mcp_client.ClientSession") as mock_session_class,
        ):
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            # Mock stdio client to return proper async context manager
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_stdio.return_value = mock_context

            # Mock tools response
            mock_tools_response = MagicMock()
            mock_tools_response.tools = []
            mock_session.list_tools.return_value = mock_tools_response

            # Connect to first server
            await simple_mcp_client.connect_to_server("test-server")
            assert simple_mcp_client.get_current_server() == "test-server"

            # Connect to second server (should disconnect from first)
            await simple_mcp_client.connect_to_server("filesystem")
            assert simple_mcp_client.get_current_server() == "filesystem"


class TestConvenienceFunctions:
    """Test convenience functions."""

    async def test_execute_mcp_tool(self, temp_config_file: str):
        """Test execute_mcp_tool convenience function."""
        with patch("simple_mcp_client.SimpleMCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            mock_client.execute_tool.return_value = "Test result"

            result = await execute_mcp_tool(
                "test-server", "test_tool", {"param": "value"}, temp_config_file
            )

            assert result == "Test result"
            mock_client.connect_to_server.assert_called_once_with("test-server")
            mock_client.execute_tool.assert_called_once_with("test_tool", {"param": "value"})

    async def test_list_mcp_tools(self, temp_config_file: str):
        """Test list_mcp_tools convenience function."""
        with patch("simple_mcp_client.SimpleMCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            expected_tools = [{"name": "test_tool", "description": "Test tool"}]
            mock_client.connect_to_server.return_value = expected_tools

            tools = await list_mcp_tools("test-server", temp_config_file)

            assert tools == expected_tools
            mock_client.connect_to_server.assert_called_once_with("test-server")
