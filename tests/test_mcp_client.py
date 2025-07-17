"""Tests for MCPClient and QueryProcessor."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import MCPConfig
from llm_providers import AnthropicProvider, LLMMessage, LLMResponse
from mcp_client import MCPClient, QueryProcessor


class TestMCPClient:
    """Test cases for MCPClient."""

    def test_init(self, sample_mcp_config: MCPConfig) -> None:
        """Test MCPClient initialization."""
        client = MCPClient(sample_mcp_config, verbose=True)
        assert client.config == sample_mcp_config
        assert client.session is None
        assert client.available_tools == []
        assert client.current_server is None
        assert client.verbose is True

    def test_init_non_verbose(self, sample_mcp_config: MCPConfig) -> None:
        """Test MCPClient initialization without verbose mode."""
        client = MCPClient(sample_mcp_config, verbose=False)
        assert client.verbose is False

    def test_list_servers(self, sample_mcp_config: MCPConfig) -> None:
        """Test listing configured servers."""
        client = MCPClient(sample_mcp_config)
        servers = client.list_servers()
        assert "test-server" in servers
        assert "filesystem" in servers
        assert len(servers) == 2

    def test_get_server_info(self, sample_mcp_config: MCPConfig) -> None:
        """Test getting server information."""
        client = MCPClient(sample_mcp_config)
        info = client.get_server_info("test-server")
        assert info["command"] == "python"
        assert info["args"] == ["-m", "test_server"]
        assert info["description"] == "Test server"

    def test_get_server_info_nonexistent(self, sample_mcp_config: MCPConfig) -> None:
        """Test getting info for nonexistent server."""
        client = MCPClient(sample_mcp_config)
        info = client.get_server_info("nonexistent")
        assert info == {}

    async def test_connect_to_server_by_name(self, sample_mcp_config: MCPConfig) -> None:
        """Test connecting to server by name."""
        client = MCPClient(sample_mcp_config)

        with (
            patch("mcp_client.stdio_client") as mock_stdio,
            patch("mcp_client.ClientSession") as mock_session_class,
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
            mock_tool = MagicMock()
            mock_tool.name = "test_tool"
            mock_tool.description = "Test tool"
            mock_tool.inputSchema = {"type": "object"}
            mock_tools_response.tools = [mock_tool]
            mock_session.list_tools.return_value = mock_tools_response

            await client.connect_to_server("test-server")

            assert client.current_server == "test-server"
            assert len(client.available_tools) == 1
            assert client.available_tools[0]["name"] == "test_tool"

    async def test_connect_to_server_direct_params(self, sample_mcp_config: MCPConfig) -> None:
        """Test connecting with direct parameters."""
        client = MCPClient(sample_mcp_config)

        with (
            patch("mcp_client.stdio_client") as mock_stdio,
            patch("mcp_client.ClientSession") as mock_session_class,
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

            await client.connect_to_server(
                server_command="python", server_args=["-m", "test_server"]
            )

            assert client.current_server == "direct"

    async def test_connect_to_server_nonexistent(self, sample_mcp_config: MCPConfig) -> None:
        """Test connecting to nonexistent server."""
        client = MCPClient(sample_mcp_config)

        with pytest.raises(ValueError, match="Server 'nonexistent' not found"):
            await client.connect_to_server("nonexistent")

    async def test_connect_to_server_no_params_empty_config(self) -> None:
        """Test connecting with no parameters and empty config."""
        client = MCPClient(MCPConfig())

        with pytest.raises(ValueError, match="No servers configured"):
            await client.connect_to_server()

    async def test_execute_tool_success(self, sample_mcp_config: MCPConfig) -> None:
        """Test successful tool execution."""
        client = MCPClient(sample_mcp_config)

        # Mock session
        mock_session = AsyncMock()
        client.session = mock_session

        # Mock tool result
        mock_result = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Tool result"
        mock_result.content = [mock_content]
        mock_session.call_tool.return_value = mock_result

        result = await client.execute_tool("test_tool", {"param": "value"})

        assert result == "Tool result"
        mock_session.call_tool.assert_called_once_with("test_tool", {"param": "value"})

    async def test_execute_tool_not_connected(self, sample_mcp_config: MCPConfig) -> None:
        """Test tool execution when not connected."""
        client = MCPClient(sample_mcp_config)

        with pytest.raises(RuntimeError, match="Not connected to MCP server"):
            await client.execute_tool("test_tool", {})

    async def test_execute_tool_multiple_content(self, sample_mcp_config: MCPConfig) -> None:
        """Test tool execution with multiple content parts."""
        client = MCPClient(sample_mcp_config)

        # Mock session
        mock_session = AsyncMock()
        client.session = mock_session

        # Mock tool result with multiple content parts
        mock_result = MagicMock()
        mock_content1 = MagicMock()
        mock_content1.text = "Part 1"
        mock_content2 = MagicMock()
        mock_content2.text = "Part 2"
        mock_result.content = [mock_content1, mock_content2]
        mock_session.call_tool.return_value = mock_result

        result = await client.execute_tool("test_tool", {})
        assert result == "Part 1\nPart 2"

    async def test_cleanup(self, sample_mcp_config: MCPConfig) -> None:
        """Test cleanup functionality."""
        client = MCPClient(sample_mcp_config)

        # Mock exit stack
        mock_exit_stack = AsyncMock()
        client.exit_stack = mock_exit_stack

        await client.cleanup()

        mock_exit_stack.aclose.assert_called_once()


class TestQueryProcessor:
    """Test cases for QueryProcessor."""

    def test_init(self, sample_mcp_config: MCPConfig) -> None:
        """Test QueryProcessor initialization."""
        mock_client = MagicMock()
        mock_provider = MagicMock()

        processor = QueryProcessor(mock_client, mock_provider, max_tokens=500)

        assert processor.mcp_client == mock_client
        assert processor.llm_provider == mock_provider
        assert processor.max_tokens == 500

    async def test_process_query_not_connected(self, sample_mcp_config: MCPConfig) -> None:
        """Test processing query when not connected."""
        mock_client = MagicMock()
        mock_client.session = None
        mock_provider = MagicMock()

        processor = QueryProcessor(mock_client, mock_provider)

        with pytest.raises(RuntimeError, match="Not connected to MCP server"):
            await processor.process_query("test query")

    async def test_process_query_no_tools(self, sample_mcp_config: MCPConfig) -> None:
        """Test processing query with no tools."""
        mock_client = MagicMock()
        mock_client.session = MagicMock()
        mock_client.available_tools = []
        mock_client.verbose = False

        mock_provider = AsyncMock()
        mock_response = LLMResponse(text_content=["Test response"], tool_calls=[], raw_response={})
        mock_provider.create_message.return_value = mock_response
        mock_provider.format_tool_for_provider.return_value = {}

        processor = QueryProcessor(mock_client, mock_provider)

        result = await processor.process_query("test query")

        assert result == "Test response"
        mock_provider.create_message.assert_called_once()

    async def test_process_query_with_tool_calls(self, sample_mcp_config: MCPConfig) -> None:
        """Test processing query with tool calls."""
        mock_client = MagicMock()
        mock_client.session = MagicMock()
        mock_client.available_tools = [{"name": "test_tool", "description": "Test tool"}]
        mock_client.verbose = False
        mock_client.execute_tool = AsyncMock(return_value="Tool result")

        mock_provider = AsyncMock()

        # First response with tool call
        mock_response1 = LLMResponse(
            text_content=["I'll use the tool"],
            tool_calls=[{"id": "call_1", "name": "test_tool", "input": {"param": "value"}}],
            raw_response={},
        )

        # Second response after tool execution
        mock_response2 = LLMResponse(
            text_content=["Final response"], tool_calls=[], raw_response={}
        )

        mock_provider.create_message.side_effect = [mock_response1, mock_response2]
        mock_provider.format_tool_for_provider.return_value = {"name": "test_tool"}

        processor = QueryProcessor(mock_client, mock_provider)

        result = await processor.process_query("test query")

        assert "I'll use the tool" in result
        assert "Final response" in result
        mock_client.execute_tool.assert_called_once_with("test_tool", {"param": "value"})

    async def test_process_query_tool_execution_error(self, sample_mcp_config: MCPConfig) -> None:
        """Test processing query with tool execution error."""
        mock_client = MagicMock()
        mock_client.session = MagicMock()
        mock_client.available_tools = [{"name": "test_tool", "description": "Test tool"}]
        mock_client.verbose = False
        mock_client.execute_tool = AsyncMock(side_effect=Exception("Tool failed"))

        mock_provider = AsyncMock()
        mock_response = LLMResponse(
            text_content=["I'll use the tool"],
            tool_calls=[{"id": "call_1", "name": "test_tool", "input": {"param": "value"}}],
            raw_response={},
        )
        mock_provider.create_message.return_value = mock_response
        mock_provider.format_tool_for_provider.return_value = {"name": "test_tool"}

        processor = QueryProcessor(mock_client, mock_provider)

        result = await processor.process_query("test query")

        assert "Error executing tool test_tool: Tool failed" in result

    async def test_process_query_verbose_mode(self, sample_mcp_config: MCPConfig) -> None:
        """Test processing query with verbose logging."""
        mock_client = MagicMock()
        mock_client.session = MagicMock()
        mock_client.available_tools = []
        mock_client.verbose = True
        mock_client.logger = MagicMock()

        mock_provider = AsyncMock()
        mock_response = LLMResponse(text_content=["Test response"], tool_calls=[], raw_response={})
        mock_provider.create_message.return_value = mock_response
        mock_provider.__class__.__name__ = "TestProvider"
        mock_provider.model = "test-model"

        processor = QueryProcessor(mock_client, mock_provider)

        result = await processor.process_query("test query")

        assert result == "Test response"
        # Verify debug logging was called
        assert mock_client.logger.debug.call_count > 0

    def test_format_tool_use_message_anthropic(self, sample_mcp_config: MCPConfig) -> None:
        """Test formatting tool use message for Anthropic provider."""
        mock_client = MagicMock()
        mock_provider = MagicMock(spec=AnthropicProvider)

        processor = QueryProcessor(mock_client, mock_provider)

        result = processor._format_tool_use_message("call_1", "test_tool", {"param": "value"})

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "tool_use"
        assert result[1]["id"] == "call_1"
        assert result[1]["name"] == "test_tool"

    def test_format_tool_use_message_openai(self, sample_mcp_config: MCPConfig) -> None:
        """Test formatting tool use message for OpenAI provider."""
        mock_client = MagicMock()
        mock_provider = MagicMock()

        processor = QueryProcessor(mock_client, mock_provider)

        result = processor._format_tool_use_message("call_1", "test_tool", {"param": "value"})

        assert isinstance(result, str)
        assert "test_tool" in result

    def test_format_tool_result_message_anthropic(self, sample_mcp_config: MCPConfig) -> None:
        """Test formatting tool result message for Anthropic provider."""
        mock_client = MagicMock()
        mock_provider = MagicMock(spec=AnthropicProvider)

        processor = QueryProcessor(mock_client, mock_provider)

        result = processor._format_tool_result_message("call_1", "Tool result")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "tool_result"
        assert result[0]["tool_use_id"] == "call_1"
        assert result[0]["content"] == "Tool result"

    def test_format_tool_result_message_openai(self, sample_mcp_config: MCPConfig) -> None:
        """Test formatting tool result message for OpenAI provider."""
        mock_client = MagicMock()
        mock_provider = MagicMock()

        processor = QueryProcessor(mock_client, mock_provider)

        result = processor._format_tool_result_message("call_1", "Tool result")

        assert isinstance(result, str)
        assert "Tool result" in result

    async def test_process_query_empty_response(self, sample_mcp_config: MCPConfig) -> None:
        """Test processing query with empty response."""
        mock_client = MagicMock()
        mock_client.session = MagicMock()
        mock_client.available_tools = []
        mock_client.verbose = False

        mock_provider = AsyncMock()
        mock_response = LLMResponse(text_content=[], tool_calls=[], raw_response={})
        mock_provider.create_message.return_value = mock_response
        mock_provider.format_tool_for_provider.return_value = {}

        processor = QueryProcessor(mock_client, mock_provider)

        result = await processor.process_query("test query")

        assert result == "No response generated."
