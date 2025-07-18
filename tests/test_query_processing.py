"""
Tests for query processing functionality.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llm_providers import AnthropicProvider, OpenAIProvider, LLMMessage, LLMResponse
from query_processing import MultiServerQueryProcessor, QueryProcessingUtils


class TestQueryProcessingUtils:
    """Test QueryProcessingUtils static methods."""

    def test_format_tool_input_for_display_sql(self):
        """Test SQL query formatting."""
        result = QueryProcessingUtils.format_tool_input_for_display(
            "query", {"sql": "SELECT * FROM users\\nWHERE active = true"}
        )
        expected = '{\n  "sql": """\nSELECT * FROM users\nWHERE active = true\n  """\n}'
        assert result == expected

    def test_format_tool_input_for_display_regular(self):
        """Test regular tool input formatting."""
        result = QueryProcessingUtils.format_tool_input_for_display(
            "test_tool", {"param": "value", "number": 42}
        )
        expected = json.dumps({"param": "value", "number": 42}, indent=2)
        assert result == expected

    def test_format_tool_use_message_anthropic(self):
        """Test tool use message formatting for Anthropic."""
        mock_provider = MagicMock(spec=AnthropicProvider)
        result = QueryProcessingUtils.format_tool_use_message(
            "tool_123", "test_tool", {"param": "value"}, mock_provider
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "tool_use"
        assert result[1]["id"] == "tool_123"
        assert result[1]["name"] == "test_tool"

    def test_format_tool_use_message_openai(self):
        """Test tool use message formatting for OpenAI."""
        mock_provider = MagicMock(spec=OpenAIProvider)
        result = QueryProcessingUtils.format_tool_use_message(
            "tool_123", "test_tool", {"param": "value"}, mock_provider
        )

        assert isinstance(result, str)
        assert "test_tool" in result

    def test_format_tool_result_message_anthropic(self):
        """Test tool result message formatting for Anthropic."""
        mock_provider = MagicMock(spec=AnthropicProvider)
        result = QueryProcessingUtils.format_tool_result_message(
            "tool_123", "Tool result content", mock_provider
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "tool_result"
        assert result[0]["tool_use_id"] == "tool_123"
        assert result[0]["content"] == "Tool result content"

    def test_format_tool_result_message_openai(self):
        """Test tool result message formatting for OpenAI."""
        mock_provider = MagicMock(spec=OpenAIProvider)
        result = QueryProcessingUtils.format_tool_result_message(
            "tool_123", "Tool result content", mock_provider
        )

        assert isinstance(result, str)
        assert "Tool result content" in result

    def test_format_tool_use_messages_single(self):
        """Test formatting single tool use message."""
        tool_calls = [{"id": "tool_123", "name": "test_tool", "input": {"param": "value"}}]
        result = QueryProcessingUtils.format_tool_use_messages(tool_calls)

        assert "test_tool" in result
        assert "tool_123" in result
        assert "param" in result

    def test_format_tool_use_messages_multiple(self):
        """Test formatting multiple tool use messages."""
        tool_calls = [
            {"id": "tool_123", "name": "tool_1", "input": {"param": "value1"}},
            {"id": "tool_456", "name": "tool_2", "input": {"param": "value2"}},
        ]
        result = QueryProcessingUtils.format_tool_use_messages(tool_calls)

        assert "tool_1" in result
        assert "tool_2" in result
        assert "following tools" in result

    def test_format_tool_use_messages_empty(self):
        """Test formatting empty tool calls."""
        result = QueryProcessingUtils.format_tool_use_messages([])
        assert result == ""


class TestMultiServerQueryProcessor:
    """Test MultiServerQueryProcessor."""

    @pytest.fixture
    def mock_simple_mcp_client(self):
        """Create a mock SimpleMCPClient."""
        mock_client = MagicMock()
        mock_client.connected_servers = {"server1"}
        mock_client.get_available_tools.return_value = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}},
                "server": "server1",
            }
        ]
        mock_client.execute_tool = AsyncMock(return_value="Tool result")
        return mock_client

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        mock_provider = MagicMock()
        mock_provider.format_tool_for_provider.return_value = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {"param": {"type": "string"}}},
        }
        return mock_provider

    def test_init(self, mock_simple_mcp_client, mock_llm_provider):
        """Test MultiServerQueryProcessor initialization."""
        processor = MultiServerQueryProcessor(
            mock_simple_mcp_client, mock_llm_provider, max_tokens=500, verbose=True
        )

        assert processor.simple_mcp_client == mock_simple_mcp_client
        assert processor.llm_provider == mock_llm_provider
        assert processor.max_tokens == 500
        assert processor.verbose is True

    @pytest.mark.asyncio
    async def test_process_query_not_connected(self, mock_llm_provider):
        """Test process_query when not connected to any servers."""
        mock_client = MagicMock()
        mock_client.connected_servers = set()

        processor = MultiServerQueryProcessor(mock_client, mock_llm_provider)

        with pytest.raises(RuntimeError, match="Not connected to any MCP servers"):
            await processor.process_query("test query")

    @pytest.mark.asyncio
    async def test_process_query_no_tool_calls(self, mock_simple_mcp_client, mock_llm_provider):
        """Test process_query with no tool calls."""
        mock_llm_provider.create_message = AsyncMock(
            return_value=LLMResponse(
                text_content=["This is a response without tool calls"], tool_calls=[]
            )
        )

        processor = MultiServerQueryProcessor(mock_simple_mcp_client, mock_llm_provider)
        response = await processor.process_query("test query")

        assert "This is a response without tool calls" in response
        mock_llm_provider.create_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_with_tool_calls(self, mock_simple_mcp_client, mock_llm_provider):
        """Test process_query with tool calls."""
        # First call returns tool calls, second call returns final response
        mock_llm_provider.create_message = AsyncMock(
            side_effect=[
                LLMResponse(
                    text_content=["I'll use a tool"],
                    tool_calls=[
                        {"id": "tool_123", "name": "test_tool", "input": {"param": "value"}}
                    ],
                ),
                LLMResponse(text_content=["Here's the final response"], tool_calls=[]),
            ]
        )

        processor = MultiServerQueryProcessor(mock_simple_mcp_client, mock_llm_provider)

        with patch("builtins.print"):  # Suppress print output during test
            response = await processor.process_query("test query")

        assert "I'll use a tool" in response
        assert "Here's the final response" in response
        assert mock_llm_provider.create_message.call_count == 2
        mock_simple_mcp_client.execute_tool.assert_called_once_with("test_tool", {"param": "value"})

    @pytest.mark.asyncio
    async def test_process_query_tool_execution_error(
        self, mock_simple_mcp_client, mock_llm_provider
    ):
        """Test process_query when tool execution fails."""
        mock_llm_provider.create_message = AsyncMock(
            side_effect=[
                LLMResponse(
                    text_content=["I'll use a tool"],
                    tool_calls=[
                        {"id": "tool_123", "name": "test_tool", "input": {"param": "value"}}
                    ],
                ),
                LLMResponse(text_content=["Error handled"], tool_calls=[]),
            ]
        )

        mock_simple_mcp_client.execute_tool = AsyncMock(side_effect=Exception("Tool failed"))

        processor = MultiServerQueryProcessor(mock_simple_mcp_client, mock_llm_provider)

        with patch("builtins.print"):  # Suppress print output during test
            response = await processor.process_query("test query")

        assert "I'll use a tool" in response
        assert "Error handled" in response

    @pytest.mark.asyncio
    async def test_process_query_max_iterations(self, mock_simple_mcp_client, mock_llm_provider):
        """Test process_query with max iterations limit."""
        # Always return tool calls to trigger max iterations
        mock_llm_provider.create_message = AsyncMock(
            return_value=LLMResponse(
                text_content=["I'll use a tool"],
                tool_calls=[{"id": "tool_123", "name": "test_tool", "input": {"param": "value"}}],
            )
        )

        processor = MultiServerQueryProcessor(mock_simple_mcp_client, mock_llm_provider)

        with patch("builtins.print"):  # Suppress print output during test
            response = await processor.process_query("test query", max_iterations=2)

        assert "maximum iteration limit (2)" in response
        assert mock_llm_provider.create_message.call_count == 2

    def test_format_tool_input_for_display(self, mock_simple_mcp_client, mock_llm_provider):
        """Test _format_tool_input_for_display method."""
        processor = MultiServerQueryProcessor(mock_simple_mcp_client, mock_llm_provider)

        result = processor._format_tool_input_for_display("test_tool", {"param": "value"})
        expected = json.dumps({"param": "value"}, indent=2)
        assert result == expected

    def test_format_tool_use_messages(self, mock_simple_mcp_client, mock_llm_provider):
        """Test _format_tool_use_messages method."""
        processor = MultiServerQueryProcessor(mock_simple_mcp_client, mock_llm_provider)

        tool_calls = [{"id": "tool_123", "name": "test_tool", "input": {"param": "value"}}]
        result = processor._format_tool_use_messages(tool_calls)

        assert "test_tool" in result
        assert "tool_123" in result
