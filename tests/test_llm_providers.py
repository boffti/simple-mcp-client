"""Tests for LLM providers."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_providers import (AnthropicProvider, BaseLLMProvider, LLMMessage,
                           LLMProvider, LLMResponse, OpenAIProvider,
                           OpenRouterProvider, create_llm_provider)


class TestLLMProvider:
    """Test cases for LLMProvider enum."""

    def test_enum_values(self):
        """Test LLMProvider enum values."""
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.OPENROUTER.value == "openrouter"

    def test_from_string(self):
        """Test creating LLMProvider from string."""
        assert LLMProvider("anthropic") == LLMProvider.ANTHROPIC
        assert LLMProvider("openai") == LLMProvider.OPENAI
        assert LLMProvider("openrouter") == LLMProvider.OPENROUTER

    def test_invalid_provider(self):
        """Test creating LLMProvider with invalid string."""
        with pytest.raises(ValueError):
            LLMProvider("invalid")


class TestLLMMessage:
    """Test cases for LLMMessage dataclass."""

    def test_create_message(self):
        """Test creating LLMMessage."""
        message = LLMMessage(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"

    def test_create_message_with_defaults(self):
        """Test creating LLMMessage with default values."""
        message = LLMMessage(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"


class TestLLMResponse:
    """Test cases for LLMResponse dataclass."""

    def test_create_response(self):
        """Test creating LLMResponse."""
        response = LLMResponse(
            text_content=["Hello", "World"],
            tool_calls=[{"name": "test", "input": {}}],
            raw_response={"status": "success"},
        )
        assert response.text_content == ["Hello", "World"]
        assert response.tool_calls == [{"name": "test", "input": {}}]
        assert response.raw_response == {"status": "success"}

    def test_create_response_with_defaults(self):
        """Test creating LLMResponse with default values."""
        response = LLMResponse(text_content=["Hello"], tool_calls=[], raw_response={})
        assert response.text_content == ["Hello"]
        assert response.tool_calls == []
        assert response.raw_response == {}


class TestBaseLLMProvider:
    """Test cases for BaseLLMProvider abstract class."""

    def test_cannot_instantiate(self):
        """Test that BaseLLMProvider cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseLLMProvider("test-key")

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""

        class IncompleteProvider(BaseLLMProvider):
            def __init__(self, api_key: str):
                super().__init__(api_key)

        with pytest.raises(TypeError):
            IncompleteProvider("test-key")


class TestAnthropicProvider:
    """Test cases for AnthropicProvider."""

    def test_init(self):
        """Test AnthropicProvider initialization."""
        provider = AnthropicProvider("test-key", "claude-3-5-sonnet-20241022")
        assert provider.api_key == "test-key"
        assert provider.model == "claude-3-5-sonnet-20241022"

    def test_init_default_model(self):
        """Test AnthropicProvider with default model."""
        provider = AnthropicProvider("test-key")
        assert provider.model == "claude-3-5-sonnet-20241022"

    def test_format_tool_for_provider(self):
        """Test formatting tool for Anthropic provider."""
        provider = AnthropicProvider("test-key")

        tool = {
            "name": "test_tool",
            "description": "Test tool",
            "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}},
        }

        formatted = provider.format_tool_for_provider(tool)

        assert formatted["name"] == "test_tool"
        assert formatted["description"] == "Test tool"
        assert formatted["input_schema"] == tool["input_schema"]

    @patch("llm_providers.Anthropic")
    async def test_create_message_success(self, mock_anthropic_class):
        """Test successful message creation."""
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello world")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider("test-key")
        messages = [LLMMessage(role="user", content="Hello")]

        result = await provider.create_message(messages)

        assert isinstance(result, LLMResponse)
        assert result.text_content == ["Hello world"]
        assert result.tool_calls == []
        assert "input_tokens" in result.raw_response
        assert "output_tokens" in result.raw_response

    @patch("llm_providers.Anthropic")
    async def test_create_message_with_tools(self, mock_anthropic_class):
        """Test message creation with tools."""
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock response with tool use
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text="I'll use the tool"),
            MagicMock(type="tool_use", id="call_1", name="test_tool", input={"param": "value"}),
        ]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider("test-key")
        messages = [LLMMessage(role="user", content="Hello")]
        tools = [{"name": "test_tool", "description": "Test"}]

        result = await provider.create_message(messages, tools)

        assert len(result.text_content) == 1
        assert result.text_content[0] == "I'll use the tool"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "test_tool"
        assert result.tool_calls[0]["id"] == "call_1"

    @patch("llm_providers.Anthropic")
    async def test_create_message_api_error(self, mock_anthropic_class):
        """Test message creation with API error."""
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock API error
        mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))

        provider = AnthropicProvider("test-key")
        messages = [LLMMessage(role="user", content="Hello")]

        with pytest.raises(Exception, match="API Error"):
            await provider.create_message(messages)

    def test_format_messages_for_anthropic(self):
        """Test formatting messages for Anthropic API."""
        provider = AnthropicProvider("test-key")

        messages = [
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi there"),
            LLMMessage(role="user", content="How are you?"),
        ]

        formatted = provider._format_messages_for_anthropic(messages)

        assert len(formatted) == 3
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"] == "Hello"
        assert formatted[1]["role"] == "assistant"
        assert formatted[1]["content"] == "Hi there"

    def test_format_messages_with_mixed_content(self):
        """Test formatting messages with mixed content types."""
        provider = AnthropicProvider("test-key")

        messages = [
            LLMMessage(role="user", content="Hello"),
            LLMMessage(
                role="assistant",
                content=[
                    {"type": "text", "text": "I'll use a tool"},
                    {"type": "tool_use", "id": "call_1", "name": "test"},
                ],
            ),
        ]

        formatted = provider._format_messages_for_anthropic(messages)

        assert len(formatted) == 2
        assert formatted[0]["content"] == "Hello"
        assert isinstance(formatted[1]["content"], list)
        assert len(formatted[1]["content"]) == 2


class TestOpenAIProvider:
    """Test cases for OpenAIProvider."""

    def test_init(self):
        """Test OpenAIProvider initialization."""
        provider = OpenAIProvider("test-key", "gpt-4")
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4"

    def test_init_default_model(self):
        """Test OpenAIProvider with default model."""
        provider = OpenAIProvider("test-key")
        assert provider.model == "gpt-4-turbo"

    def test_format_tool_for_provider(self):
        """Test formatting tool for OpenAI provider."""
        provider = OpenAIProvider("test-key")

        tool = {
            "name": "test_tool",
            "description": "Test tool",
            "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}},
        }

        formatted = provider.format_tool_for_provider(tool)

        assert formatted["type"] == "function"
        assert formatted["function"]["name"] == "test_tool"
        assert formatted["function"]["description"] == "Test tool"
        assert formatted["function"]["parameters"] == tool["input_schema"]

    @patch("llm_providers.OpenAI")
    async def test_create_message_success(self, mock_openai_class):
        """Test successful message creation."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hello world", tool_calls=None))
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider("test-key")
        messages = [LLMMessage(role="user", content="Hello")]

        result = await provider.create_message(messages)

        assert isinstance(result, LLMResponse)
        assert result.text_content == ["Hello world"]
        assert result.tool_calls == []

    @patch("llm_providers.OpenAI")
    async def test_create_message_with_tool_calls(self, mock_openai_class):
        """Test message creation with tool calls."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock response with tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"param": "value"}'

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="I'll use the tool", tool_calls=[mock_tool_call]))
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider("test-key")
        messages = [LLMMessage(role="user", content="Hello")]
        tools = [{"name": "test_tool", "description": "Test"}]

        result = await provider.create_message(messages, tools)

        assert result.text_content == ["I'll use the tool"]
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "test_tool"
        assert result.tool_calls[0]["id"] == "call_1"
        assert result.tool_calls[0]["input"] == {"param": "value"}


class TestOpenRouterProvider:
    """Test cases for OpenRouterProvider."""

    def test_init(self):
        """Test OpenRouterProvider initialization."""
        provider = OpenRouterProvider("test-key", "anthropic/claude-3.5-sonnet")
        assert provider.api_key == "test-key"
        assert provider.model == "anthropic/claude-3.5-sonnet"

    def test_init_default_model(self):
        """Test OpenRouterProvider with default model."""
        provider = OpenRouterProvider("test-key")
        assert provider.model == "anthropic/claude-3.5-sonnet"

    def test_format_tool_for_provider(self):
        """Test formatting tool for OpenRouter provider."""
        provider = OpenRouterProvider("test-key")

        tool = {
            "name": "test_tool",
            "description": "Test tool",
            "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}},
        }

        formatted = provider.format_tool_for_provider(tool)

        # Should use same format as OpenAI
        assert formatted["type"] == "function"
        assert formatted["function"]["name"] == "test_tool"
        assert formatted["function"]["description"] == "Test tool"
        assert formatted["function"]["parameters"] == tool["input_schema"]

    @patch("llm_providers.OpenAI")
    async def test_create_message_success(self, mock_openai_class):
        """Test successful message creation."""
        # Mock OpenAI client (OpenRouter uses OpenAI client)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hello world", tool_calls=None))
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenRouterProvider("test-key")
        messages = [LLMMessage(role="user", content="Hello")]

        result = await provider.create_message(messages)

        assert isinstance(result, LLMResponse)
        assert result.text_content == ["Hello world"]
        assert result.tool_calls == []

    def test_openrouter_base_url(self):
        """Test that OpenRouter uses correct base URL."""
        provider = OpenRouterProvider("test-key")

        with patch("llm_providers.OpenAI") as mock_openai:
            provider._get_client()

            mock_openai.assert_called_once_with(
                api_key="test-key", base_url="https://openrouter.ai/api/v1"
            )


class TestCreateLLMProvider:
    """Test cases for create_llm_provider function."""

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        provider = create_llm_provider(
            LLMProvider.ANTHROPIC, "test-key", "claude-3-5-sonnet-20241022"
        )

        assert isinstance(provider, AnthropicProvider)
        assert provider.api_key == "test-key"
        assert provider.model == "claude-3-5-sonnet-20241022"

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        provider = create_llm_provider(LLMProvider.OPENAI, "test-key", "gpt-4")

        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4"

    def test_create_openrouter_provider(self):
        """Test creating OpenRouter provider."""
        provider = create_llm_provider(
            LLMProvider.OPENROUTER, "test-key", "anthropic/claude-3.5-sonnet"
        )

        assert isinstance(provider, OpenRouterProvider)
        assert provider.api_key == "test-key"
        assert provider.model == "anthropic/claude-3.5-sonnet"

    def test_create_provider_with_options(self):
        """Test creating provider with additional options."""
        provider = create_llm_provider(
            LLMProvider.ANTHROPIC,
            "test-key",
            "claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=1500,
        )

        assert isinstance(provider, AnthropicProvider)
        assert provider.api_key == "test-key"
        assert provider.model == "claude-3-5-sonnet-20241022"

    def test_create_provider_invalid_type(self):
        """Test creating provider with invalid type."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            # Create a mock enum value that's not supported
            invalid_provider = MagicMock()
            invalid_provider.value = "invalid"
            create_llm_provider(invalid_provider, "test-key")
