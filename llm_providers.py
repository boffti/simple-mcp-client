"""
LLM Provider Module
==================

This module contains the LLM provider implementations for different AI services.
Supports Anthropic, OpenAI, and OpenRouter providers with a unified interface.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from anthropic import Anthropic


class LLMProvider(Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""

    text_content: list[str]
    tool_calls: list[dict[str, Any]]
    raw_response: Any


@dataclass
class LLMMessage:
    """Standardized message format for LLM providers."""

    role: str
    content: Any


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    async def create_message(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Create a message with the LLM provider."""
        pass

    @abstractmethod
    def format_tool_for_provider(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Format MCP tool for the specific provider."""
        pass


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = Anthropic(api_key=api_key)

    async def create_message(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Create a message using Anthropic's API."""
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append({"role": msg.role, "content": msg.content})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=anthropic_messages,
            tools=tools if tools else None,
        )

        # Parse response
        text_content = []
        tool_calls = []

        for content in response.content:
            if content.type == "text":
                text_content.append(content.text)
            elif content.type == "tool_use":
                tool_calls.append({"id": content.id, "name": content.name, "input": content.input})

        return LLMResponse(text_content=text_content, tool_calls=tool_calls, raw_response=response)

    def format_tool_for_provider(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Format tool for Anthropic API."""
        return {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["input_schema"],
        }


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""

    def __init__(
        self, api_key: str, model: str = "gpt-4-turbo", base_url: str | None = None, **kwargs
    ):
        super().__init__(api_key, model, **kwargs)
        try:
            import openai

            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        except ImportError as err:
            raise ImportError(
                "openai package is required for OpenAI provider. Install with: pip install openai"
            ) from err

    async def create_message(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Create a message using OpenAI's API."""
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_messages.append({"role": msg.role, "content": msg.content})

        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)

        # Parse response
        text_content = []
        tool_calls = []

        message = response.choices[0].message
        if message.content:
            text_content.append(message.content)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": json.loads(tool_call.function.arguments),
                    }
                )

        return LLMResponse(text_content=text_content, tool_calls=tool_calls, raw_response=response)

    def format_tool_for_provider(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Format tool for OpenAI API."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider implementation (OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet", **kwargs):
        super().__init__(
            api_key=api_key, model=model, base_url="https://openrouter.ai/api/v1", **kwargs
        )


def create_llm_provider(
    provider: LLMProvider, api_key: str, model: str | None = None, **kwargs
) -> BaseLLMProvider:
    """Factory function to create LLM providers."""
    if provider == LLMProvider.ANTHROPIC:
        return AnthropicProvider(api_key, model or "claude-3-5-sonnet-20241022", **kwargs)
    elif provider == LLMProvider.OPENAI:
        return OpenAIProvider(api_key, model or "gpt-4-turbo", **kwargs)
    elif provider == LLMProvider.OPENROUTER:
        return OpenRouterProvider(api_key, model or "anthropic/claude-3.5-sonnet", **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
