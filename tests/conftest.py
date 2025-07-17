"""Test configuration and fixtures for Simple MCP Client tests."""

import asyncio
import json
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import ClientSession

from config import LLMConfig, MCPConfig
from simple_mcp_client import SimpleMCPClient


@pytest.fixture
def sample_mcp_config() -> MCPConfig:
    """Sample MCP configuration for testing."""
    return MCPConfig(
        servers={
            "test-server": {
                "command": "python",
                "args": ["-m", "test_server"],
                "description": "Test server",
            },
            "filesystem": {
                "command": "python",
                "args": ["-m", "mcp_server_filesystem"],
                "description": "Filesystem server",
                "env": {"TEST_VAR": "test_value"},
            },
        }
    )


@pytest.fixture
def sample_llm_config() -> LLMConfig:
    """Sample LLM configuration for testing."""
    return LLMConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        options={"temperature": 0.7},
    )


@pytest.fixture
def sample_tools() -> list[dict[str, Any]]:
    """Sample tools for testing."""
    return [
        {
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}},
        },
        {
            "name": "another_tool",
            "description": "Another test tool",
            "input_schema": {"type": "object", "properties": {"value": {"type": "number"}}},
        },
    ]


@pytest.fixture
def mock_client_session() -> AsyncMock:
    """Mock MCP client session."""
    mock_session = AsyncMock(spec=ClientSession)
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock()
    mock_session.call_tool = AsyncMock()
    return mock_session


@pytest.fixture
def mock_stdio_client():
    """Mock stdio client for MCP connections."""
    mock_client = AsyncMock()

    # Create mock context manager that returns read_stream, write_stream
    async def async_context_manager():
        return (AsyncMock(), AsyncMock())

    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=async_context_manager())
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client.return_value = mock_context
    return mock_client


@pytest.fixture
def temp_config_file(sample_mcp_config: MCPConfig) -> AsyncGenerator[str, None]:
    """Create temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"mcpServers": sample_mcp_config.servers}, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_llm_config_file(sample_llm_config: LLMConfig) -> AsyncGenerator[str, None]:
    """Create temporary LLM config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "provider": sample_llm_config.provider,
                "model": sample_llm_config.model,
                "max_tokens": sample_llm_config.max_tokens,
                "options": sample_llm_config.options,
            },
            f,
        )
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def simple_mcp_client(sample_mcp_config: MCPConfig) -> SimpleMCPClient:
    """Create SimpleMCPClient instance for testing."""
    return SimpleMCPClient(config=sample_mcp_config)


@pytest.fixture
async def connected_client(
    simple_mcp_client: SimpleMCPClient,
    mock_client_session: AsyncMock,
    mock_stdio_client: AsyncMock,
    sample_tools: list[dict[str, Any]],
) -> AsyncGenerator[SimpleMCPClient, None]:
    """Create connected SimpleMCPClient for testing."""
    # Mock the stdio client
    import simple_mcp_client as smc_module

    original_stdio_client = smc_module.stdio_client
    smc_module.stdio_client = mock_stdio_client

    # Mock the ClientSession
    original_client_session = smc_module.ClientSession
    smc_module.ClientSession = lambda *args, **kwargs: mock_client_session

    # Mock list_tools response
    mock_tools_response = MagicMock()
    mock_tools_response.tools = []
    for tool in sample_tools:
        mock_tool = MagicMock()
        mock_tool.name = tool["name"]
        mock_tool.description = tool["description"]
        mock_tool.inputSchema = tool["input_schema"]
        mock_tools_response.tools.append(mock_tool)

    mock_client_session.list_tools.return_value = mock_tools_response

    # Connect the client
    await simple_mcp_client.connect_to_server("test-server")

    yield simple_mcp_client

    # Cleanup
    await simple_mcp_client.cleanup()
    smc_module.stdio_client = original_stdio_client
    smc_module.ClientSession = original_client_session


@pytest.fixture
def mock_tool_result():
    """Mock tool execution result."""
    mock_result = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "Test result"
    mock_result.content = [mock_content]
    return mock_result


@pytest.fixture(autouse=True)
def reset_async_exit_stack():
    """Reset AsyncExitStack for each test."""
    yield
    # Any cleanup needed for AsyncExitStack
