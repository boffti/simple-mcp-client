"""Tests for configuration management."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from config import (
    LLMConfig,
    MCPConfig,
    load_llm_config,
    load_mcp_config,
    load_legacy_config,
    get_api_key_from_env,
    auto_detect_provider_from_env,
    validate_llm_config,
    save_llm_config,
    save_mcp_config,
)
from llm_providers import LLMProvider


class TestLLMConfig:
    """Test cases for LLMConfig."""

    def test_init_defaults(self):
        """Test LLMConfig initialization with defaults."""
        config = LLMConfig()
        assert config.provider == "anthropic"
        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.max_tokens == 1000
        assert config.options == {}

    def test_init_with_values(self):
        """Test LLMConfig initialization with custom values."""
        config = LLMConfig(
            provider="openai", model="gpt-4", max_tokens=2000, options={"temperature": 0.7}
        )
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.max_tokens == 2000
        assert config.options == {"temperature": 0.7}

    def test_init_openai_default_model(self):
        """Test LLMConfig with OpenAI provider default model."""
        config = LLMConfig(provider="openai")
        assert config.model == "gpt-4-turbo"

    def test_init_openrouter_default_model(self):
        """Test LLMConfig with OpenRouter provider default model."""
        config = LLMConfig(provider="openrouter")
        assert config.model == "anthropic/claude-3.5-sonnet"

    def test_post_init_none_options(self):
        """Test post_init with None options."""
        config = LLMConfig(options=None)
        assert config.options == {}


class TestMCPConfig:
    """Test cases for MCPConfig."""

    def test_init_defaults(self):
        """Test MCPConfig initialization with defaults."""
        config = MCPConfig()
        assert config.servers == {}

    def test_init_with_servers(self):
        """Test MCPConfig initialization with servers."""
        servers = {"test-server": {"command": "python", "args": ["-m", "test_server"]}}
        config = MCPConfig(servers=servers)
        assert config.servers == servers

    def test_post_init_none_servers(self):
        """Test post_init with None servers."""
        config = MCPConfig(servers=None)
        assert config.servers == {}


class TestLoadLLMConfig:
    """Test cases for load_llm_config function."""

    def test_load_valid_config(self):
        """Test loading valid LLM config file."""
        config_data = {
            "provider": "openai",
            "model": "gpt-4",
            "max_tokens": 1500,
            "options": {"temperature": 0.5},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_llm_config(temp_path)
            assert config.provider == "openai"
            assert config.model == "gpt-4"
            assert config.max_tokens == 1500
            assert config.options == {"temperature": 0.5}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_missing_file(self):
        """Test loading missing LLM config file."""
        config = load_llm_config("nonexistent.json")
        assert isinstance(config, LLMConfig)
        assert config.provider == "anthropic"  # default

    def test_load_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json {")
            temp_path = f.name

        try:
            config = load_llm_config(temp_path)
            assert isinstance(config, LLMConfig)
            assert config.provider == "anthropic"  # default
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_partial_config(self):
        """Test loading partial config file."""
        config_data = {"provider": "openai"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_llm_config(temp_path)
            assert config.provider == "openai"
            assert config.model == "gpt-4-turbo"  # default for openai
            assert config.max_tokens == 1000  # default
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestLoadMCPConfig:
    """Test cases for load_mcp_config function."""

    def test_load_valid_config_new_format(self):
        """Test loading valid MCP config file (new format)."""
        config_data = {
            "servers": {"test-server": {"command": "python", "args": ["-m", "test_server"]}}
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_mcp_config(temp_path)
            assert "test-server" in config.servers
            assert config.servers["test-server"]["command"] == "python"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_valid_config_old_format(self):
        """Test loading valid MCP config file (old format)."""
        config_data = {
            "mcpServers": {"test-server": {"command": "python", "args": ["-m", "test_server"]}}
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_mcp_config(temp_path)
            assert "test-server" in config.servers
            assert config.servers["test-server"]["command"] == "python"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_missing_file(self):
        """Test loading missing MCP config file."""
        config = load_mcp_config("nonexistent.json")
        assert isinstance(config, MCPConfig)
        assert config.servers == {}

    def test_load_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json {")
            temp_path = f.name

        try:
            config = load_mcp_config(temp_path)
            assert isinstance(config, MCPConfig)
            assert config.servers == {}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_empty_config(self):
        """Test loading empty config file."""
        config_data = {}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_mcp_config(temp_path)
            assert config.servers == {}
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestLoadLegacyConfig:
    """Test cases for load_legacy_config function."""

    def test_load_legacy_combined_config(self):
        """Test loading legacy combined config file."""
        config_data = {
            "mcpServers": {"test-server": {"command": "python", "args": ["-m", "test_server"]}},
            "llm": {"provider": "openai", "model": "gpt-4"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            mcp_config, llm_config = load_legacy_config(temp_path)
            assert "test-server" in mcp_config.servers
            assert llm_config.provider == "openai"
            assert llm_config.model == "gpt-4"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_legacy_mcp_only(self):
        """Test loading legacy config with only MCP servers."""
        config_data = {
            "mcpServers": {"test-server": {"command": "python", "args": ["-m", "test_server"]}}
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            mcp_config, llm_config = load_legacy_config(temp_path)
            assert "test-server" in mcp_config.servers
            assert llm_config.provider == "anthropic"  # default
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_legacy_missing_file(self):
        """Test loading missing legacy config file."""
        mcp_config, llm_config = load_legacy_config("nonexistent.json")
        assert isinstance(mcp_config, MCPConfig)
        assert isinstance(llm_config, LLMConfig)
        assert mcp_config.servers == {}
        assert llm_config.provider == "anthropic"


class TestGetApiKeyFromEnv:
    """Test cases for get_api_key_from_env function."""

    def test_anthropic_api_key(self):
        """Test getting Anthropic API key from environment."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            key = get_api_key_from_env("anthropic")
            assert key == "test-key"

    def test_openai_api_key(self):
        """Test getting OpenAI API key from environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            key = get_api_key_from_env("openai")
            assert key == "test-key"

    def test_openrouter_api_key(self):
        """Test getting OpenRouter API key from environment."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            key = get_api_key_from_env("openrouter")
            assert key == "test-key"

    def test_unknown_provider(self):
        """Test getting API key for unknown provider."""
        key = get_api_key_from_env("unknown")
        assert key is None

    def test_missing_api_key(self):
        """Test getting API key when not set."""
        with patch.dict(os.environ, {}, clear=True):
            key = get_api_key_from_env("anthropic")
            assert key is None


class TestAutoDetectProviderFromEnv:
    """Test cases for auto_detect_provider_from_env function."""

    def test_detect_anthropic(self):
        """Test auto-detecting Anthropic provider."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            provider, key = auto_detect_provider_from_env()
            assert provider == "anthropic"
            assert key == "test-key"

    def test_detect_openai(self):
        """Test auto-detecting OpenAI provider."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            provider, key = auto_detect_provider_from_env()
            assert provider == "openai"
            assert key == "test-key"

    def test_detect_openrouter(self):
        """Test auto-detecting OpenRouter provider."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            provider, key = auto_detect_provider_from_env()
            assert provider == "openrouter"
            assert key == "test-key"

    def test_detect_priority_order(self):
        """Test provider detection priority order."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "anthropic-key",
                "OPENAI_API_KEY": "openai-key",
                "OPENROUTER_API_KEY": "openrouter-key",
            },
            clear=True,
        ):
            provider, key = auto_detect_provider_from_env()
            assert provider == "anthropic"  # Should prefer Anthropic
            assert key == "anthropic-key"

    def test_detect_no_keys(self):
        """Test auto-detection when no API keys are set."""
        with patch.dict(os.environ, {}, clear=True):
            provider, key = auto_detect_provider_from_env()
            assert provider is None
            assert key is None


class TestValidateLLMConfig:
    """Test cases for validate_llm_config function."""

    def test_validate_valid_config(self):
        """Test validating valid LLM config."""
        config = LLMConfig(provider="anthropic", max_tokens=1000)
        # Should not raise any exception
        validate_llm_config(config)

    def test_validate_invalid_provider(self):
        """Test validating config with invalid provider."""
        config = LLMConfig(provider="invalid", max_tokens=1000)
        with pytest.raises(ValueError, match="Invalid LLM provider"):
            validate_llm_config(config)

    def test_validate_invalid_max_tokens(self):
        """Test validating config with invalid max_tokens."""
        config = LLMConfig(provider="anthropic", max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            validate_llm_config(config)

    def test_validate_negative_max_tokens(self):
        """Test validating config with negative max_tokens."""
        config = LLMConfig(provider="anthropic", max_tokens=-100)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            validate_llm_config(config)


class TestSaveLLMConfig:
    """Test cases for save_llm_config function."""

    def test_save_config(self):
        """Test saving LLM config to file."""
        config = LLMConfig(
            provider="openai", model="gpt-4", max_tokens=1500, options={"temperature": 0.7}
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_llm_config(config, temp_path)

            # Verify file was created and contains correct data
            with open(temp_path, "r") as f:
                saved_data = json.load(f)

            assert saved_data["provider"] == "openai"
            assert saved_data["model"] == "gpt-4"
            assert saved_data["max_tokens"] == 1500
            assert saved_data["options"] == {"temperature": 0.7}
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSaveMCPConfig:
    """Test cases for save_mcp_config function."""

    def test_save_config(self):
        """Test saving MCP config to file."""
        config = MCPConfig(
            servers={
                "test-server": {
                    "command": "python",
                    "args": ["-m", "test_server"],
                    "description": "Test server",
                }
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_mcp_config(config, temp_path)

            # Verify file was created and contains correct data
            with open(temp_path, "r") as f:
                saved_data = json.load(f)

            assert "mcpServers" in saved_data
            assert "test-server" in saved_data["mcpServers"]
            assert saved_data["mcpServers"]["test-server"]["command"] == "python"
        finally:
            Path(temp_path).unlink(missing_ok=True)
