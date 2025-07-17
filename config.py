"""
Configuration Module
===================

This module handles loading and managing configuration for both MCP servers and LLM providers.
Supports separate configuration files for better organization.
"""

import json
import os
from dataclasses import dataclass
from typing import Any

from llm_providers import LLMProvider


@dataclass
class LLMConfig:
    """LLM configuration structure."""

    provider: str = "anthropic"
    model: str | None = None
    max_tokens: int = 1000
    options: dict[str, Any] = None

    def __post_init__(self):
        if self.options is None:
            self.options = {}

        # Set default models if not specified
        if self.model is None:
            if self.provider == "anthropic":
                self.model = "claude-3-5-sonnet-20241022"
            elif self.provider == "openai":
                self.model = "gpt-4-turbo"
            elif self.provider == "openrouter":
                self.model = "anthropic/claude-3.5-sonnet"


@dataclass
class MCPConfig:
    """MCP configuration structure."""

    servers: dict[str, dict[str, Any]] = None

    def __post_init__(self):
        if self.servers is None:
            self.servers = {}


def load_llm_config(config_path: str = "llm_config.json") -> LLMConfig:
    """Load LLM configuration from JSON file."""
    try:
        with open(config_path) as f:
            config_data = json.load(f)
            return LLMConfig(**config_data)
    except FileNotFoundError:
        print(f"LLM config file {config_path} not found. Using default configuration.")
        return LLMConfig()
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in LLM config file: {e}")
        return LLMConfig()
    except Exception as e:
        print(f"Error loading LLM config: {e}")
        return LLMConfig()


def load_mcp_config(config_path: str = "mcp_config.json") -> MCPConfig:
    """Load MCP configuration from JSON file."""
    try:
        with open(config_path) as f:
            config_data = json.load(f)
            # Support both old and new format for backward compatibility
            if "servers" in config_data:
                servers = config_data["servers"]
            elif "mcpServers" in config_data:
                servers = config_data["mcpServers"]
            else:
                servers = {}

            return MCPConfig(servers=servers)
    except FileNotFoundError:
        print(f"MCP config file {config_path} not found. Using default configuration.")
        return MCPConfig()
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in MCP config file: {e}")
        return MCPConfig()
    except Exception as e:
        print(f"Error loading MCP config: {e}")
        return MCPConfig()


def load_legacy_config(config_path: str = "mcp_config.json") -> tuple[MCPConfig, LLMConfig]:
    """Load legacy combined configuration and split into MCP and LLM configs."""
    try:
        with open(config_path) as f:
            config = json.load(f)

            # Extract MCP configuration
            if "servers" in config:
                mcp_servers = config["servers"]
            elif "mcpServers" in config:
                mcp_servers = config["mcpServers"]
            else:
                mcp_servers = {}

            mcp_config = MCPConfig(servers=mcp_servers)

            # Extract LLM configuration
            if "llm" in config:
                llm_config = LLMConfig(**config["llm"])
            else:
                llm_config = LLMConfig()

            return mcp_config, llm_config

    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default configuration.")
        return MCPConfig(), LLMConfig()
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config file: {e}")
        return MCPConfig(), LLMConfig()
    except Exception as e:
        print(f"Error loading config: {e}")
        return MCPConfig(), LLMConfig()


def get_api_key_from_env(provider: str) -> str | None:
    """Get API key from environment variables based on provider."""
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    elif provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY")
    else:
        return None


def auto_detect_provider_from_env() -> tuple[str | None, str | None]:
    """Auto-detect provider and API key from environment variables."""
    providers = [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ]

    for provider, env_var in providers:
        api_key = os.getenv(env_var)
        if api_key:
            return provider, api_key

    return None, None


def validate_llm_config(config: LLMConfig) -> None:
    """Validate LLM configuration."""
    try:
        LLMProvider(config.provider)
    except ValueError as err:
        raise ValueError(
            f"Invalid LLM provider: {config.provider}. Supported providers: {[p.value for p in LLMProvider]}"
        ) from err

    if config.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")


def save_llm_config(config: LLMConfig, config_path: str = "llm_config.json") -> None:
    """Save LLM configuration to JSON file."""
    config_dict = {
        "provider": config.provider,
        "model": config.model,
        "max_tokens": config.max_tokens,
        "options": config.options,
    }

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def save_mcp_config(config: MCPConfig, config_path: str = "mcp_config.json") -> None:
    """Save MCP configuration to JSON file."""
    config_dict = {"mcpServers": config.servers}

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
