#!/usr/bin/env python3
"""
Interactive MCP CLI
==================

An interactive command-line interface for testing MCP servers with LLM integration.
This is a separate application that uses the SimpleMCPClient for easy testing and debugging.

Usage:
    python interactive_cli.py

Features:
- Auto-detect LLM providers from environment
- Interactive chat with MCP tool execution
- Support for multiple LLM providers (Anthropic, OpenAI, OpenRouter)
- Verbose logging for debugging
- Fallback to legacy config format
"""

import asyncio
from typing import Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

from config import LLMConfig, auto_detect_provider_from_env, load_llm_config, validate_llm_config
from llm_providers import LLMProvider, create_llm_provider
from mcp_client import QueryProcessor
from simple_mcp_client import SimpleMCPClient


class InteractiveMCPCLI:
    """Interactive CLI for testing MCP servers with LLM integration."""

    def __init__(
        self,
        mcp_client: SimpleMCPClient,
        llm_config: LLMConfig,
        api_key: str,
        verbose: bool = False,
    ):
        self.mcp_client = mcp_client
        self.llm_config = llm_config
        self.api_key = api_key
        self.verbose = verbose

        # Create LLM provider
        provider_enum = LLMProvider(llm_config.provider)
        self.llm_provider = create_llm_provider(
            provider=provider_enum,
            api_key=api_key,
            model=llm_config.model,
            **llm_config.options,
        )

        # Create query processor
        self.query_processor: Optional[QueryProcessor] = None

    async def connect_to_server(self, server_name: str | None = None) -> bool:
        """Connect to MCP server and initialize query processor."""
        try:
            # Connect to server
            tools = await self.mcp_client.connect_to_server(server_name)

            print(f"Connected to MCP Server: {self.mcp_client.get_current_server()}")
            print(f"Available tools: {[tool['name'] for tool in tools]}")

            # Create query processor with the MCP client's internal client
            from mcp_client import MCPClient

            mcp_internal_client = MCPClient(self.mcp_client.config, verbose=self.verbose)
            mcp_internal_client.client = self.mcp_client.client
            mcp_internal_client.available_tools = self.mcp_client.available_tools
            mcp_internal_client.current_server = self.mcp_client.current_server

            self.query_processor = QueryProcessor(
                mcp_internal_client, self.llm_provider, max_tokens=self.llm_config.max_tokens
            )

            return True

        except Exception as e:
            print(f"Failed to connect to MCP server: {e}")
            return False

    async def run_interactive_session(self) -> None:
        """Run the interactive chat session."""
        print("\n" + "=" * 60)
        print(f"Interactive MCP CLI ({self.llm_config.provider})")
        print("=" * 60)
        print("Type 'quit', 'exit', or 'q' to exit")
        print("Type '/help' for available commands")
        print("Type '/tools' to list available MCP tools")
        print("=" * 60)

        while True:
            try:
                query = input("\n🤖 Your query: ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not query:
                    continue

                # Handle special commands
                if query.lower() == "/help":
                    self._show_help()
                    continue

                if query.lower() == "/tools":
                    self._show_tools()
                    continue

                if query.lower() == "/status":
                    self._show_status()
                    continue

                if query.lower().startswith("/switch "):
                    server_name = query[8:].strip()
                    if await self.connect_to_server(server_name):
                        continue
                    else:
                        print(f"❌ Failed to switch to MCP server: {server_name}")
                        continue

                # Process query with LLM and MCP tools
                print(f"\n🔄 Processing: {query}")

                if self.query_processor:
                    response = await self.query_processor.process_query(query)
                    print(f"\n✅ Response: {response}")
                else:
                    print("❌ No query processor available. Please connect to an MCP server first.")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                if self.verbose:
                    import traceback

                    traceback.print_exc()
                continue

    def _show_help(self) -> None:
        """Show help information."""
        print("\n📖 Available Commands:")
        print("  /help         - Show this help message")
        print("  /tools        - List available MCP tools")
        print("  /status       - Show connection status")
        print("  /switch <name> - Switch to a different MCP server")
        print("  quit/exit/q   - Exit the CLI")
        print("\n💡 Tips:")
        print("  - Ask questions and the LLM will use MCP tools to help")
        print("  - Use natural language to interact with your MCP tools")
        print("  - Tools are executed automatically when needed")

    def _show_tools(self) -> None:
        """Show available MCP tools."""
        if not self.mcp_client.is_connected():
            print("❌ Not connected to any MCP server")
            return

        tools = self.mcp_client.get_available_tools()
        if not tools:
            print("ℹ️  No tools available")
            return

        print(f"\n🔧 Available Tools ({len(tools)}):")
        for tool in tools:
            print(f"  • {tool['name']}: {tool['description']}")

    def _show_status(self) -> None:
        """Show connection and configuration status."""
        print("\n📊 Status:")
        print(f"  LLM Provider: {self.llm_config.provider}")
        print(f"  Model: {self.llm_config.model}")
        print(f"  Max Tokens: {self.llm_config.max_tokens}")
        print(f"  Connected: {self.mcp_client.is_connected()}")
        print(f"  Current MCP Server: {self.mcp_client.get_current_server()}")
        print(f"  Available MCP Servers: {self.mcp_client.list_servers()}")
        print(f"  Available Tools: {len(self.mcp_client.get_available_tools())}")


async def main() -> None:
    """Main entry point for the interactive CLI."""
    print("🚀 Simple MCP Interactive CLI")
    print("=" * 50)

    # Auto-detect provider from environment
    provider, api_key = auto_detect_provider_from_env()

    if not api_key:
        print("❌ Error: No API key found.")
        print("Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY")
        return

    print(f"🔑 Using {provider} provider")

    # Load configurations
    try:
        llm_config = load_llm_config()
        if provider:
            llm_config.provider = provider
        validate_llm_config(llm_config)
    except Exception as e:
        print(f"❌ Error loading LLM config: {e}")
        return

    # Create MCP client
    try:
        mcp_client = SimpleMCPClient()
        cli = InteractiveMCPCLI(mcp_client, llm_config, api_key, verbose=True)
    except Exception as e:
        print(f"❌ Error creating MCP client: {e}")
        return

    # List available servers
    servers = mcp_client.list_servers()
    if not servers:
        print("⚠️  No MCP servers configured in mcp_config.json")
        print("📝 Please configure at least one server to continue")
        return

    print(f"📡 Available MCP servers: {servers}")

    # Connect to first server
    if not await cli.connect_to_server():
        print("❌ Failed to connect to any MCP server")
        return

    # Run interactive session
    try:
        await cli.run_interactive_session()
    except Exception as e:
        print(f"❌ Error in interactive session: {e}")
    finally:
        await mcp_client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
