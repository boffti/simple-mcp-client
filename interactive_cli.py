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

from simple_mcp_client import SimpleMCPClient


class InteractiveMCPCLI:
    """Interactive CLI for testing MCP servers with LLM integration."""

    def __init__(
        self,
        mcp_client: SimpleMCPClient,
        llm_config: LLMConfig,
        api_key: str,
        verbose: bool = False,
        max_iterations: int = 5,
    ):
        self.mcp_client = mcp_client
        self.llm_config = llm_config
        self.api_key = api_key
        self.verbose = verbose
        self.max_iterations = max_iterations

        # Create LLM provider
        provider_enum = LLMProvider(llm_config.provider)
        self.llm_provider = create_llm_provider(
            provider=provider_enum,
            api_key=api_key,
            model=llm_config.model,
            **llm_config.options,
        )

        # Create query processor
        self.query_processor: Optional[MultiServerQueryProcessor] = None

    async def connect_to_server(self, server_name: str | None = None) -> bool:
        """Connect to MCP server(s) and initialize query processor."""
        try:
            if server_name:
                # Connect to specific server
                tools = await self.mcp_client.connect_to_server(server_name)
                print(f"Connected to MCP Server: {server_name}")
                print(f"Available tools: {[tool['name'] for tool in tools]}")
            else:
                # Connect to all servers
                server_tools = await self.mcp_client.connect_to_all_servers()

                print("Connected to MCP Servers:")
                for server, tools in server_tools.items():
                    if tools:  # Only show servers with tools
                        tool_names = [tool["name"] for tool in tools]
                        print(f"  • {server}: {tool_names}")
                    else:
                        print(f"  • {server}: (connection failed)")

                all_tools = self.mcp_client.get_available_tools()
                print(f"\nTotal available tools: {len(all_tools)}")

            # Create query processor that works with multi-server SimpleMCPClient
            from query_processing import MultiServerQueryProcessor

            self.query_processor = MultiServerQueryProcessor(
                self.mcp_client,
                self.llm_provider,
                max_tokens=self.llm_config.max_tokens,
                verbose=self.verbose,
            )

            return True

        except Exception as e:
            print(f"Failed to connect to MCP server(s): {e}")
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

                if query.lower() == "/servers":
                    servers = self.mcp_client.list_servers()
                    print("📡 Available servers from config:")
                    for server in servers:
                        status = (
                            "✅ Connected"
                            if server in self.mcp_client.connected_servers
                            else "❌ Not connected"
                        )
                        print(f"  • {server}: {status}")
                    continue

                if query.lower().startswith("/connect "):
                    server_name = query[9:].strip()
                    if server_name in self.mcp_client.list_servers():
                        if await self.connect_to_server(server_name):
                            print(f"✅ Successfully connected to {server_name}")
                        else:
                            print(f"❌ Failed to connect to server: {server_name}")
                    else:
                        print(f"❌ Server '{server_name}' not found in config")
                        available = self.mcp_client.list_servers()
                        print(f"Available servers: {available}")
                    continue

                if query.lower() == "/reconnect":
                    print("🔄 Reconnecting to all servers...")
                    if await self.connect_to_server():
                        print("✅ Successfully reconnected to all servers")
                    else:
                        print("❌ Failed to reconnect to servers")
                    continue

                # Process query with LLM and MCP tools
                print(f"\n🔄 Processing: {query}")

                if self.query_processor:
                    response = await self.query_processor.process_query(
                        query, max_iterations=self.max_iterations
                    )
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
        """Show available commands."""
        print("📚 Available Commands:")
        print("-" * 30)
        print("/help       - Show this help message")
        print("/tools      - List all available MCP tools")
        print("/status     - Show connection and client status")
        print("/servers    - List available servers from config")
        print("/connect <server> - Connect to a specific server")
        print("/reconnect  - Reconnect to all servers")
        print("quit/exit/q - Exit the CLI")
        print("")
        print("💬 Or type any question to interact with the MCP tools via LLM")
        print(
            f"🔄 Supports iterative tool calling (max {self.max_iterations} rounds for sequential operations)"
        )

    def _show_tools(self) -> None:
        """Show available tools grouped by server."""
        if not self.mcp_client.connected_servers:
            print("❌ Not connected to any MCP servers")
            return

        print("🔧 Available MCP Tools:")
        print("-" * 50)

        for server_name in self.mcp_client.get_connected_servers():
            tools = self.mcp_client.get_tools_by_server(server_name)
            if tools:
                print(f"\n📡 Server: {server_name}")
                for tool in tools:
                    description = tool.get("description", "No description")
                    print(f"  • {tool['name']}: {description}")
            else:
                print(f"\n📡 Server: {server_name} (no tools)")

        total_tools = len(self.mcp_client.get_available_tools())
        print(f"\nTotal tools available: {total_tools}")

    def _show_status(self) -> None:
        """Show current connection status and server information."""
        print("📊 MCP Client Status:")
        print("-" * 30)

        connected_servers = self.mcp_client.get_connected_servers()
        if connected_servers:
            print(f"Connected servers: {len(connected_servers)}")
            for server in connected_servers:
                tool_count = len(self.mcp_client.get_tools_by_server(server))
                print(f"  • {server}: {tool_count} tools")

            total_tools = len(self.mcp_client.get_available_tools())
            print(f"Total tools: {total_tools}")
        else:
            print("❌ Not connected to any servers")

        print(f"LLM Provider: {self.llm_config.provider}")
        print(f"Query Processor: {'✅ Ready' if self.query_processor else '❌ Not initialized'}")


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
        cli = InteractiveMCPCLI(mcp_client, llm_config, api_key, verbose=True, max_iterations=5)
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
