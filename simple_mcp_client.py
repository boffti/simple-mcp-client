#!/usr/bin/env python3
"""
Simple MCP Client Example
=========================

This is a minimal implementation of an MCP client that demonstrates the core concepts
without the complexity of the full CLI application. Use this as a starting point
for integrating MCP into your existing applications.

Key Components:
1. Connect to MCP server via stdio
2. Discover available tools
3. Execute tools when needed
4. Handle responses

Dependencies:
- pip install mcp anthropic

Usage:
    python simple_mcp_client.py

Configuration:
    Create mcp_config.json with your MCP servers
"""

import asyncio
import json
import os
import logging
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip


def load_config(config_path: str = "mcp_config.json") -> Dict[str, Any]:
    """Load MCP configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            # Support both old and new format for backward compatibility
            if "servers" in config and "mcpServers" not in config:
                config["mcpServers"] = config["servers"]
            return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default configuration.")
        return {"mcpServers": {}}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config file: {e}")
        return {"mcpServers": {}}


class SimpleMCPClient:
    """A minimal MCP client implementation with config support."""

    def __init__(
        self, anthropic_api_key: str, config_path: str = "mcp_config.json", verbose: bool = False
    ):
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)
        self.config = load_config(config_path)
        self.session: Optional[ClientSession] = None
        self.available_tools: List[Dict[str, Any]] = []
        self.exit_stack = AsyncExitStack()
        self.current_server: Optional[str] = None
        self.verbose = verbose

        # Set up logging if verbose
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)

    def list_servers(self) -> List[str]:
        """List all configured servers."""
        return list(self.config["mcpServers"].keys())

    def get_server_info(self, server_name: str) -> Dict[str, Any]:
        """Get information about a specific server."""
        return self.config["mcpServers"].get(server_name, {})

    async def connect_to_server(
        self,
        server_name: Optional[str] = None,
        server_command: Optional[str] = None,
        server_args: Optional[List[str]] = None,
    ) -> None:
        """Connect to an MCP server via stdio (from config or direct params)."""

        if server_name:
            # Connect using config
            if server_name not in self.config["mcpServers"]:
                raise ValueError(f"Server '{server_name}' not found in config")

            server_config = self.config["mcpServers"][server_name]
            command = server_config["command"]
            args = server_config["args"]
            env = server_config.get("env")
            self.current_server = server_name

            print(f"Connecting to configured server: {server_name}")
            print(f"Description: {server_config.get('description', 'No description')}")

        elif server_command and server_args:
            # Connect using direct parameters
            command = server_command
            args = server_args
            env = None
            self.current_server = "direct"

            print(f"Connecting to MCP server: {command} {' '.join(args)}")

        else:
            # Try to auto-connect to first available server
            if not self.config["mcpServers"]:
                raise ValueError(
                    "No servers configured and no direct connection parameters provided"
                )

            first_server = list(self.config["mcpServers"].keys())[0]
            await self.connect_to_server(server_name=first_server)
            return

        # Create server parameters
        server_params = StdioServerParameters(command=command, args=args, env=env)

        # Connect to server
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport

        # Create session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        # Initialize session
        await self.session.initialize()

        # Discover available tools
        tools_response = await self.session.list_tools()
        self.available_tools = []

        for tool in tools_response.tools:
            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            self.available_tools.append(tool_def)

        print(f"Connected! Available tools: {[t['name'] for t in self.available_tools]}")

    async def process_query(self, user_query: str) -> str:
        """Process a user query, potentially using MCP tools."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        # Prepare tools for Anthropic API
        anthropic_tools = []
        for tool in self.available_tools:
            anthropic_tools.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                }
            )

        # Add system context about MCP
        system_context = """You are an AI assistant with access to MCP (Model Context Protocol) tools. 
MCP allows you to use external tools and functions to help users. When users mention "MCP" or "using MCP", 
they're referring to these external tools you have access to. You should use the appropriate tools 
to fulfill their requests and explain what you're doing.

Available MCP tools: {tools_list}

Always use the appropriate MCP tools when they can help fulfill the user's request.""".format(
            tools_list=", ".join(
                [f"{tool['name']} ({tool['description']})" for tool in self.available_tools]
            )
        )

        # Log the system context if verbose
        if self.verbose:
            self.logger.debug(f"System context: {system_context}")

        # Initial request to Claude with system context
        messages = [{"role": "user", "content": f"{system_context}\n\nUser query: {user_query}"}]

        # Log the request if verbose
        if self.verbose:
            self.logger.debug(
                f"LLM Request:\nModel: claude-3-5-sonnet-20241022\nMessages: {json.dumps(messages, indent=2)}\nTools: {json.dumps(anthropic_tools, indent=2)}"
            )

        response = self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=anthropic_tools if anthropic_tools else None,
        )

        # Log the response if verbose
        if self.verbose:
            self.logger.debug(f"LLM Response: {response}")

        # Handle response
        final_text = []

        for content in response.content:
            if content.type == "text":
                final_text.append(content.text)

            elif content.type == "tool_use":
                tool_name = content.name
                tool_input = content.input

                print(f"\n🔧 Executing tool: {tool_name}")
                print(f"📥 Tool input: {json.dumps(tool_input, indent=2)}")

                # Execute tool via MCP server
                try:
                    if self.verbose:
                        self.logger.debug(f"MCP Tool Call: {tool_name} with args: {tool_input}")

                    tool_result = await self.session.call_tool(tool_name, tool_input)

                    if self.verbose:
                        self.logger.debug(f"MCP Tool Result: {tool_result}")

                    # Extract tool output
                    tool_output = []
                    for result_content in tool_result.content:
                        if hasattr(result_content, "text"):
                            tool_output.append(result_content.text)
                        else:
                            tool_output.append(str(result_content))

                    tool_result_text = "\n".join(tool_output)
                    print(f"📤 Tool output: {tool_result_text}")

                    # Get final response from Claude with tool result
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "I'll use the available tools to help answer your question.",
                                },
                                {
                                    "type": "tool_use",
                                    "id": content.id,
                                    "name": tool_name,
                                    "input": tool_input,
                                },
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": tool_result_text,
                                }
                            ],
                        }
                    )

                    if self.verbose:
                        self.logger.debug(f"Final LLM Request: {json.dumps(messages, indent=2)}")

                    final_response = self.anthropic_client.messages.create(
                        model="claude-3-5-sonnet-20241022", max_tokens=1000, messages=messages
                    )

                    if self.verbose:
                        self.logger.debug(f"Final LLM Response: {final_response}")

                    for final_content in final_response.content:
                        if final_content.type == "text":
                            final_text.append(final_content.text)

                except Exception as e:
                    print(f"❌ Tool execution error: {e}")
                    final_text.append(f"Error executing tool {tool_name}: {e}")

        return "\n".join(final_text) if final_text else "No response generated."

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.exit_stack.aclose()


async def main():
    """Example usage of the simple MCP client with config support."""

    # Configuration - load from environment
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    # Create client with config file (verbose=True for detailed logging)
    client = SimpleMCPClient(ANTHROPIC_API_KEY, verbose=True)

    try:
        # List available servers
        servers = client.list_servers()
        if servers:
            print(f"Available servers: {servers}")

            # Connect to first server (or specify one)
            server_name = servers[0]  # or choose specific server
            await client.connect_to_server(server_name=server_name)
        else:
            print("No servers configured. Using direct connection example.")
            # Fallback to direct connection
            await client.connect_to_server(
                server_command="python", server_args=["path/to/your/mcp_server.py"]
            )

        # Interactive mode
        print("\n" + "=" * 50)
        print("Interactive MCP Client - Type 'quit' to exit")
        print("=" * 50)

        while True:
            try:
                query = input("\nYour query: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if not query:
                    continue

                print(f"\n--- Processing: {query} ---")
                response = await client.process_query(query)
                print(f"Response: {response}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
                continue

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await client.cleanup()


if __name__ == "__main__":
    # Simple usage example
    print("Simple MCP Client Example")
    print("=" * 50)
    print("This demonstrates the core concepts of MCP integration.")
    print("1. Connect to MCP server")
    print("2. Discover tools")
    print("3. Process queries with tool execution")
    print("4. Handle responses")
    print()

    asyncio.run(main())

    print("To use this client:")
    print("1. Install dependencies: uv sync")
    print("2. Set your ANTHROPIC_API_KEY")
    print("3. Create mcp_config.json with your servers")
    print("5. Run: python simple_mcp_client.py")
    print()
