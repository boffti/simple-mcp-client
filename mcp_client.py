"""
MCP Client Module
================

This module contains the MCP client implementation for connecting to MCP servers
and managing tool execution.
"""

import json
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import MCPConfig
from llm_providers import BaseLLMProvider, LLMMessage, AnthropicProvider


class MCPClient:
    """MCP client for connecting to servers and executing tools."""

    def __init__(self, config: MCPConfig, verbose: bool = False):
        self.config = config
        self.session: Optional[ClientSession] = None
        self.available_tools: List[Dict[str, Any]] = []
        self.exit_stack = AsyncExitStack()
        self.current_server: Optional[str] = None
        self.verbose = verbose

        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)

    def list_servers(self) -> List[str]:
        """List all configured servers."""
        return list(self.config.servers.keys())

    def get_server_info(self, server_name: str) -> Dict[str, Any]:
        """Get information about a specific server."""
        return self.config.servers.get(server_name, {})

    async def connect_to_server(
        self,
        server_name: Optional[str] = None,
        server_command: Optional[str] = None,
        server_args: Optional[List[str]] = None,
    ) -> None:
        """Connect to an MCP server via stdio (from config or direct params)."""

        if server_name:
            # Connect using config
            if server_name not in self.config.servers:
                raise ValueError(f"Server '{server_name}' not found in config")

            server_config = self.config.servers[server_name]
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
            if not self.config.servers:
                raise ValueError(
                    "No servers configured and no direct connection parameters provided"
                )

            first_server = list(self.config.servers.keys())[0]
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

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

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

        return "\n".join(tool_output)

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.exit_stack.aclose()


class QueryProcessor:
    """Processes user queries using LLM and MCP tools."""

    def __init__(
        self, mcp_client: MCPClient, llm_provider: BaseLLMProvider, max_tokens: int = 1000
    ):
        self.mcp_client = mcp_client
        self.llm_provider = llm_provider
        self.max_tokens = max_tokens

    async def process_query(self, user_query: str) -> str:
        """Process a user query, potentially using MCP tools."""
        if not self.mcp_client.session:
            raise RuntimeError("Not connected to MCP server")

        # Prepare tools for the LLM provider
        provider_tools = []
        for tool in self.mcp_client.available_tools:
            provider_tools.append(self.llm_provider.format_tool_for_provider(tool))

        # Add system context about MCP
        system_context = """You are an AI assistant with access to MCP (Model Context Protocol) tools. 
MCP allows you to use external tools and functions to help users. When users mention "MCP" or "using MCP", 
they're referring to these external tools you have access to. You should use the appropriate tools 
to fulfill their requests and explain what you're doing.

Available MCP tools: {tools_list}

Always use the appropriate MCP tools when they can help fulfill the user's request.""".format(
            tools_list=", ".join(
                [
                    f"{tool['name']} ({tool['description']})"
                    for tool in self.mcp_client.available_tools
                ]
            )
        )

        # Log the system context if verbose
        if self.mcp_client.verbose:
            self.mcp_client.logger.debug(f"System context: {system_context}")

        # Initial request to LLM with system context
        messages = [
            LLMMessage(role="user", content=f"{system_context}\n\nUser query: {user_query}")
        ]

        # Log the request if verbose
        if self.mcp_client.verbose:
            self.mcp_client.logger.debug(
                f"LLM Request:\nProvider: {self.llm_provider.__class__.__name__}\nModel: {self.llm_provider.model}\nMessages: {json.dumps([{'role': m.role, 'content': m.content} for m in messages], indent=2)}\nTools: {json.dumps(provider_tools, indent=2)}"
            )

        response = await self.llm_provider.create_message(
            messages=messages,
            tools=provider_tools if provider_tools else None,
            max_tokens=self.max_tokens,
        )

        # Log the response if verbose
        if self.mcp_client.verbose:
            self.mcp_client.logger.debug(f"LLM Response: {response}")

        # Handle response
        final_text = response.text_content.copy()

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            tool_id = tool_call["id"]

            print(f"\n🔧 Executing tool: {tool_name}")
            print(f"📥 Tool input: {json.dumps(tool_input, indent=2)}")

            # Execute tool via MCP server
            try:
                tool_result_text = await self.mcp_client.execute_tool(tool_name, tool_input)
                print(f"📤 Tool output: {tool_result_text}")

                # Get final response from LLM with tool result
                messages.append(
                    LLMMessage(
                        role="assistant",
                        content=self._format_tool_use_message(tool_id, tool_name, tool_input),
                    )
                )
                messages.append(
                    LLMMessage(
                        role="user",
                        content=self._format_tool_result_message(tool_id, tool_result_text),
                    )
                )

                if self.mcp_client.verbose:
                    self.mcp_client.logger.debug(
                        f"Final LLM Request: {json.dumps([{'role': m.role, 'content': m.content} for m in messages], indent=2)}"
                    )

                final_response = await self.llm_provider.create_message(
                    messages=messages, max_tokens=self.max_tokens
                )

                if self.mcp_client.verbose:
                    self.mcp_client.logger.debug(f"Final LLM Response: {final_response}")

                final_text.extend(final_response.text_content)

            except Exception as e:
                print(f"❌ Tool execution error: {e}")
                final_text.append(f"Error executing tool {tool_name}: {e}")

        return "\n".join(final_text) if final_text else "No response generated."

    def _format_tool_use_message(
        self, tool_id: str, tool_name: str, tool_input: Dict[str, Any]
    ) -> Any:
        """Format tool use message for the specific provider."""
        if isinstance(self.llm_provider, AnthropicProvider):
            return [
                {
                    "type": "text",
                    "text": "I'll use the available tools to help answer your question.",
                },
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                },
            ]
        else:
            # For OpenAI-compatible providers
            return f"I'll use the {tool_name} tool to help answer your question."

    def _format_tool_result_message(self, tool_id: str, result_text: str) -> Any:
        """Format tool result message for the specific provider."""
        if isinstance(self.llm_provider, AnthropicProvider):
            return [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_text,
                }
            ]
        else:
            # For OpenAI-compatible providers
            return f"Tool result: {result_text}"
