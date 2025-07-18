"""
Query processing utilities for MCP clients.

This module provides common utilities for processing user queries with LLM integration
and MCP tool execution.
"""

import json
import logging
from typing import Any

from llm_providers import AnthropicProvider, BaseLLMProvider, LLMMessage, LLMResponse


class QueryProcessingUtils:
    """Utility class containing common query processing methods."""

    @staticmethod
    def format_tool_input_for_display(tool_name: str, tool_input: dict) -> str:
        """Format tool input for better display in CLI."""
        if tool_name == "query" and "sql" in tool_input:
            # Special formatting for SQL queries
            sql = tool_input["sql"]
            # Replace literal \n with actual newlines for better readability
            formatted_sql = sql.replace("\\n", "\n")
            # Create a nice format for SQL
            return f'{{\n  "sql": """\n{formatted_sql}\n  """\n}}'
        else:
            # Standard JSON formatting for other tools
            return json.dumps(tool_input, indent=2)

    @staticmethod
    def format_tool_use_message(
        tool_id: str, tool_name: str, tool_input: dict, llm_provider: BaseLLMProvider
    ) -> Any:
        """Format tool use message for the specific provider."""
        if isinstance(llm_provider, AnthropicProvider):
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

    @staticmethod
    def format_tool_result_message(
        tool_id: str, result_text: str, llm_provider: BaseLLMProvider
    ) -> Any:
        """Format tool result message for the specific provider."""
        if isinstance(llm_provider, AnthropicProvider):
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

    @staticmethod
    def format_tool_use_messages(tool_calls: list) -> str:
        """Format multiple tool use messages for the LLM."""
        if not tool_calls:
            return ""

        formatted_calls = []
        for tool_call in tool_calls:
            tool_id = tool_call["id"]
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]

            formatted_call = f"""
                                <tool_use>
                                <tool_id>{tool_id}</tool_id>
                                <tool_name>{tool_name}</tool_name>
                                <parameters>
                                    {json.dumps(tool_input, indent=2)}
                                </parameters>
                                </tool_use>"""
            formatted_calls.append(formatted_call)

        if len(formatted_calls) == 1:
            return f"I'll use the {tool_calls[0]['name']} tool.\n\n{formatted_calls[0]}"
        else:
            tool_names = [call["name"] for call in tool_calls]
            return f"I'll use the following tools: {', '.join(tool_names)}.\n\n" + "\n\n".join(
                formatted_calls
            )

    @staticmethod
    def format_single_tool_use_message(tool_id: str, tool_name: str, tool_input: dict) -> str:
        """Format a single tool use message for the LLM."""
        return f"""I'll use the {tool_name} tool.
                <tool_use>
                <tool_id>{tool_id}</tool_id>
                <tool_name>{tool_name}</tool_name>
                <parameters>
                {json.dumps(tool_input, indent=2)}
                </parameters>
                </tool_use>"""

    @staticmethod
    def format_single_tool_result_message(tool_id: str, tool_result: str) -> str:
        """Format a single tool result message for the LLM."""
        return f"""<tool_result>
                <tool_id>{tool_id}</tool_id>
                <result>
                {tool_result}
                </result>
                </tool_result>"""


class MultiServerQueryProcessor:
    """Enhanced QueryProcessor that works with SimpleMCPClient for multi-server support."""

    def __init__(
        self, simple_mcp_client, llm_provider, max_tokens: int = 1000, verbose: bool = False
    ):
        from simple_mcp_client import SimpleMCPClient
        from llm_providers import BaseLLMProvider

        self.simple_mcp_client: SimpleMCPClient = simple_mcp_client
        self.llm_provider: BaseLLMProvider = llm_provider
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Set up logger
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def _format_tool_input_for_display(self, tool_name: str, tool_input: dict) -> str:
        """Format tool input for better display in CLI."""
        return QueryProcessingUtils.format_tool_input_for_display(tool_name, tool_input)

    def _format_tool_use_messages(self, tool_calls: list) -> str:
        """Format multiple tool use messages for the LLM."""
        return QueryProcessingUtils.format_tool_use_messages(tool_calls)

    async def process_query(self, user_query: str, max_iterations: int = 5) -> str:
        """Process a user query, potentially using MCP tools from multiple servers.

        Supports iterative tool calling where the LLM can make multiple rounds of tool calls
        based on previous results. This is essential for MCP servers like Context7 that
        require sequential tool execution (e.g., resolve-library-id → get-library-docs).

        Args:
            user_query: The user's query to process
            max_iterations: Maximum number of tool calling rounds to prevent infinite loops
        """
        if not self.simple_mcp_client.connected_servers:
            raise RuntimeError("Not connected to any MCP servers")

        # Prepare tools for the LLM provider
        provider_tools = []
        all_tools = self.simple_mcp_client.get_available_tools()
        for tool in all_tools:
            provider_tools.append(self.llm_provider.format_tool_for_provider(tool))

        # Add system context about MCP
        system_context = """You are an AI assistant with access to MCP (Model Context Protocol) tools.
                            MCP allows you to use external tools and functions to help users. When users mention "MCP" or "using MCP",
                            they're referring to these external tools you have access to. You should use the appropriate tools
                            to fulfill their requests and explain what you're doing.

                            Available MCP tools: {tools_list}

                            Always use the appropriate MCP tools when they can help fulfill the user's request.
                            For tools that require sequential execution (like resolve-library-id followed by get-library-docs),
                            call them in the correct order based on the results of previous tool calls.""".format(
            tools_list=", ".join([f"{tool['name']} ({tool['description']})" for tool in all_tools])
        )

        # Log the system context if verbose
        if self.verbose:
            self.logger.debug(f"System context: {system_context}")

        # Initial request to LLM with system context
        messages = [
            LLMMessage(role="user", content=f"{system_context}\n\nUser query: {user_query}")
        ]

        # Iterative tool calling loop
        iteration = 0
        final_text = []

        while iteration < max_iterations:
            iteration += 1

            if self.verbose:
                self.logger.debug(f"Tool calling iteration {iteration}/{max_iterations}")

            # Log the request if verbose
            if self.verbose:
                self.logger.debug(
                    f"LLM Request (iteration {iteration}):\nProvider: {self.llm_provider.__class__.__name__}\nModel: {self.llm_provider.model}\nMessages: {json.dumps([{'role': m.role, 'content': m.content} for m in messages], indent=2)}\nTools: {json.dumps(provider_tools, indent=2)}"
                )

            response = await self.llm_provider.create_message(
                messages=messages,
                tools=provider_tools if provider_tools else None,
                max_tokens=self.max_tokens,
            )

            # Log the response if verbose
            if self.verbose:
                self.logger.debug(f"LLM Response (iteration {iteration}): {response}")

            # Store the assistant's text response
            if response.text_content:
                final_text.extend(response.text_content)

            # Check if there are tool calls to execute
            if not response.tool_calls:
                # No more tool calls, we're done
                break

            # Process tool calls
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["input"]
                tool_id = tool_call["id"]

                print(f"\n🔧 Executing tool: {tool_name} (iteration {iteration})")
                print(
                    f"📥 Tool input: {self._format_tool_input_for_display(tool_name, tool_input)}"
                )

                # Execute tool via SimpleMCPClient (handles multi-server automatically)
                try:
                    tool_result_text = await self.simple_mcp_client.execute_tool(
                        tool_name, tool_input
                    )
                    print(f"📤 Tool output: {tool_result_text}")

                    tool_results.append(
                        {
                            "id": tool_id,
                            "name": tool_name,
                            "input": tool_input,
                            "result": tool_result_text,
                            "success": True,
                        }
                    )

                except Exception as e:
                    print(f"❌ Tool execution error: {e}")
                    tool_results.append(
                        {
                            "id": tool_id,
                            "name": tool_name,
                            "input": tool_input,
                            "result": f"Error: {e}",
                            "success": False,
                        }
                    )

            # Add assistant's response with tool calls to message history
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=self._format_tool_use_messages(response.tool_calls),
                )
            )

            # Add tool results to message history
            for tool_result in tool_results:
                messages.append(
                    LLMMessage(
                        role="user",
                        content=QueryProcessingUtils.format_single_tool_result_message(
                            tool_result["id"], tool_result["result"]
                        ),
                    )
                )

        # If we hit max iterations, add a note
        if iteration >= max_iterations:
            final_text.append(
                f"\n\nNote: Reached maximum iteration limit ({max_iterations}). Some tool calls may have been truncated."
            )

        return "\n".join(final_text) if final_text else "No response generated."
