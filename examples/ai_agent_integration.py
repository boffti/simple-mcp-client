#!/usr/bin/env python3
"""
AI Agent Integration Example
============================

This example shows how to integrate SimpleMCPClient with AI/LLM applications
to create intelligent agents that can use MCP tools.
"""

import asyncio
import json
from typing import Any

from config import get_api_key_from_env, load_llm_config
from llm_providers import LLMMessage, LLMProvider, create_llm_provider
from simple_mcp_client import SimpleMCPClient


class MCPAgent:
    """
    An intelligent agent that can use MCP tools to accomplish tasks.

    This agent combines an LLM with MCP tools to create a powerful
    assistant that can interact with external systems.
    """

    def __init__(self, llm_provider, mcp_client: SimpleMCPClient):
        self.llm_provider = llm_provider
        self.mcp_client = mcp_client
        self.conversation_history = []

    async def process_task(self, task: str) -> str:
        """Process a task using LLM reasoning and MCP tools"""

        # Get available tools
        tools = self.mcp_client.get_available_tools()

        # Create system prompt
        system_prompt = self._create_system_prompt(tools)

        # Create messages
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=task),
        ]

        # Add conversation history
        messages.extend(self.conversation_history)

        # Get LLM response
        response = await self.llm_provider.create_message(
            messages=messages, tools=self._format_tools_for_llm(tools), max_tokens=1000
        )

        # Process response and execute tools if needed
        result = await self._process_response(response, messages)

        # Update conversation history
        self.conversation_history.append(LLMMessage(role="user", content=task))
        self.conversation_history.append(LLMMessage(role="assistant", content=result))

        return result

    def _create_system_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Create system prompt with tool descriptions"""
        tool_descriptions = []
        for tool in tools:
            tool_descriptions.append(f"- {tool['name']}: {tool['description']}")

        return f"""You are an intelligent assistant with access to MCP tools.
You can use these tools to help users accomplish tasks:

{chr(10).join(tool_descriptions)}

When you need to use a tool, the system will execute it for you and provide the results.
Be helpful and use the appropriate tools to complete user requests."""

    def _format_tools_for_llm(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tools for LLM provider"""
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(self.llm_provider.format_tool_for_provider(tool))
        return formatted_tools

    async def _process_response(self, response, messages: list[LLMMessage]) -> str:
        """Process LLM response and execute tools if needed"""
        result_parts = response.text_content.copy()

        # Execute any tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["input"]

            print(f"🔧 Executing tool: {tool_name}")
            print(f"📥 Arguments: {json.dumps(tool_args, indent=2)}")

            try:
                # Execute tool
                tool_result = await self.mcp_client.execute_tool(tool_name, tool_args)
                print(f"📤 Result: {tool_result}")

                # Get follow-up response from LLM
                messages.append(LLMMessage(role="assistant", content=f"Used tool {tool_name}"))
                messages.append(LLMMessage(role="user", content=f"Tool result: {tool_result}"))

                follow_up = await self.llm_provider.create_message(
                    messages=messages, max_tokens=1000
                )

                result_parts.extend(follow_up.text_content)

            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {e}"
                print(f"❌ {error_msg}")
                result_parts.append(error_msg)

        return "\n".join(result_parts) if result_parts else "No response generated."

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


async def example_file_management_agent():
    """Example: File management agent"""
    print("=== File Management Agent ===")

    # Set up LLM provider
    llm_config = load_llm_config()
    api_key = get_api_key_from_env(llm_config.provider)

    if not api_key:
        print("❌ No API key found for LLM provider")
        return

    llm_provider = create_llm_provider(LLMProvider(llm_config.provider), api_key, llm_config.model)

    # Set up MCP client
    async with SimpleMCPClient() as mcp_client:
        # Connect to filesystem server
        try:
            await mcp_client.connect_to_server("test-server")
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return

        # Create agent
        agent = MCPAgent(llm_provider, mcp_client)

        # Example tasks
        tasks = [
            "List the files in the current directory",
            "Create a new file called 'test.txt' with some content",
            "Read the contents of the file we just created",
            "Delete the test file",
        ]

        for task in tasks:
            print(f"\n📋 Task: {task}")
            result = await agent.process_task(task)
            print(f"✅ Result: {result}")


async def example_multi_server_agent():
    """Example: Agent that works with multiple servers"""
    print("\n=== Multi-Server Agent ===")

    # Set up LLM provider
    llm_config = load_llm_config()
    api_key = get_api_key_from_env(llm_config.provider)

    if not api_key:
        print("❌ No API key found for LLM provider")
        return

    llm_provider = create_llm_provider(LLMProvider(llm_config.provider), api_key, llm_config.model)

    # Set up MCP client
    mcp_client = SimpleMCPClient()

    try:
        # Get available servers
        servers = mcp_client.list_servers()
        print(f"Available servers: {servers}")

        # Connect to first server
        if servers:
            await mcp_client.connect_to_server(servers[0])

            # Create agent
            agent = MCPAgent(llm_provider, mcp_client)

            # Example complex task
            task = "Help me understand what tools are available and demonstrate one of them"
            print(f"\n📋 Complex Task: {task}")
            result = await agent.process_task(task)
            print(f"✅ Result: {result}")

    finally:
        await mcp_client.cleanup()


async def example_conversational_agent():
    """Example: Conversational agent with memory"""
    print("\n=== Conversational Agent ===")

    # Set up LLM provider
    llm_config = load_llm_config()
    api_key = get_api_key_from_env(llm_config.provider)

    if not api_key:
        print("❌ No API key found for LLM provider")
        return

    llm_provider = create_llm_provider(LLMProvider(llm_config.provider), api_key, llm_config.model)

    # Set up MCP client
    async with SimpleMCPClient() as mcp_client:
        try:
            await mcp_client.connect_to_server("test-server")
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return

        # Create agent
        agent = MCPAgent(llm_provider, mcp_client)

        # Simulate conversation
        conversation = [
            "What can you help me with?",
            "Can you show me what tools you have access to?",
            "Use one of the tools to demonstrate what you can do",
            "Thank you for the demonstration",
        ]

        for message in conversation:
            print(f"\n👤 User: {message}")
            response = await agent.process_task(message)
            print(f"🤖 Agent: {response}")


class TaskPlanningAgent(MCPAgent):
    """
    Extended agent that can plan and execute multi-step tasks
    """

    async def execute_plan(self, goal: str) -> str:
        """Execute a multi-step plan to achieve a goal"""

        # First, create a plan
        plan_prompt = f"""
        Create a step-by-step plan to accomplish this goal: {goal}

        Consider the available tools and break down the task into logical steps.
        Return the plan as a numbered list.
        """

        plan_response = await self.process_task(plan_prompt)
        print(f"📋 Plan:\n{plan_response}")

        # Execute the plan
        execution_prompt = f"""
        Now execute the plan you created:
        {plan_response}

        Goal: {goal}
        Execute each step using the available tools.
        """

        result = await self.process_task(execution_prompt)
        return result


async def example_task_planning_agent():
    """Example: Task planning agent"""
    print("\n=== Task Planning Agent ===")

    # Set up LLM provider
    llm_config = load_llm_config()
    api_key = get_api_key_from_env(llm_config.provider)

    if not api_key:
        print("❌ No API key found for LLM provider")
        return

    llm_provider = create_llm_provider(LLMProvider(llm_config.provider), api_key, llm_config.model)

    # Set up MCP client
    async with SimpleMCPClient() as mcp_client:
        try:
            await mcp_client.connect_to_server("test-server")
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return

        # Create planning agent
        agent = TaskPlanningAgent(llm_provider, mcp_client)

        # Complex goal
        goal = "Create a report about the current directory contents"
        print(f"\n🎯 Goal: {goal}")

        result = await agent.execute_plan(goal)
        print(f"✅ Final Result: {result}")


async def main():
    """Run all examples"""
    print("🤖 AI Agent Integration Examples")
    print("=" * 50)

    examples = [
        example_file_management_agent,
        example_multi_server_agent,
        example_conversational_agent,
        example_task_planning_agent,
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"❌ Example failed: {e}")

        print("\n" + "-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
