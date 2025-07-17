#!/usr/bin/env python3
"""
Basic MCP Client Usage Examples
===============================

This file demonstrates how to use the SimpleMCPClient in your own applications.
"""

import asyncio
from simple_mcp_client import SimpleMCPClient, execute_mcp_tool, list_mcp_tools


async def example_1_basic_usage():
    """Example 1: Basic MCP client usage"""
    print("=== Example 1: Basic Usage ===")

    # Create client
    client = SimpleMCPClient()

    try:
        # Connect to server
        tools = await client.connect_to_server("test-server")
        print(f"Connected! Available tools: {[tool['name'] for tool in tools]}")

        # Execute a tool (example - adjust based on your server)
        if tools:
            first_tool = tools[0]
            print(f"Executing tool: {first_tool['name']}")
            # Note: Adjust arguments based on your tool's schema
            result = await client.execute_tool(first_tool["name"], {})
            print(f"Result: {result}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.cleanup()


async def example_2_context_manager():
    """Example 2: Using async context manager"""
    print("\n=== Example 2: Context Manager ===")

    async with SimpleMCPClient() as client:
        # Connect to server
        tools = await client.connect_to_server("test-server")
        print(f"Available tools: {len(tools)}")

        # Check connection status
        print(f"Connected: {client.is_connected()}")
        print(f"Current server: {client.get_current_server()}")


async def example_3_direct_connection():
    """Example 3: Direct server connection (no config file)"""
    print("\n=== Example 3: Direct Connection ===")

    async with SimpleMCPClient() as client:
        # Connect directly to a server
        tools = await client.connect_to_server(
            server_command="python",
            server_args=["-m", "some_mcp_server"],
            server_env={"CUSTOM_VAR": "value"},
        )
        print(f"Connected directly! Tools: {len(tools)}")


async def example_4_convenience_functions():
    """Example 4: Using convenience functions"""
    print("\n=== Example 4: Convenience Functions ===")

    try:
        # List tools from a server
        tools = await list_mcp_tools("test-server")
        print(f"Tools from server: {[tool['name'] for tool in tools]}")

        # Execute a single tool
        if tools:
            result = await execute_mcp_tool(
                server_name="test-server", tool_name=tools[0]["name"], arguments={}
            )
            print(f"Tool result: {result}")

    except Exception as e:
        print(f"Error: {e}")


async def example_5_error_handling():
    """Example 5: Proper error handling"""
    print("\n=== Example 5: Error Handling ===")

    client = SimpleMCPClient()

    try:
        # Try to connect to non-existent server
        await client.connect_to_server("non-existent-server")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except RuntimeError as e:
        print(f"Connection error: {e}")

    try:
        # Try to execute tool without connection
        await client.execute_tool("some_tool", {})
    except RuntimeError as e:
        print(f"Execution error: {e}")

    finally:
        await client.cleanup()


async def example_6_server_management():
    """Example 6: Server management and info"""
    print("\n=== Example 6: Server Management ===")

    client = SimpleMCPClient()

    # List all configured servers
    servers = client.list_servers()
    print(f"Available servers: {servers}")

    # Get server info
    for server in servers:
        info = client.get_server_info(server)
        print(f"Server {server}: {info}")

    # Connect to each server and get tools
    for server in servers:
        try:
            tools = await client.connect_to_server(server)
            print(f"Server {server} has {len(tools)} tools")

            # Get current status
            print(f"  Current server: {client.get_current_server()}")
            print(f"  Is connected: {client.is_connected()}")

        except Exception as e:
            print(f"Failed to connect to {server}: {e}")

    await client.cleanup()


async def main():
    """Run all examples"""
    print("🚀 SimpleMCPClient Examples")
    print("=" * 50)

    examples = [
        example_1_basic_usage,
        example_2_context_manager,
        example_3_direct_connection,
        example_4_convenience_functions,
        example_5_error_handling,
        example_6_server_management,
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Example failed: {e}")

        print()  # Add spacing between examples


if __name__ == "__main__":
    asyncio.run(main())
