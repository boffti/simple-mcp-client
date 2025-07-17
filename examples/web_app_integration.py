#!/usr/bin/env python3
"""
Web Application Integration Example
==================================

This example shows how to integrate SimpleMCPClient into a web application.
Uses FastAPI as an example, but the pattern works with any web framework.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

# FastAPI example (install with: pip install fastapi uvicorn)
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn")

from simple_mcp_client import SimpleMCPClient


class ToolExecutionRequest(BaseModel):
    """Request model for tool execution"""

    server_name: str
    tool_name: str
    arguments: dict[str, Any]


class ToolExecutionResponse(BaseModel):
    """Response model for tool execution"""

    success: bool
    result: str = ""
    error: str = ""


class MCPWebService:
    """Web service wrapper for MCP client"""

    def __init__(self):
        self.clients: dict[str, SimpleMCPClient] = {}

    async def get_or_create_client(self, server_name: str) -> SimpleMCPClient:
        """Get existing client or create new one for server"""
        if server_name not in self.clients:
            client = SimpleMCPClient()
            await client.connect_to_server(server_name)
            self.clients[server_name] = client
        return self.clients[server_name]

    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """Execute a tool on the specified server"""
        try:
            client = await self.get_or_create_client(request.server_name)
            result = await client.execute_tool(request.tool_name, request.arguments)
            return ToolExecutionResponse(success=True, result=result)
        except Exception as e:
            return ToolExecutionResponse(success=False, error=str(e))

    async def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        """List available tools for a server"""
        client = await self.get_or_create_client(server_name)
        return client.get_available_tools()

    async def list_servers(self) -> list[str]:
        """List all available servers"""
        # Create a temporary client to get server list
        temp_client = SimpleMCPClient()
        servers = temp_client.list_servers()
        return servers

    async def cleanup(self):
        """Clean up all clients"""
        for client in self.clients.values():
            await client.cleanup()
        self.clients.clear()


# Global service instance
mcp_service = MCPWebService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    print("🚀 Starting MCP Web Service")
    yield
    # Shutdown
    print("🛑 Shutting down MCP Web Service")
    await mcp_service.cleanup()


if FASTAPI_AVAILABLE:
    # Create FastAPI app
    app = FastAPI(
        title="MCP Web Service",
        description="Web API for MCP tool execution",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {"message": "MCP Web Service", "status": "running"}

    @app.get("/servers")
    async def get_servers():
        """Get list of available MCP servers"""
        try:
            servers = await mcp_service.list_servers()
            return {"servers": servers}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/servers/{server_name}/tools")
    async def get_tools(server_name: str):
        """Get available tools for a server"""
        try:
            tools = await mcp_service.list_tools(server_name)
            return {"server": server_name, "tools": tools}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/execute")
    async def execute_tool(request: ToolExecutionRequest):
        """Execute a tool on an MCP server"""
        response = await mcp_service.execute_tool(request)
        if not response.success:
            raise HTTPException(status_code=400, detail=response.error)
        return response

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "clients": len(mcp_service.clients)}


async def example_direct_usage():
    """Example of using MCPWebService directly (without FastAPI)"""
    print("=== Direct Web Service Usage ===")

    service = MCPWebService()

    try:
        # List servers
        servers = await service.list_servers()
        print(f"Available servers: {servers}")

        if servers:
            server_name = servers[0]

            # List tools
            tools = await service.list_tools(server_name)
            print(f"Tools for {server_name}: {[tool['name'] for tool in tools]}")

            # Execute a tool
            if tools:
                request = ToolExecutionRequest(
                    server_name=server_name, tool_name=tools[0]["name"], arguments={}
                )
                response = await service.execute_tool(request)
                print(f"Execution result: {response}")

    finally:
        await service.cleanup()


async def example_batch_operations():
    """Example of batch tool operations"""
    print("\n=== Batch Operations ===")

    service = MCPWebService()

    try:
        servers = await service.list_servers()

        # Execute multiple tools in parallel
        tasks = []
        for server in servers[:2]:  # Limit to first 2 servers
            tools = await service.list_tools(server)
            if tools:
                request = ToolExecutionRequest(
                    server_name=server, tool_name=tools[0]["name"], arguments={}
                )
                tasks.append(service.execute_tool(request))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                print(f"Task {i + 1}: {result}")

    finally:
        await service.cleanup()


def run_web_server():
    """Run the FastAPI web server"""
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        return

    import uvicorn

    print("🌐 Starting MCP Web Server on http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)


async def main():
    """Run examples"""
    print("🌐 MCP Web Integration Examples")
    print("=" * 50)

    await example_direct_usage()
    await example_batch_operations()

    print("\n" + "=" * 50)
    print("To run the web server:")
    print("python examples/web_app_integration.py --server")
    print("or")
    print("uvicorn examples.web_app_integration:app --reload")


if __name__ == "__main__":
    import sys

    if "--server" in sys.argv:
        run_web_server()
    else:
        asyncio.run(main())
