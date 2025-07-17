# Simple MCP Client

A clean, minimal MCP (Model Context Protocol) client for easy integration into any application. This library provides the core functionality for connecting to MCP servers and executing tools, while keeping the Interactive CLI separate for testing purposes.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

### Basic Usage

```python
from simple_mcp_client import SimpleMCPClient

async def example():
    # Create client
    async with SimpleMCPClient() as client:
        # Connect to server
        tools = await client.connect_to_server("my-server")
        
        # Execute a tool
        result = await client.execute_tool("my_tool", {"arg": "value"})
        print(result)
```

## 📁 Project Structure

```
simple-mcp-client/
├── simple_mcp_client.py    # 🎯 Main MCP client (for developers)
├── interactive_cli.py      # 🖥️ Interactive CLI (for testing)
├── llm_providers.py        # 🧠 LLM provider implementations
├── config.py               # ⚙️ Configuration management
├── mcp_client.py           # 🔧 Internal MCP client components
├── mcp_config.json         # 📡 MCP server configuration
├── llm_config.json         # 🤖 LLM provider configuration
└── examples/               # 📚 Integration examples
    ├── basic_usage.py
    ├── web_app_integration.py
    └── ai_agent_integration.py
```

## 🎯 For Developers: Using SimpleMCPClient

The `SimpleMCPClient` is designed to be the **main interface** for developers who want to integrate MCP functionality into their applications.

### Key Features

- **Clean API**: Simple, intuitive methods for MCP operations
- **Async Support**: Full async/await support with context managers
- **Error Handling**: Comprehensive error handling with clear exceptions
- **Tool Validation**: Automatic tool validation and helpful error messages
- **Connection Management**: Automatic connection lifecycle management
- **No Dependencies**: Zero dependencies on CLI or LLM components

### API Reference

#### `SimpleMCPClient`

```python
class SimpleMCPClient:
    def __init__(self, config_path: str = "mcp_config.json", config: Optional[MCPConfig] = None)
    
    # Server Management
    def list_servers(self) -> List[str]
    def get_server_info(self, server_name: str) -> Dict[str, Any]
    
    # Connection Management
    async def connect_to_server(self, server_name: str = None, ...) -> List[Dict[str, Any]]
    def is_connected(self) -> bool
    def get_current_server(self) -> Optional[str]
    
    # Tool Operations
    def get_available_tools(self) -> List[Dict[str, Any]]
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str
    
    # Cleanup
    async def cleanup(self) -> None
    
    # Context Manager Support
    async def __aenter__(self) -> SimpleMCPClient
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None
```

#### Convenience Functions

```python
# Execute a single tool
async def execute_mcp_tool(server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str

# List available tools
async def list_mcp_tools(server_name: str) -> List[Dict[str, Any]]
```

### Usage Examples

#### 1. Basic Usage

```python
from simple_mcp_client import SimpleMCPClient

async def basic_example():
    client = SimpleMCPClient()
    
    try:
        # Connect to server
        tools = await client.connect_to_server("filesystem")
        print(f"Available tools: {[tool['name'] for tool in tools]}")
        
        # Execute tool
        result = await client.execute_tool("read_file", {"path": "example.txt"})
        print(f"File contents: {result}")
        
    finally:
        await client.cleanup()
```

#### 2. Context Manager (Recommended)

```python
async def context_manager_example():
    async with SimpleMCPClient() as client:
        tools = await client.connect_to_server("my-server")
        result = await client.execute_tool("my_tool", {"param": "value"})
        return result
```

#### 3. Direct Connection (No Config File)

```python
async def direct_connection_example():
    async with SimpleMCPClient() as client:
        tools = await client.connect_to_server(
            server_command="python",
            server_args=["-m", "my_mcp_server"],
            server_env={"CUSTOM_VAR": "value"}
        )
        # Use tools...
```

#### 4. Error Handling

```python
async def error_handling_example():
    async with SimpleMCPClient() as client:
        try:
            await client.connect_to_server("non-existent-server")
        except ValueError as e:
            print(f"Configuration error: {e}")
        except RuntimeError as e:
            print(f"Connection error: {e}")
        
        try:
            result = await client.execute_tool("invalid_tool", {})
        except ValueError as e:
            print(f"Tool not found: {e}")
        except RuntimeError as e:
            print(f"Execution error: {e}")
```

#### 5. Server Management

```python
async def server_management_example():
    client = SimpleMCPClient()
    
    # List available servers
    servers = client.list_servers()
    print(f"Available servers: {servers}")
    
    # Get server information
    for server in servers:
        info = client.get_server_info(server)
        print(f"{server}: {info}")
    
    # Connect to each server
    for server in servers:
        try:
            tools = await client.connect_to_server(server)
            print(f"{server} has {len(tools)} tools")
        except Exception as e:
            print(f"Failed to connect to {server}: {e}")
    
    await client.cleanup()
```

## 🖥️ For Testing: Interactive CLI

The `interactive_cli.py` provides a **separate testing interface** that combines the MCP client with LLM providers for interactive testing.

### Running the CLI

```bash
# Set your API key
export ANTHROPIC_API_KEY=your_key_here
# or
export OPENAI_API_KEY=your_key_here

# Run the interactive CLI
python interactive_cli.py
```

### CLI Features

- **Auto-detection**: Automatically detects LLM providers from environment
- **Multi-provider**: Supports Anthropic, OpenAI, and OpenRouter
- **Interactive Chat**: Natural language interaction with MCP tools
- **Tool Execution**: Automatic tool execution based on LLM decisions
- **Server Switching**: Switch between different MCP servers
- **Debugging**: Verbose logging for development

### CLI Commands

- `help` - Show available commands
- `tools` - List available MCP tools
- `status` - Show connection and configuration status
- `switch <server>` - Switch to a different MCP server
- `quit/exit/q` - Exit the CLI

## ⚙️ Configuration

### MCP Server Configuration (`mcp_config.json`)

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["-m", "mcp_server_filesystem"],
      "description": "File system operations",
      "env": {
        "CUSTOM_VAR": "value"
      }
    },
    "web": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-web"],
      "description": "Web scraping and requests"
    }
  }
}
```

### LLM Configuration (`llm_config.json`) - CLI Only

```json
{
  "provider": "anthropic",
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1000,
  "options": {}
}
```

## 📚 Integration Examples

### Web Application Integration

```python
from simple_mcp_client import SimpleMCPClient
from fastapi import FastAPI

app = FastAPI()
mcp_client = SimpleMCPClient()

@app.post("/execute-tool")
async def execute_tool(server: str, tool: str, args: dict):
    async with SimpleMCPClient() as client:
        await client.connect_to_server(server)
        result = await client.execute_tool(tool, args)
        return {"result": result}
```

### AI Agent Integration

```python
from simple_mcp_client import SimpleMCPClient
from llm_providers import create_llm_provider, LLMProvider

class MCPAgent:
    def __init__(self, llm_provider, mcp_client):
        self.llm_provider = llm_provider
        self.mcp_client = mcp_client
    
    async def process_task(self, task: str) -> str:
        # Use LLM to determine which tools to use
        # Execute tools via mcp_client
        # Return processed result
        pass
```

### Batch Processing

```python
async def batch_process_files(file_paths: List[str]):
    async with SimpleMCPClient() as client:
        await client.connect_to_server("filesystem")
        
        results = []
        for path in file_paths:
            result = await client.execute_tool("read_file", {"path": path})
            results.append(result)
        
        return results
```

## 🔧 Development

### Project Setup

```bash
# Clone repository
git clone <repository-url>
cd simple-mcp-client

# Install dependencies
uv sync

# Install with development dependencies
uv sync --dev
```

### Code Quality

```bash
# Format code
black *.py

# Type checking
mypy simple_mcp_client.py

# Run tests
pytest
```

### Testing

```bash
# Test basic functionality
python -c "from simple_mcp_client import SimpleMCPClient; print('Import successful')"

# Run examples
python examples/basic_usage.py

# Test interactive CLI
python interactive_cli.py
```

## 🌟 Key Benefits

### For Developers

1. **Clean API**: Simple, intuitive interface focused on MCP functionality
2. **Zero Dependencies**: No CLI or LLM dependencies in the core client
3. **Async Support**: Full async/await support with proper resource management
4. **Error Handling**: Clear error messages and proper exception handling
5. **Flexible Connection**: Support for both config-based and direct connections
6. **Context Managers**: Automatic cleanup with async context managers

### For Testing

1. **Separate CLI**: Interactive testing interface separate from core client
2. **LLM Integration**: Natural language interaction with MCP tools
3. **Multi-provider**: Support for different LLM providers
4. **Debugging**: Verbose logging for development and troubleshooting

## 📦 Dependencies

### Core Client (`simple_mcp_client.py`)

- `mcp>=1.11.0` - MCP protocol implementation
- `python-dotenv>=1.0.0` - Environment variable loading

### Interactive CLI (`interactive_cli.py`)

- `anthropic>=0.8.0` - Anthropic Claude API
- `openai>=1.0.0` - OpenAI API (optional)

### Development

- `pytest>=7.0.0` - Testing framework
- `black>=23.0.0` - Code formatting
- `mypy>=1.0.0` - Type checking

## 🚀 Getting Started

1. **Install dependencies**: `uv sync`
2. **Configure MCP servers**: Edit `mcp_config.json`
3. **For testing**: Set API key and run `python interactive_cli.py`
4. **For development**: Import `SimpleMCPClient` and start building!

## 📝 Examples

Check out the `examples/` directory for comprehensive integration examples:

- `basic_usage.py` - Basic MCP client usage patterns
- `web_app_integration.py` - Web application integration with FastAPI
- `ai_agent_integration.py` - AI agent development with LLM integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details