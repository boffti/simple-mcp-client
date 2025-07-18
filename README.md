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
├── simple_mcp_client.py      # 🎯 Main MCP client (for developers)
├── interactive_cli.py        # 🖥️ Interactive CLI (for testing)
├── llm_providers.py          # 🧠 LLM provider implementations
├── config.py                 # ⚙️ Configuration management
├── query_processing.py       # 🔄 Query processing utilities & multi-server support
├── mcp_client.py             # 🔧 Legacy compatibility module
├── mcp_config_example.json   # 📡 MCP server configuration template
├── mcp_config.json           # 📡 Your local MCP config (gitignored)
├── llm_config.json           # 🤖 Your local LLM config (gitignored)
├── example_configs/          # 📋 Configuration examples
└── examples/                 # 📚 Integration examples
    ├── basic_usage.py
    ├── web_app_integration.py
    └── ai_agent_integration.py
```

## 🎯 For Developers: Using SimpleMCPClient

The `SimpleMCPClient` is designed to be the **main interface** for developers who want to integrate MCP functionality into their applications.

**🔄 Recently Refactored**: The codebase has been streamlined for better maintainability:
- **Eliminated redundant code**: Removed ~250 lines of duplicate functionality
- **Consolidated architecture**: Common utilities moved to `query_processing.py`
- **Improved multi-server support**: Enhanced support for multiple MCP servers
- **Better testing**: More focused and comprehensive test coverage
- **Maintained backward compatibility**: All existing APIs continue to work

### Key Features

- **Clean API**: Simple, intuitive methods for MCP operations
- **Multi-Server Support**: Connect to multiple MCP servers simultaneously
- **Async Support**: Full async/await support with context managers
- **Error Handling**: Comprehensive error handling with clear exceptions
- **Tool Validation**: Automatic tool validation and helpful error messages
- **Connection Management**: Automatic connection lifecycle management
- **FastMCP v2 Compatible**: Built on FastMCP v2 for optimal performance
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
    async def connect_to_all_servers(self) -> Dict[str, List[Dict[str, Any]]]
    def is_connected(self) -> bool
    def get_current_server(self) -> Optional[str]
    def get_connected_servers(self) -> List[str]
    
    # Tool Operations
    def get_available_tools(self) -> List[Dict[str, Any]]
    def get_tools_by_server(self, server_name: str) -> List[Dict[str, Any]]
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str
    async def execute_tool_on_server(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str
    
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

#### 5. Multi-Server Support

```python
async def multi_server_example():
    async with SimpleMCPClient() as client:
        # Connect to all servers at once
        server_tools = await client.connect_to_all_servers()
        
        # Print tools by server
        for server, tools in server_tools.items():
            print(f"{server}: {[tool['name'] for tool in tools]}")
        
        # Get all available tools across servers
        all_tools = client.get_available_tools()
        print(f"Total tools: {len(all_tools)}")
        
        # Execute tool on specific server
        result = await client.execute_tool_on_server(
            "filesystem", "read_file", {"path": "example.txt"}
        )
        
        # Or let the client find the tool automatically
        result = await client.execute_tool("read_file", {"path": "example.txt"})
```

#### 6. Server Management

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
- **Multi-server**: Connect to multiple MCP servers simultaneously
- **Interactive Chat**: Natural language interaction with MCP tools
- **Iterative Tool Execution**: Supports sequential tool calling for complex workflows
- **Tool Execution**: Automatic tool execution based on LLM decisions
- **Server Management**: List, connect, and switch between different MCP servers
- **Debugging**: Verbose logging for development

### CLI Commands

- `/help` - Show available commands
- `/tools` - List available MCP tools from all connected servers
- `/status` - Show connection and configuration status
- `/servers` - List available servers and their connection status
- `/connect <server>` - Connect to a specific MCP server
- `/reconnect` - Reconnect to all configured servers
- `quit/exit/q` - Exit the CLI

## ⚙️ Configuration

### Initial Setup

**Important**: Configuration files containing sensitive information are not included in the repository for security reasons.

1. **Copy the example configuration**:
   ```bash
   cp mcp_config_example.json mcp_config.json
   ```

2. **Edit your local configuration**:
   - Replace placeholder paths with your actual project paths
   - Update database connection strings with your credentials
   - Modify server commands as needed for your environment

3. **Your `mcp_config.json` is gitignored** to prevent accidentally committing sensitive information.

### MCP Server Configuration (`mcp_config.json`)

Create your local `mcp_config.json` based on `mcp_config_example.json`:

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
    },
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://username:password@localhost:5432/database_name"
      ],
      "description": "PostgreSQL database server"
    }
  }
}
```

### LLM Configuration (`llm_config.json`) - CLI Only

Similarly, create your local `llm_config.json` (also gitignored):

```json
{
  "provider": "anthropic",
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1000,
  "options": {}
}
```

### Security Notes

⚠️ **Never commit configuration files containing**:
- API keys or tokens
- Database credentials
- Absolute file paths with personal information
- Any sensitive environment variables

The example files provide safe templates you can customize locally.

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
2. **Multi-Server Support**: Connect to multiple MCP servers simultaneously
3. **Zero Dependencies**: No CLI or LLM dependencies in the core client
4. **FastMCP v2**: Built on latest FastMCP v2 for optimal performance
5. **Async Support**: Full async/await support with proper resource management
6. **Error Handling**: Clear error messages and proper exception handling
7. **Flexible Connection**: Support for both config-based and direct connections
8. **Context Managers**: Automatic cleanup with async context managers

### For Testing

1. **Separate CLI**: Interactive testing interface separate from core client
2. **LLM Integration**: Natural language interaction with MCP tools
3. **Multi-server Support**: Test multiple MCP servers simultaneously
4. **Iterative Tool Execution**: Support for complex sequential workflows
5. **Multi-provider**: Support for different LLM providers
6. **Debugging**: Verbose logging for development and troubleshooting

## 📦 Dependencies

### Core Client (`simple_mcp_client.py`)

- `fastmcp>=2.0.0` - FastMCP v2 protocol implementation
- `python-dotenv>=1.0.0` - Environment variable loading

### Interactive CLI (`interactive_cli.py`)

- `anthropic>=0.8.0` - Anthropic Claude API
- `openai>=1.0.0` - OpenAI API (optional)

### Development

- `pytest>=7.0.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async testing support
- `black>=23.0.0` - Code formatting
- `ruff>=0.1.0` - Fast Python linter
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