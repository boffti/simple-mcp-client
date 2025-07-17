# Simple MCP Client

A minimal implementation of an MCP (Model Context Protocol) client demonstrating core concepts without CLI complexity. This client provides a straightforward way to connect to MCP servers, discover tools, and execute them with AI integration.

## What is MCP?

MCP (Model Context Protocol) allows your application to connect to "servers" that provide tools, resources, and capabilities. Think of it as a way to give your AI applications access to external functionality. A lot of projects and tutorials cover creation of MCP Servers but not a lot cover the creation of Clients. These are especially helpful when you already have an AI application and want to enhance it with MCP powers.

## Features

- **Simple Integration**: Connect to MCP servers with minimal code
- **Configuration-Based**: Use JSON config files for easy server management
- **Tool Discovery**: Automatically discover available tools from servers
- **AI Integration**: Built-in support for Anthropic's Claude API
- **Interactive Mode**: Command-line interface for testing and exploration
- **Verbose Logging**: Detailed debugging output when needed

## Installation

```bash
# Install the package
uv sync
```

## Quick Start

### 1. Set up your environment

```bash
export ANTHROPIC_API_KEY=your_api_key_here
# or
cp .env.example
```

### 2. Create a configuration file

Create `mcp_config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["-m", "mcp_server_filesystem"],
      "description": "File system operations"
    }
  }
}
```

### 3. Run the client

```bash
source .venv/bin/activate

python simple_mcp_client.py
```

## Configuration

The client uses the standard MCP server configuration format in `mcp_config.json`:

```json
{
  "mcpServers": {
    "server_name": {
      "command": "executable_command",
      "args": ["arg1", "arg2"],
      "description": "Server description",
      "env": {
        "VAR": "value"
      }
    }
  }
}
```

### Example Configurations

**Single Server Setup:**

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["-m", "mcp_server_filesystem"],
      "description": "File system operations"
    }
  }
}
```

**Multiple Servers:**

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["-m", "mcp_server_filesystem"],
      "description": "File system operations"
    },
    "web": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-web"],
      "description": "Web scraping and requests"
    },
    "database": {
      "command": "python",
      "args": ["/path/to/database_server.py"],
      "description": "Database operations",
      "env": {
        "DB_HOST": "localhost",
        "DB_PORT": "5432"
      }
    }
  }
}
```

## Usage

### Basic Usage

```python
from simple_mcp_client import SimpleMCPClient
import os

# Initialize client
client = SimpleMCPClient(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    config_path="mcp_config.json",
    verbose=True
)

# Connect to server
await client.connect_to_server("filesystem")

# Process queries
response = await client.process_query("List files in the current directory")
print(response)

# Clean up
await client.cleanup()
```

### Direct Server Connection

```python
# Connect without config file
await client.connect_to_server(
    server_command="python",
    server_args=["-m", "mcp_server_filesystem"]
)
```

### Interactive Mode

Run the script directly for an interactive session:

```bash
python simple_mcp_client.py
```

This will:

1. Load your configuration
2. Connect to the first available server
3. Start an interactive chat session
4. Allow you to query the AI with access to MCP tools

## Core Architecture

### SimpleMCPClient Class

The main client class handles:

- **Server Connection**: Connect to MCP servers via stdio
- **Tool Discovery**: Automatically discover available tools
- **Query Processing**: Process natural language queries with AI
- **Tool Execution**: Execute MCP tools when needed by the AI
- **Response Handling**: Return processed responses

### Key Methods

- `connect_to_server()`: Connect to MCP server (config-based or direct)
- `process_query()`: Process user query with AI and tool execution
- `list_servers()`: List configured servers
- `get_server_info()`: Get server configuration details
- `cleanup()`: Clean up connections and resources

## Development

### Setup for Development

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install optional dependencies
pip install -e ".[optional]"
```

### Code Quality

```bash
# Format code
black simple_mcp_client.py

# Lint code
ruff check simple_mcp_client.py

# Type checking
mypy simple_mcp_client.py
```

### Testing

```bash
# Run tests
pytest

# Run tests with coverage
coverage run -m pytest
coverage report
```

## Examples

The `example_configs/` directory contains sample configurations:

- `filesystem_server.json`: Basic filesystem operations
- `multiple_servers.json`: Multiple server setup example
- `custom_server.json`: Custom server with environment variables

## Environment Variables

- `ANTHROPIC_API_KEY`: Required for AI integration
- Optional: Create `.env` file for environment variables (requires `python-dotenv`)

## Integration Patterns

### Pattern 1: Add to Existing AI App

```python
# Your existing AI application
client = SimpleMCPClient(api_key, "my_app_config.json")
await client.connect_to_server("filesystem")

# Now AI can use MCP tools
response = await client.process_query("Help me analyze this data file")
```

### Pattern 2: Tool Execution Only

```python
# Execute MCP tools directly
client = SimpleMCPClient(api_key)
await client.connect_to_server("api_server")

# Use tools without AI processing
tools = client.available_tools
```

### Pattern 3: Multi-Server Setup

```python
# Connect to different servers as needed
client = SimpleMCPClient(api_key)

# Switch between servers
await client.connect_to_server("filesystem")
await client.connect_to_server("web")
```

## Configuration Benefits

- **Industry Standard**: Uses the widely adopted MCP config format
- **Easy Management**: Define multiple servers in one place
- **Environment Support**: Pass custom environment variables to servers
- **Documentation**: Built-in descriptions for each server
- **Compatibility**: Works with other MCP tools and frameworks

## Common Use Cases

1. **Enhance Existing AI Apps**: Add MCP capabilities to current applications
2. **Tool Automation**: Execute MCP tools programmatically
3. **Multi-Server Integration**: Connect to multiple MCP servers
4. **Development/Testing**: Different configs for different environments

## Troubleshooting

### Common Issues

1. **Server Not Found**: Check that the server command and args are correct
2. **Connection Timeout**: Ensure the MCP server is properly installed
3. **API Key Issues**: Verify `ANTHROPIC_API_KEY` is set correctly
4. **Tool Execution Errors**: Check tool input format matches server expectations

### Verbose Mode

Enable detailed logging for debugging:

```python
client = SimpleMCPClient(api_key, verbose=True)
```

This shows:

- System context sent to AI
- Tool calls and responses
- MCP server communication
- Error details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Related Projects

- [MCP Specification](https://github.com/modelcontextprotocol/specification)
- [MCP Server Examples](https://github.com/modelcontextprotocol/servers)
- [Anthropic MCP Servers](https://github.com/anthropic/mcp-servers)

## Next Steps

1. Start with the basic example
2. Create your `mcp_config.json`
3. Test with provided example configs
4. Integrate into your existing application
5. Add error handling and logging as needed

The key insight is that MCP is just a protocol for connecting to external tools. This implementation focuses on the essential functionality needed for most integration scenarios while remaining simple and maintainable.
