"""
MCP Client Module - Legacy Compatibility
========================================

This module provides backward compatibility imports for the refactored MCP client.
The main functionality has been moved to:
- SimpleMCPClient: simple_mcp_client.py
- MultiServerQueryProcessor: query_processing.py
"""

# Import the main implementations for backward compatibility
from query_processing import MultiServerQueryProcessor

# For any legacy imports that might still exist
__all__ = ["MultiServerQueryProcessor"]
