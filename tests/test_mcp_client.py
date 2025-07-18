"""
Tests for backward compatibility of mcp_client module.

This module now only provides import compatibility for the refactored architecture.
The actual functionality is tested in:
- test_simple_mcp_client.py
- test_query_processing.py
"""

import pytest


class TestMCPClientBackwardCompatibility:
    """Test backward compatibility imports."""

    def test_import_multiserver_query_processor(self):
        """Test that MultiServerQueryProcessor can be imported from mcp_client."""
        from mcp_client import MultiServerQueryProcessor

        assert MultiServerQueryProcessor is not None

    def test_import_from_query_processing(self):
        """Test that MultiServerQueryProcessor can be imported from query_processing."""
        from query_processing import MultiServerQueryProcessor

        assert MultiServerQueryProcessor is not None

    def test_same_class_imported(self):
        """Test that imports from both modules reference the same class."""
        from mcp_client import MultiServerQueryProcessor as MCPMultiServerQueryProcessor
        from query_processing import MultiServerQueryProcessor as QPMultiServerQueryProcessor

        assert MCPMultiServerQueryProcessor is QPMultiServerQueryProcessor

    def test_module_all_attribute(self):
        """Test that __all__ is properly defined."""
        import mcp_client

        assert hasattr(mcp_client, "__all__")
        assert "MultiServerQueryProcessor" in mcp_client.__all__
