# tests/test_cli/test_cli.py
"""Tests for CLI interface."""

import pytest
from unittest.mock import patch, Mock
from aws_genai_architecture.cli import main, load_config, create_default_config


class TestCLI:
    """Test cases for CLI interface."""

    def test_load_config(self):
        """Test configuration loading."""
        # Test with non-existent file
        config = load_config("non_existent_file.json")
        assert config == {}

    def test_create_default_config(self):
        """Test default configuration creation."""
        config = create_default_config()

        assert isinstance(config, dict)
        assert "services" in config
        assert "communication" in config
        assert "monitoring" in config
        assert "web" in config

    @patch('sys.argv', ['aws-genai-viz', '--export-config', 'test_config.json'])
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_export_config(self, mock_json_dump, mock_open):
        """Test configuration export."""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        result = main()

        assert result == 0
        mock_open.assert_called_once_with('test_config.json', 'w')
        mock_json_dump.assert_called_once()