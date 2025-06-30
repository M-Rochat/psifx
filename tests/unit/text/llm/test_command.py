"""Tests for the LLM command module."""

import pytest
import argparse
from unittest.mock import patch, MagicMock
from pathlib import Path

from psifx.text.llm.command import format_llm_namespace, add_llm_argument


class TestLLMCommand:
    """Test the LLM command functions."""

    def test_add_llm_argument(self):
        """Test adding LLM arguments to a parser."""
        # Setup
        parser = argparse.ArgumentParser()
        
        # Execute
        add_llm_argument(parser)
        
        # Verify
        # Get all argument names
        arg_names = [action.dest for action in parser._actions]
        
        assert 'provider' in arg_names
        assert 'model' in arg_names
        assert 'model_config' in arg_names
        assert 'api_key' in arg_names

    @patch('psifx.text.llm.command.YAMLReader')
    def test_format_llm_namespace_with_defaults(self, mock_yaml_reader):
        """Test formatting LLM namespace with default values."""
        # Setup
        mock_yaml_reader.read.return_value = {}
        args = argparse.Namespace(model_config=None, api_key=None)
        
        # Execute
        result = format_llm_namespace(args)
        
        # Verify
        assert result.provider == 'ollama'
        assert result.model == 'llama3.1:8b'
        assert not hasattr(result, 'api_key')

    @patch('psifx.text.llm.command.YAMLReader')
    def test_format_llm_namespace_with_args(self, mock_yaml_reader):
        """Test formatting LLM namespace with arguments."""
        # Setup
        mock_yaml_reader.read.return_value = {}
        args = argparse.Namespace(
            model_config=None,
            provider='openai',
            model='gpt-4',
            api_key='test-key'
        )
        
        # Execute
        result = format_llm_namespace(args)
        
        # Verify
        assert result.provider == 'openai'
        assert result.model == 'gpt-4'
        assert result.api_key == 'test-key'

    @patch('psifx.text.llm.command.YAMLReader')
    def test_format_llm_namespace_with_config(self, mock_yaml_reader):
        """Test formatting LLM namespace with config file."""
        # Setup
        mock_yaml_reader.read.return_value = {
            'provider': 'anthropic',
            'model': 'claude-3',
            'temperature': 0.7
        }
        args = argparse.Namespace(
            model_config=Path('dummy_config.yaml'),
            api_key='test-key'
        )
        
        # Execute
        result = format_llm_namespace(args)
        
        # Verify
        assert result.provider == 'anthropic'
        assert result.model == 'claude-3'
        assert result.temperature == 0.7
        assert result.api_key == 'test-key'

    @patch('psifx.text.llm.command.YAMLReader')
    def test_format_llm_namespace_api_key_in_config_error(self, mock_yaml_reader):
        """Test error when api_key is in config file."""
        # Setup
        mock_yaml_reader.read.return_value = {
            'provider': 'openai',
            'model': 'gpt-4',
            'api_key': 'config-key'
        }
        args = argparse.Namespace(
            model_config=Path('dummy_config.yaml')
        )
        
        # Execute and Verify
        with pytest.raises(AssertionError, match=r'api_key should not be specified in the model config file'):
            format_llm_namespace(args)

    @patch('psifx.text.llm.command.YAMLReader')
    def test_format_llm_namespace_duplicate_provider_error(self, mock_yaml_reader):
        """Test error when provider is specified in both args and config."""
        # Setup
        mock_yaml_reader.read.return_value = {
            'provider': 'openai',
            'model': 'gpt-4'
        }
        args = argparse.Namespace(
            model_config=Path('dummy_config.yaml'),
            provider='anthropic'
        )
        
        # Execute and Verify
        with pytest.raises(AssertionError, match=r'provider is specified both as an argument and in the model config file'):
            format_llm_namespace(args)

    @patch('psifx.text.llm.command.YAMLReader')
    def test_format_llm_namespace_duplicate_model_error(self, mock_yaml_reader):
        """Test error when model is specified in both args and config."""
        # Setup
        mock_yaml_reader.read.return_value = {
            'provider': 'openai',
            'model': 'gpt-4'
        }
        args = argparse.Namespace(
            model_config=Path('dummy_config.yaml'),
            model='gpt-3.5'
        )
        
        # Execute and Verify
        with pytest.raises(AssertionError, match=r'model is specified both as an argument and in the model config file'):
            format_llm_namespace(args)