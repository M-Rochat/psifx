"""Tests for the Anthropic LLM provider."""

import os
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic

from psifx.text.llm.anthropic.tool import get_anthropic


class TestAnthropicProvider:
    """Test the Anthropic provider functions."""

    @patch('psifx.text.llm.anthropic.tool.ChatAnthropic')
    def test_get_anthropic_default_params(self, mock_chat_anthropic):
        """Test getting an Anthropic model with default parameters."""
        # Setup
        mock_instance = MagicMock(spec=ChatAnthropic)
        mock_chat_anthropic.return_value = mock_instance
        
        # Execute
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            result = get_anthropic()
            
            # Verify (inside the context manager to ensure the env var is still set)
            mock_chat_anthropic.assert_called_once_with(
                model='claude-3-5-sonnet-20240620',
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2
            )
            assert result == mock_instance
            assert os.environ.get('ANTHROPIC_API_KEY') == 'test-key'

    @patch('psifx.text.llm.anthropic.tool.ChatAnthropic')
    def test_get_anthropic_custom_params(self, mock_chat_anthropic):
        """Test getting an Anthropic model with custom parameters."""
        # Setup
        mock_instance = MagicMock(spec=ChatAnthropic)
        mock_chat_anthropic.return_value = mock_instance
        
        # Execute
        result = get_anthropic(
            model='claude-3-opus',
            api_key='custom-key',
            temperature=0.7,
            max_tokens=100,
            timeout=30,
            max_retries=3,
            custom_param='value'
        )
        
        # Verify
        mock_chat_anthropic.assert_called_once_with(
            model='claude-3-opus',
            temperature=0.7,
            max_tokens=100,
            timeout=30,
            max_retries=3,
            custom_param='value'
        )
        assert result == mock_instance
        assert os.environ.get('ANTHROPIC_API_KEY') == 'custom-key'

    @patch('psifx.text.llm.anthropic.tool.ChatAnthropic')
    def test_get_anthropic_no_api_key(self, mock_chat_anthropic):
        """Test getting an Anthropic model without an API key."""
        # Setup
        mock_instance = MagicMock(spec=ChatAnthropic)
        mock_chat_anthropic.return_value = mock_instance
        
        # Execute
        with patch.dict(os.environ, {}, clear=True):
            result = get_anthropic()
        
        # Verify
        mock_chat_anthropic.assert_called_once()
        assert result == mock_instance