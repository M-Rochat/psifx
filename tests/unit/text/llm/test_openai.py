"""Tests for the OpenAI LLM provider."""

import os
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from psifx.text.llm.openai.tool import get_openai


class TestOpenAIProvider:
    """Test the OpenAI provider functions."""

    @patch('psifx.text.llm.openai.tool.ChatOpenAI')
    def test_get_openai_default_params(self, mock_chat_openai):
        """Test getting an OpenAI model with default parameters."""
        # Setup
        mock_instance = MagicMock(spec=ChatOpenAI)
        mock_chat_openai.return_value = mock_instance
        
        # Execute
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            result = get_openai()
        
        # Verify
        mock_chat_openai.assert_called_once_with(
            model='gpt-4o',
            api_key='test-key',
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        assert result == mock_instance

    @patch('psifx.text.llm.openai.tool.ChatOpenAI')
    def test_get_openai_custom_params(self, mock_chat_openai):
        """Test getting an OpenAI model with custom parameters."""
        # Setup
        mock_instance = MagicMock(spec=ChatOpenAI)
        mock_chat_openai.return_value = mock_instance
        
        # Execute
        result = get_openai(
            model='gpt-3.5-turbo',
            api_key='custom-key',
            temperature=0.7,
            max_tokens=100,
            timeout=30,
            max_retries=3,
            custom_param='value'
        )
        
        # Verify
        mock_chat_openai.assert_called_once_with(
            model='gpt-3.5-turbo',
            api_key='custom-key',
            temperature=0.7,
            max_tokens=100,
            timeout=30,
            max_retries=3,
            custom_param='value'
        )
        assert result == mock_instance