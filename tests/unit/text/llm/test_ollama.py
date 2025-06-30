"""Tests for the Ollama LLM provider."""

import pytest
from unittest.mock import patch, MagicMock, call

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from psifx.text.llm.ollama.tool import get_ollama, pull_model


class TestOllamaProvider:
    """Test the Ollama provider functions."""

    @patch('psifx.text.llm.ollama.tool.ChatOllama')
    @patch('psifx.text.llm.ollama.tool.ollama')
    def test_get_ollama_model_exists(self, mock_ollama, mock_chat_ollama):
        """Test getting an Ollama model that already exists."""
        # Setup
        mock_instance = MagicMock(spec=ChatOllama)
        mock_chat_ollama.return_value = mock_instance
        mock_ollama.list.return_value = {'models': [{'model': 'llama3.1'}]}
        
        # Execute
        result = get_ollama(model='llama3.1')
        
        # Verify
        assert mock_ollama.list.call_count > 0
        mock_chat_ollama.assert_called_once_with(model='llama3.1')
        assert result == mock_instance

    @patch('psifx.text.llm.ollama.tool.ChatOllama')
    @patch('psifx.text.llm.ollama.tool.ollama')
    @patch('psifx.text.llm.ollama.tool.pull_model')
    def test_get_ollama_model_not_exists(self, mock_pull_model, mock_ollama, mock_chat_ollama):
        """Test getting an Ollama model that doesn't exist yet."""
        # Setup
        mock_instance = MagicMock(spec=ChatOllama)
        mock_chat_ollama.return_value = mock_instance
        mock_ollama.list.return_value = {'models': [{'model': 'llama3.1'}]}
        
        # Execute
        result = get_ollama(model='mistral')
        
        # Verify
        assert mock_ollama.list.call_count > 0
        mock_pull_model.assert_called_once_with('mistral')
        mock_chat_ollama.assert_called_once_with(model='mistral')
        assert result == mock_instance

    @patch('psifx.text.llm.ollama.tool.ChatOllama')
    @patch('psifx.text.llm.ollama.tool.ollama')
    @patch('psifx.text.llm.ollama.tool.subprocess.Popen')
    @patch('psifx.text.llm.ollama.tool.time')
    def test_get_ollama_service_not_running(self, mock_time, mock_popen, mock_ollama, mock_chat_ollama):
        """Test getting an Ollama model when the service is not running."""
        # Setup
        mock_instance = MagicMock(spec=ChatOllama)
        mock_chat_ollama.return_value = mock_instance
        mock_ollama.list.side_effect = [Exception("Service not running"), {'models': [{'model': 'llama3.1'}]}]
        
        # Mock process
        mock_process = MagicMock()
        mock_process.stderr.readline.return_value = b"Listening on localhost:11434\n"
        mock_popen.return_value = mock_process
        
        # Mock time
        mock_time.time.side_effect = [0, 1]  # Start time, check time
        
        # Execute
        result = get_ollama(model='llama3.1')
        
        # Verify
        assert mock_ollama.list.call_count == 2
        mock_popen.assert_called_once()
        mock_chat_ollama.assert_called_once_with(model='llama3.1')
        assert result == mock_instance

    @patch('psifx.text.llm.ollama.tool.tqdm')
    @patch('psifx.text.llm.ollama.tool.ollama')
    def test_pull_model(self, mock_ollama, mock_tqdm):
        """Test pulling an Ollama model."""
        # Setup
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar
        
        # Mock the generator for ollama.pull
        mock_ollama.pull.return_value = [
            {'status': 'pulling', 'digest': 'digest1', 'total': 100, 'completed': 0},
            {'status': 'pulling', 'digest': 'digest1', 'total': 100, 'completed': 50},
            {'status': 'pulling', 'digest': 'digest1', 'total': 100, 'completed': 100},
            {'status': 'verifying', 'digest': 'digest2', 'total': 50, 'completed': 0},
            {'status': 'verifying', 'digest': 'digest2', 'total': 50, 'completed': 50}
        ]
        
        # Execute
        pull_model('test-model')
        
        # Verify
        mock_ollama.pull.assert_called_once_with('test-model', stream=True)
        assert mock_tqdm.call_count == 2
        assert mock_pbar.close.call_count == 2