"""Tests for the Hugging Face LLM provider."""

import os
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from psifx.text.llm.hf.tool import HFChat, get_lc_hf, get_transformers_pipeline


class TestHFProvider:
    """Test the Hugging Face provider functions."""

    @patch('psifx.text.llm.hf.tool.get_transformers_pipeline')
    def test_get_lc_hf(self, mock_get_pipeline):
        """Test getting a Hugging Face model."""
        # Setup
        mock_pipeline = MagicMock()
        mock_get_pipeline.return_value = mock_pipeline
        
        # Execute
        result = get_lc_hf(model='mistral', api_key='test-key')
        
        # Verify
        mock_get_pipeline.assert_called_once_with(model='mistral', api_key='test-key')
        assert isinstance(result, HFChat)
        assert result.pipeline == mock_pipeline

    @patch('psifx.text.llm.hf.tool.AutoTokenizer')
    @patch('psifx.text.llm.hf.tool.AutoModelForCausalLM')
    @patch('psifx.text.llm.hf.tool.pipeline')
    def test_get_transformers_pipeline_basic(self, mock_pipeline, mock_auto_model, mock_tokenizer):
        """Test getting a transformers pipeline with basic parameters."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_pipeline_instance = MagicMock()
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_auto_model.from_pretrained.return_value = mock_model_instance
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Execute
        with patch.dict(os.environ, {'HF_TOKEN': 'env-token'}):
            result = get_transformers_pipeline(model='mistral')
        
        # Verify
        mock_tokenizer.from_pretrained.assert_called_once_with('mistral', token='env-token')
        mock_auto_model.from_pretrained.assert_called_once_with('mistral', token='env-token')
        mock_pipeline.assert_called_once_with(
            "text-generation", 
            model=mock_model_instance, 
            tokenizer=mock_tokenizer_instance, 
            return_full_text=False,
            max_new_tokens=None
        )
        assert result == mock_pipeline_instance

    @patch('psifx.text.llm.hf.tool.AutoTokenizer')
    @patch('psifx.text.llm.hf.tool.AutoModelForCausalLM')
    @patch('psifx.text.llm.hf.tool.pipeline')
    @patch('psifx.text.llm.hf.tool.BitsAndBytesConfig')
    def test_get_transformers_pipeline_with_quantization(self, mock_bnb_config, mock_pipeline, mock_auto_model, mock_tokenizer):
        """Test getting a transformers pipeline with quantization."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_config_instance = MagicMock()
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_auto_model.from_pretrained.return_value = mock_model_instance
        mock_pipeline.return_value = mock_pipeline_instance
        mock_bnb_config.return_value = mock_config_instance
        
        # Execute
        result = get_transformers_pipeline(
            model='mistral',
            api_key='test-key',
            quantization='4bit',
            max_new_tokens=100
        )
        
        # Verify
        mock_bnb_config.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once_with('mistral', token='test-key')
        mock_auto_model.from_pretrained.assert_called_once_with(
            'mistral', 
            token='test-key',
            quantization_config=mock_config_instance
        )
        mock_pipeline.assert_called_once_with(
            "text-generation", 
            model=mock_model_instance, 
            tokenizer=mock_tokenizer_instance, 
            return_full_text=False,
            max_new_tokens=100
        )
        assert result == mock_pipeline_instance

    @patch('psifx.text.llm.hf.tool.AutoTokenizer')
    @patch('psifx.text.llm.hf.tool.AutoModelForCausalLM')
    @patch('psifx.text.llm.hf.tool.pipeline')
    @patch('psifx.text.llm.hf.tool.getpass')
    def test_get_transformers_pipeline_gated_repo(self, mock_getpass, mock_pipeline, mock_auto_model, mock_tokenizer):
        """Test getting a transformers pipeline with a gated repository."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_pipeline_instance = MagicMock()
        
        # First call raises GatedRepoError, second call succeeds
        from huggingface_hub.errors import GatedRepoError
        gated_error = GatedRepoError("This model requires authorization")
        env_error = EnvironmentError("Environment error")
        env_error.__cause__ = gated_error
        
        mock_tokenizer.from_pretrained.side_effect = [env_error, mock_tokenizer_instance]
        mock_auto_model.from_pretrained.return_value = mock_model_instance
        mock_pipeline.return_value = mock_pipeline_instance
        mock_getpass.getpass.return_value = 'user-input-token'
        
        # Execute
        result = get_transformers_pipeline(model='mistral')
        
        # Verify
        assert mock_tokenizer.from_pretrained.call_count == 2
        mock_getpass.getpass.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_with('mistral', token='user-input-token')
        mock_auto_model.from_pretrained.assert_called_once_with('mistral', token='user-input-token')
        assert result == mock_pipeline_instance

    def test_hfchat_to_chatml_format(self):
        """Test converting LangChain messages to ChatML format."""
        # Setup
        chat = HFChat(pipeline=MagicMock())
        system_msg = SystemMessage(content="You are a helpful assistant")
        ai_msg = AIMessage(content="I'll help you with that")
        human_msg = HumanMessage(content="Hello")
        
        # Execute
        system_result = chat._to_chatml_format(system_msg)
        ai_result = chat._to_chatml_format(ai_msg)
        human_result = chat._to_chatml_format(human_msg)
        
        # Verify
        assert system_result == {"role": "system", "content": "You are a helpful assistant"}
        assert ai_result == {"role": "assistant", "content": "I'll help you with that"}
        assert human_result == {"role": "user", "content": "Hello"}
        
    def test_hfchat_to_chatml_format_invalid(self):
        """Test converting an invalid message type to ChatML format."""
        # Setup
        chat = HFChat(pipeline=MagicMock())
        invalid_msg = MagicMock()
        
        # Execute and Verify
        with pytest.raises(ValueError, match=r"Unknown message type:"):
            chat._to_chatml_format(invalid_msg)

    @patch.object(HFChat, '_to_chatml_format')
    def test_hfchat_call(self, mock_to_chatml):
        """Test calling the HFChat model."""
        # Setup
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{'generated_text': 'Generated response'}]
        chat = HFChat(pipeline=mock_pipeline)
        
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello")
        ]
        
        # Mock the conversion to ChatML format
        mock_to_chatml.side_effect = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        
        # Execute
        result = chat._call(messages)
        
        # Verify
        assert mock_to_chatml.call_count == 2
        mock_pipeline.assert_called_once_with([
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ])
        assert result == 'Generated response'