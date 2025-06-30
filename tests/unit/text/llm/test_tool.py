"""Tests for the LLM tool module."""

import os
import re
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from psifx.text.llm.tool import LLMTool
from psifx.io.yaml import YAMLReader


class TestLLMTool:
    """Test the LLMTool class."""

    def test_init(self):
        """Test the initialization of the LLMTool class."""
        tool = LLMTool()
        assert 'default' in tool.parsers
        assert 'hf' in tool.providers
        assert 'ollama' in tool.providers
        assert 'openai' in tool.providers
        assert 'anthropic' in tool.providers

    @patch('psifx.text.llm.tool.TxtReader')
    def test_load_template_from_string(self, mock_txt_reader):
        """Test loading a template from a string."""
        # Setup
        prompt_str = "system: You are a helpful assistant.\nuser: Hello\nassistant: Hi there!"
        
        # Mock TxtReader to raise NameError to simulate string input
        mock_txt_reader.read.side_effect = NameError()
        
        # Execute
        template = LLMTool.load_template(prompt_str)
        
        # Verify
        assert isinstance(template, ChatPromptTemplate)
        # The structure of ChatPromptTemplate has changed, so we just verify it's created correctly
        assert len(template.messages) == 3

    @patch('psifx.text.llm.tool.TxtReader')
    def test_load_template_from_file(self, mock_txt_reader):
        """Test loading a template from a file."""
        # Setup
        prompt_str = "system: You are a helpful assistant.\nuser: Hello\nassistant: Hi there!"
        mock_txt_reader.read.return_value = prompt_str
        
        # Execute
        template = LLMTool.load_template("dummy_path.txt")
        
        # Verify
        mock_txt_reader.read.assert_called_once_with(path="dummy_path.txt")
        assert isinstance(template, ChatPromptTemplate)
        assert len(template.messages) == 3

    def test_default_parser_basic(self):
        """Test the default parser with basic input."""
        # Setup
        tool = LLMTool()
        generation = AIMessage(content="This is a test message")
        data = {"input": "test"}
        
        # Execute
        result = tool.default_parser(generation=generation, data=data)
        
        # Verify
        assert result == "This is a test message"

    def test_default_parser_with_start_after(self):
        """Test the default parser with start_after parameter."""
        # Setup
        tool = LLMTool()
        generation = AIMessage(content="Prefix: This is the actual content")
        data = {"input": "test"}
        
        # Execute
        result = tool.default_parser(generation=generation, data=data, start_after="Prefix:")
        
        # Verify
        assert result == "This is the actual content"

    def test_default_parser_with_regex(self):
        """Test the default parser with regex parameter."""
        # Setup
        tool = LLMTool()
        generation = AIMessage(content="The answer is: 42!")
        data = {"input": "test"}
        
        # Execute
        result = tool.default_parser(generation=generation, data=data, regex=r"(\d+)")
        
        # Verify
        assert result == "42"

    def test_default_parser_with_to_lower(self):
        """Test the default parser with to_lower parameter."""
        # Setup
        tool = LLMTool()
        generation = AIMessage(content="UPPERCASE TEXT")
        data = {"input": "test"}
        
        # Execute
        result = tool.default_parser(generation=generation, data=data, to_lower=True)
        
        # Verify
        assert result == "uppercase text"

    def test_default_parser_with_expect(self):
        """Test the default parser with expect parameter."""
        # Setup
        tool = LLMTool()
        generation = AIMessage(content="yes")
        data = {"input": "test"}
        
        # Execute
        result = tool.default_parser(generation=generation, data=data, expect=["yes", "no"])
        
        # Verify
        assert result == "yes"

    def test_default_parser_with_on_failure(self):
        """Test the default parser with on_failure parameter."""
        # Setup
        tool = LLMTool()
        generation = AIMessage(content="unexpected")
        data = {"input": "test"}
        
        # Execute
        result = tool.default_parser(
            generation=generation, 
            data=data, 
            expect=["yes", "no"], 
            on_failure="unknown"
        )
        
        # Verify
        assert result == "unknown"

    @patch('psifx.text.llm.tool.YAMLReader')
    @patch.object(LLMTool, 'load_template')
    @patch.object(LLMTool, 'make_chain')
    def test_chain_from_yaml(self, mock_make_chain, mock_load_template, mock_yaml_reader):
        """Test creating a chain from a YAML file."""
        # Setup
        tool = LLMTool()
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_yaml_reader.read.return_value = {
            'prompt': 'dummy_prompt',
            'parser': {'kind': 'default'}
        }
        mock_prompt = MagicMock(spec=ChatPromptTemplate)
        mock_load_template.return_value = mock_prompt
        
        # Execute
        tool.chain_from_yaml(llm=mock_llm, yaml_path="dummy_path.yaml")
        
        # Verify
        mock_yaml_reader.read.assert_called_once_with("dummy_path.yaml")
        mock_load_template.assert_called_once_with(prompt='dummy_prompt')
        mock_make_chain.assert_called_once()

    @patch('psifx.text.llm.tool.YAMLReader')
    @patch.object(LLMTool, 'load_template')
    @patch.object(LLMTool, 'make_chain')
    def test_chains_from_yaml(self, mock_make_chain, mock_load_template, mock_yaml_reader):
        """Test creating multiple chains from a YAML file."""
        # Setup
        tool = LLMTool()
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_yaml_reader.read.return_value = {
            'chain1': {'prompt': 'prompt1'},
            'chain2': {'prompt': 'prompt2', 'parser': {'kind': 'default'}}
        }
        
        # Execute
        chains = tool.chains_from_yaml(llm=mock_llm, yaml_path="dummy_path.yaml")
        
        # Verify
        assert len(chains) == 2
        assert 'chain1' in chains
        assert 'chain2' in chains
        assert mock_load_template.call_count == 2
        assert mock_make_chain.call_count == 2

    def test_instantiate_llm_openai(self):
        """Test instantiating an OpenAI LLM."""
        # This test is mocked at a higher level to avoid OpenAI API key issues
        # Setup
        tool = LLMTool(verbose=False)
        
        # Mock the providers dictionary directly
        mock_provider = MagicMock()
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_provider.return_value = mock_llm
        tool.providers['openai'] = mock_provider
        
        # Execute
        result = tool.instantiate_llm(provider='openai', model='gpt-4')
        
        # Verify
        mock_provider.assert_called_once_with(model='gpt-4')
        assert result == mock_llm

    def test_instantiate_llm_ollama(self):
        """Test instantiating an Ollama LLM."""
        # This test is mocked at a higher level to avoid Ollama installation issues
        # Setup
        tool = LLMTool(verbose=False)
        
        # Mock the providers dictionary directly
        mock_provider = MagicMock()
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_provider.return_value = mock_llm
        tool.providers['ollama'] = mock_provider
        
        # Execute
        result = tool.instantiate_llm(provider='ollama', model='llama3')
        
        # Verify
        mock_provider.assert_called_once_with(model='llama3')
        assert result == mock_llm

    def test_instantiate_llm_invalid_provider(self):
        """Test instantiating an LLM with an invalid provider."""
        # Setup
        tool = LLMTool(verbose=False)
        
        # Execute and Verify
        with pytest.raises(NameError, match=r'model provider should be one of:'):
            tool.instantiate_llm(provider='invalid_provider')

    def test_instantiate_parser_default(self):
        """Test instantiating the default parser."""
        # Setup
        tool = LLMTool()
        
        # Execute
        parser = tool.instantiate_parser(kind='default')
        
        # Verify
        assert callable(parser)

    def test_instantiate_parser_invalid(self):
        """Test instantiating an invalid parser."""
        # Setup
        tool = LLMTool()
        
        # Execute and Verify
        with pytest.raises(NameError, match=r'parser kind should be one of:'):
            tool.instantiate_parser(kind='invalid_parser')

    @patch('psifx.text.llm.tool.RunnableParallel')
    @patch('psifx.text.llm.tool.RunnablePassthrough')
    def test_make_chain(self, mock_passthrough, mock_parallel):
        """Test making a chain."""
        # Setup
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_prompt = MagicMock(spec=ChatPromptTemplate)
        mock_parser = lambda generation, data: "parsed_result"
        
        # Mock the chain creation
        mock_chain = MagicMock()
        mock_parallel.return_value = mock_chain
        mock_chain.__or__.return_value = mock_chain
        
        # Execute
        chain = LLMTool.make_chain(llm=mock_llm, prompt=mock_prompt, parser=mock_parser)
        
        # Verify
        assert mock_parallel.called
        assert chain is mock_chain