"""Tests for psifx.utils.command module."""
import pytest
import argparse
from unittest.mock import MagicMock
from psifx.utils.command import Command, has_subparsers, iter_subparsers, register_command, register_main_command

def test_command_base_class():
    """Test Command base class."""
    # Test that NotImplementedError is raised when setup is not implemented
    with pytest.raises(NotImplementedError):
        Command.setup(None)
    
    # Test that NotImplementedError is raised when execute is not implemented
    with pytest.raises(NotImplementedError):
        Command.execute(None, None)

def test_has_subparsers():
    """Test has_subparsers function."""
    # Create a parser without subparsers
    parser = argparse.ArgumentParser()
    assert not has_subparsers(parser)
    
    # Create a parser with subparsers
    parser = argparse.ArgumentParser()
    parser.add_subparsers()
    assert has_subparsers(parser)

def test_iter_subparsers():
    """Test iter_subparsers function."""
    # Create a parser with subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser('sub1')
    subparsers.add_parser('sub2')
    
    # Count the number of subparsers
    count = sum(1 for _ in iter_subparsers(parser))
    assert count == 2

def test_register_command():
    """Test register_command function."""
    # Create a mock Command class
    class MockCommand(Command):
        """Mock command for testing."""
        @staticmethod
        def setup(parser):
            parser.add_argument('--test', help='Test argument')
        
        @staticmethod
        def execute(parser, args):
            pass
    
    # Create a parent parser with subparsers
    parent = argparse.ArgumentParser()
    subparsers = parent.add_subparsers()
    
    # Register the command
    parser = register_command(subparsers, 'mock', MockCommand)
    
    # Check that the parser was created correctly
    assert parser is not None
    
    # Check that the parser has the expected argument
    has_test_arg = any(action.dest == 'test' for action in parser._actions)
    assert has_test_arg

def test_register_main_command():
    """Test register_main_command function."""
    # Create a mock Command class
    class MockMainCommand(Command):
        """Mock main command for testing."""
        @staticmethod
        def setup(parser):
            parser.add_argument('--test', help='Test argument')
        
        @staticmethod
        def execute(parser, args):
            pass
    
    # Register the main command
    parser = register_main_command(MockMainCommand, version='1.0.0')
    
    # Check that the parser was created correctly
    assert parser is not None
    
    # Check that the parser has the expected argument
    has_test_arg = any(action.dest == 'test' for action in parser._actions)
    assert has_test_arg
    
    # Check that the version argument was added
    has_version_arg = any(action.dest == 'version' for action in parser._actions)
    assert has_version_arg