"""Tests for psifx.command module."""
import argparse
from unittest.mock import patch, MagicMock


def test_psifx_command_setup():
    """Test PsifxCommand setup method."""
    from psifx.command import PsifxCommand
    parser = argparse.ArgumentParser()
    PsifxCommand.setup(parser)
    assert hasattr(parser, '_subparsers')


def test_psifx_command_execute():
    """Test PsifxCommand execute method."""
    from psifx.command import PsifxCommand
    parser = argparse.ArgumentParser()
    args = argparse.Namespace()

    with patch.object(parser, 'print_help') as mock_print_help:
        PsifxCommand.execute(parser, args)
        mock_print_help.assert_called_once()


def test_get_parser():
    """Test get_parser function."""
    from psifx.command import get_parser
    parser = get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert "psifx" in parser.description.lower()


def test_main():
    """Test main function."""
    from psifx.command import main

    with patch('psifx.command.get_parser') as mock_get_parser:
        mock_args = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_get_parser.return_value = mock_parser

        main()

        mock_get_parser.assert_called_once()
        mock_parser.parse_args.assert_called_once()
        mock_args.execute.assert_called_once_with(mock_args)
