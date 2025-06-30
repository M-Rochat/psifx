"""Tests for psifx.audio.manipulation.command module."""
import argparse
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from psifx.audio.manipulation.command import (
    ManipulationCommand,
    ExtractionCommand,
    ConversionCommand,
    SplitCommand,
    MixDownCommand,
    NormalizationCommand,
    TrimCommand
)


class TestManipulationCommand:
    """Test the ManipulationCommand class."""
    
    def test_setup(self):
        """Test setup method."""
        parser = argparse.ArgumentParser()
        ManipulationCommand.setup(parser)
        
        # Verify subparsers were added
        assert hasattr(parser, '_subparsers')
        
    def test_execute(self):
        """Test execute method."""
        parser = argparse.ArgumentParser()
        args = parser.parse_args([])
        
        # Mock print_help to verify it's called
        with patch.object(parser, 'print_help') as mock_print_help:
            ManipulationCommand.execute(parser, args)
            mock_print_help.assert_called_once()


class TestExtractionCommand:
    """Test the ExtractionCommand class."""
    
    def test_setup(self):
        """Test setup method."""
        parser = argparse.ArgumentParser()
        ExtractionCommand.setup(parser)
        
        # Parse args to verify the arguments were added correctly
        args = parser.parse_args(['--video', 'video.mp4', '--audio', 'audio.wav'])
        assert args.video == Path('video.mp4')
        assert args.audio == Path('audio.wav')
        assert args.overwrite is False
        assert args.verbose is True
    
    def test_execute(self):
        """Test execute method."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            video=Path('video.mp4'),
            audio=Path('audio.wav'),
            overwrite=True,
            verbose=False
        )
        
        # Mock ManipulationTool to verify it's called with correct arguments
        with patch('psifx.audio.manipulation.command.ManipulationTool') as MockTool:
            mock_tool_instance = MockTool.return_value
            
            ExtractionCommand.execute(parser, args)
            
            MockTool.assert_called_once_with(overwrite=True, verbose=False)
            mock_tool_instance.extraction.assert_called_once_with(
                video_path=Path('video.mp4'),
                audio_path=Path('audio.wav')
            )


class TestConversionCommand:
    """Test the ConversionCommand class."""
    
    def test_setup(self):
        """Test setup method."""
        parser = argparse.ArgumentParser()
        ConversionCommand.setup(parser)
        
        # Parse args to verify the arguments were added correctly
        args = parser.parse_args(['--audio', 'audio.mp3', '--mono_audio', 'mono.wav'])
        assert args.audio == Path('audio.mp3')
        assert args.mono_audio == Path('mono.wav')
        assert args.overwrite is False
        assert args.verbose is True
    
    def test_execute(self):
        """Test execute method."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            audio=Path('audio.mp3'),
            mono_audio=Path('mono.wav'),
            overwrite=True,
            verbose=False
        )
        
        # Mock ManipulationTool to verify it's called with correct arguments
        with patch('psifx.audio.manipulation.command.ManipulationTool') as MockTool:
            mock_tool_instance = MockTool.return_value
            
            ConversionCommand.execute(parser, args)
            
            MockTool.assert_called_once_with(overwrite=True, verbose=False)
            mock_tool_instance.convert.assert_called_once_with(
                audio_path=Path('audio.mp3'),
                mono_audio_path=Path('mono.wav')
            )


class TestSplitCommand:
    """Test the SplitCommand class."""
    
    def test_setup(self):
        """Test setup method."""
        parser = argparse.ArgumentParser()
        SplitCommand.setup(parser)
        
        # Parse args to verify the arguments were added correctly
        args = parser.parse_args([
            '--stereo_audio', 'stereo.wav',
            '--left_audio', 'left.wav',
            '--right_audio', 'right.wav'
        ])
        assert args.stereo_audio == Path('stereo.wav')
        assert args.left_audio == Path('left.wav')
        assert args.right_audio == Path('right.wav')
        assert args.overwrite is False
        assert args.verbose is True
    
    def test_execute(self):
        """Test execute method."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            stereo_audio=Path('stereo.wav'),
            left_audio=Path('left.wav'),
            right_audio=Path('right.wav'),
            overwrite=True,
            verbose=False
        )
        
        # Mock ManipulationTool to verify it's called with correct arguments
        with patch('psifx.audio.manipulation.command.ManipulationTool') as MockTool:
            mock_tool_instance = MockTool.return_value
            
            SplitCommand.execute(parser, args)
            
            MockTool.assert_called_once_with(overwrite=True, verbose=False)
            mock_tool_instance.split.assert_called_once_with(
                stereo_audio_path=Path('stereo.wav'),
                left_audio_path=Path('left.wav'),
                right_audio_path=Path('right.wav')
            )


class TestMixDownCommand:
    """Test the MixDownCommand class."""
    
    def test_setup(self):
        """Test setup method."""
        parser = argparse.ArgumentParser()
        MixDownCommand.setup(parser)
        
        # Parse args to verify the arguments were added correctly
        args = parser.parse_args([
            '--mono_audios', 'mono1.wav', 'mono2.wav',
            '--mixed_audio', 'mixed.wav'
        ])
        assert args.mono_audios == [Path('mono1.wav'), Path('mono2.wav')]
        assert args.mixed_audio == Path('mixed.wav')
        assert args.overwrite is False
        assert args.verbose is True
    
    def test_execute(self):
        """Test execute method."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            mono_audios=[Path('mono1.wav'), Path('mono2.wav')],
            mixed_audio=Path('mixed.wav'),
            overwrite=True,
            verbose=False
        )
        
        # Mock ManipulationTool to verify it's called with correct arguments
        with patch('psifx.audio.manipulation.command.ManipulationTool') as MockTool:
            mock_tool_instance = MockTool.return_value
            
            MixDownCommand.execute(parser, args)
            
            MockTool.assert_called_once_with(overwrite=True, verbose=False)
            mock_tool_instance.mixdown.assert_called_once_with(
                mono_audio_paths=[Path('mono1.wav'), Path('mono2.wav')],
                mixed_audio_path=Path('mixed.wav')
            )


class TestNormalizationCommand:
    """Test the NormalizationCommand class."""
    
    def test_setup(self):
        """Test setup method."""
        parser = argparse.ArgumentParser()
        NormalizationCommand.setup(parser)
        
        # Parse args to verify the arguments were added correctly
        args = parser.parse_args([
            '--audio', 'audio.wav',
            '--normalized_audio', 'normalized.wav'
        ])
        assert args.audio == Path('audio.wav')
        assert args.normalized_audio == Path('normalized.wav')
        assert args.overwrite is False
        assert args.verbose is True
    
    def test_execute(self):
        """Test execute method."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            audio=Path('audio.wav'),
            normalized_audio=Path('normalized.wav'),
            overwrite=True,
            verbose=False
        )
        
        # Mock ManipulationTool to verify it's called with correct arguments
        with patch('psifx.audio.manipulation.command.ManipulationTool') as MockTool:
            mock_tool_instance = MockTool.return_value
            
            NormalizationCommand.execute(parser, args)
            
            MockTool.assert_called_once_with(overwrite=True, verbose=False)
            mock_tool_instance.normalization.assert_called_once_with(
                audio_path=Path('audio.wav'),
                normalized_audio_path=Path('normalized.wav')
            )


class TestTrimCommand:
    """Test the TrimCommand class."""
    
    def test_setup(self):
        """Test setup method."""
        parser = argparse.ArgumentParser()
        TrimCommand.setup(parser)
        
        # Parse args to verify the arguments were added correctly
        args = parser.parse_args([
            '--audio', 'audio.wav',
            '--trimmed_audio', 'trimmed.wav',
            '--start_time', '1.5',
            '--end_time', '3.0'
        ])
        assert args.audio == Path('audio.wav')
        assert args.trimmed_audio == Path('trimmed.wav')
        assert args.start_time == 1.5
        assert args.end_time == 3.0
        assert args.overwrite is False
        assert args.verbose is True
    
    def test_execute(self):
        """Test execute method."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            audio=Path('audio.wav'),
            trimmed_audio=Path('trimmed.wav'),
            start_time=1.5,
            end_time=3.0,
            overwrite=True,
            verbose=False
        )
        
        # Mock ManipulationTool to verify it's called with correct arguments
        with patch('psifx.audio.manipulation.command.ManipulationTool') as MockTool:
            mock_tool_instance = MockTool.return_value
            
            TrimCommand.execute(parser, args)
            
            MockTool.assert_called_once_with(overwrite=True, verbose=False)
            mock_tool_instance.trim.assert_called_once_with(
                audio_path=Path('audio.wav'),
                trimmed_audio_path=Path('trimmed.wav'),
                start_time=1.5,
                end_time=3.0
            )