"""Tests for psifx.video.manipulation.tool module."""
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock ffmpeg module
import sys
sys.modules['ffmpeg'] = MagicMock()
import numpy as np

from psifx.video.manipulation.tool import ManipulationTool


@pytest.fixture
def manipulation_tool():
    """Create a ManipulationTool instance for testing."""
    return ManipulationTool(overwrite=True, verbose=False)


@pytest.fixture
def sample_video_path(temp_dir):
    """Return a path for a sample video file."""
    return Path(temp_dir) / "sample_video.mp4"


def test_manipulation_tool_init():
    """Test ManipulationTool initialization."""
    # Test with default parameters
    tool = ManipulationTool()
    assert tool.device == "cpu"
    assert not tool.overwrite
    assert tool.verbose is True

    # Test with custom parameters
    tool = ManipulationTool(overwrite=True, verbose=False)
    assert tool.device == "cpu"
    assert tool.overwrite
    assert not tool.verbose


@patch("ffmpeg.output")
@patch("ffmpeg.input")
def test_process_trim(mock_input, mock_output, manipulation_tool, sample_video_path, temp_dir):
    """Test video trimming functionality."""
    # Setup
    output_path = Path(temp_dir) / "output_video.mp4"
    
    # Mock ffmpeg input and output
    mock_input.return_value = MagicMock()
    mock_input.return_value.video = MagicMock()
    mock_input.return_value.audio = MagicMock()
    mock_output.return_value = MagicMock()
    
    # Call the method
    manipulation_tool.process(
        in_video_path=sample_video_path,
        out_video_path=output_path,
        start=10.0,
        end=20.0
    )
    
    # Assertions
    mock_input.assert_called_once_with(str(sample_video_path), ss=10.0, to=20.0)
    mock_output.assert_called_once()


@patch("ffmpeg.output")
@patch("ffmpeg.input")
def test_process_crop(mock_input, mock_output, manipulation_tool, sample_video_path, temp_dir):
    """Test video cropping functionality."""
    # Setup
    output_path = Path(temp_dir) / "output_video.mp4"
    
    # Mock ffmpeg input and output
    mock_input.return_value = MagicMock()
    mock_input.return_value.video = MagicMock()
    mock_input.return_value.audio = MagicMock()
    mock_output.return_value = MagicMock()
    
    # Call the method
    manipulation_tool.process(
        in_video_path=sample_video_path,
        out_video_path=output_path,
        x_min=100,
        y_min=100,
        x_max=500,
        y_max=400
    )
    
    # Assertions
    mock_input.assert_called_once_with(str(sample_video_path))
    mock_input.return_value.video.crop.assert_called_once_with(
        x=100, y=100, width=400, height=300
    )
    mock_output.assert_called_once()


@patch("ffmpeg.output")
@patch("ffmpeg.input")
def test_process_resize(mock_input, mock_output, manipulation_tool, sample_video_path, temp_dir):
    """Test video resizing functionality."""
    # Setup
    output_path = Path(temp_dir) / "output_video.mp4"
    
    # Mock ffmpeg input and output
    mock_input.return_value = MagicMock()
    mock_input.return_value.video = MagicMock()
    mock_input.return_value.audio = MagicMock()
    mock_output.return_value = MagicMock()
    
    # Call the method
    manipulation_tool.process(
        in_video_path=sample_video_path,
        out_video_path=output_path,
        width=1280,
        height=720
    )
    
    # Assertions
    mock_input.assert_called_once_with(str(sample_video_path))
    mock_input.return_value.video.filter.assert_called_once_with("scale", "1280x720")
    mock_output.assert_called_once()


@patch("ffmpeg.output")
@patch("ffmpeg.input")
def test_process_all_operations(mock_input, mock_output, manipulation_tool, sample_video_path, temp_dir):
    """Test all video operations together (trim, crop, resize)."""
    # Setup
    output_path = Path(temp_dir) / "output_video.mp4"
    
    # Mock ffmpeg input and output
    mock_input.return_value = MagicMock()
    mock_input.return_value.video = MagicMock()
    mock_input.return_value.audio = MagicMock()
    mock_output.return_value = MagicMock()
    
    # Call the method
    manipulation_tool.process(
        in_video_path=sample_video_path,
        out_video_path=output_path,
        start=10.0,
        end=20.0,
        x_min=100,
        y_min=100,
        x_max=500,
        y_max=400,
        width=1280,
        height=720
    )
    
    # Assertions
    mock_input.assert_called_once_with(str(sample_video_path), ss=10.0, to=20.0)
    mock_input.return_value.video.crop.assert_called_once_with(
        x=100, y=100, width=400, height=300
    )
    mock_input.return_value.video.crop.return_value.filter.assert_called_once_with("scale", "1280x720")
    mock_output.assert_called_once()


def test_process_invalid_parameters(manipulation_tool, sample_video_path, temp_dir):
    """Test error handling for invalid parameters."""
    output_path = Path(temp_dir) / "output_video.mp4"
    
    # Test with invalid crop parameters (x_min > x_max)
    with pytest.raises(AssertionError):
        manipulation_tool.process(
            in_video_path=sample_video_path,
            out_video_path=output_path,
            x_min=500,
            y_min=100,
            x_max=100,
            y_max=400
        )
    
    # Test with invalid crop parameters (y_min > y_max)
    with pytest.raises(AssertionError):
        manipulation_tool.process(
            in_video_path=sample_video_path,
            out_video_path=output_path,
            x_min=100,
            y_min=400,
            x_max=500,
            y_max=100
        )
    
    # Test with partial crop parameters (missing y_min)
    with pytest.raises(AssertionError):
        manipulation_tool.process(
            in_video_path=sample_video_path,
            out_video_path=output_path,
            x_min=100,
            x_max=500,
            y_max=400
        )
    
    # Test with partial resize parameters (missing height)
    with pytest.raises(AssertionError):
        manipulation_tool.process(
            in_video_path=sample_video_path,
            out_video_path=output_path,
            width=1280
        )


@patch("pathlib.Path.exists")
@patch("pathlib.Path.unlink")
@patch("ffmpeg.output")
@patch("ffmpeg.input")
def test_process_file_exists(mock_input, mock_output, mock_unlink, mock_exists, 
                            manipulation_tool, sample_video_path, temp_dir):
    """Test handling of existing output files."""
    # Setup
    output_path = Path(temp_dir) / "output_video.mp4"
    
    # Mock ffmpeg input and output
    mock_input.return_value = MagicMock()
    mock_input.return_value.video = MagicMock()
    mock_input.return_value.audio = MagicMock()
    mock_output.return_value = MagicMock()
    
    # Test with overwrite=True
    mock_exists.return_value = True
    manipulation_tool.process(
        in_video_path=sample_video_path,
        out_video_path=output_path
    )
    mock_unlink.assert_called_once()
    
    # Test with overwrite=False
    mock_unlink.reset_mock()
    manipulation_tool.overwrite = False
    mock_exists.return_value = True
    with pytest.raises(FileExistsError):
        manipulation_tool.process(
            in_video_path=sample_video_path,
            out_video_path=output_path
        )
    mock_unlink.assert_not_called()


def test_process_same_input_output(manipulation_tool, sample_video_path):
    """Test error handling when input and output paths are the same."""
    with pytest.raises(AssertionError):
        manipulation_tool.process(
            in_video_path=sample_video_path,
            out_video_path=sample_video_path
        )