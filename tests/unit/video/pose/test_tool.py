"""Tests for psifx.video.pose.tool module."""
import os
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
from PIL import Image

# Mock modules
import sys
sys.modules['psifx.io.video'] = MagicMock()
sys.modules['psifx.io.tar'] = MagicMock()

from psifx.video.pose.tool import PoseEstimationTool


@pytest.fixture
def pose_tool():
    """Create a PoseEstimationTool instance for testing."""
    class TestPoseEstimationTool(PoseEstimationTool):
        """Test implementation of PoseEstimationTool."""
        def inference(self, video_path, poses_path):
            """Override abstract method for testing."""
            pass
    
    return TestPoseEstimationTool(device="cpu", overwrite=True, verbose=False)


@pytest.fixture
def sample_video_path(temp_dir):
    """Return a path for a sample video file."""
    return Path(temp_dir) / "sample_video.mp4"


@pytest.fixture
def sample_poses_path(temp_dir):
    """Return a path for a sample poses archive."""
    return Path(temp_dir) / "sample_poses.tar.xz"


@pytest.fixture
def sample_visualization_path(temp_dir):
    """Return a path for a sample visualization file."""
    return Path(temp_dir) / "sample_visualization.mp4"


@pytest.fixture
def sample_poses_data():
    """Create sample pose data for testing."""
    # Create a simple pose data structure
    edges = {
        "pose_keypoints_2d": [(0, 1), (1, 2)],
        "face_keypoints_2d": [(0, 1), (1, 2)],
    }
    
    # Create pose data for 3 frames
    poses = {}
    for i in range(3):
        poses[f"{i:015d}.json"] = json.dumps({
            "pose_keypoints_2d": [100.0, 200.0, 0.9, 150.0, 250.0, 0.8, 200.0, 300.0, 0.7],
            "face_keypoints_2d": [110.0, 210.0, 0.95, 160.0, 260.0, 0.85]
        })
    
    # Add edges
    poses["edges.json"] = json.dumps(edges)
    
    return poses


def test_pose_tool_init():
    """Test PoseEstimationTool initialization."""
    class TestPoseEstimationTool(PoseEstimationTool):
        """Test implementation of PoseEstimationTool."""
        def inference(self, video_path, poses_path):
            """Override abstract method for testing."""
            pass
    
    # Test with default parameters
    tool = TestPoseEstimationTool()
    assert tool.device == "cpu"
    assert not tool.overwrite
    assert tool.verbose is True
    
    # Test with custom parameters
    tool = TestPoseEstimationTool(device="cuda", overwrite=True, verbose=False)
    assert tool.device == "cuda"
    assert tool.overwrite
    assert not tool.verbose


def test_inference_abstract_method():
    """Test that the base inference method raises NotImplementedError."""
    tool = PoseEstimationTool()
    with pytest.raises(NotImplementedError):
        tool.inference("video.mp4", "poses.tar.xz")


@patch("psifx.io.tar.TarReader.read")
@patch("psifx.io.tar.TarReader.check")
@patch("psifx.io.video.VideoReader")
@patch("psifx.io.video.VideoWriter")
def test_visualization(mock_video_writer, mock_video_reader, mock_tar_check, 
                      mock_tar_read, pose_tool, sample_video_path, 
                      sample_poses_path, sample_visualization_path, sample_poses_data):
    """Test the visualization method."""
    # Setup mocks
    mock_tar_read.return_value = sample_poses_data
    
    # Mock video reader and writer
    mock_video_reader_instance = MagicMock()
    mock_video_reader_instance.frame_rate = 30
    mock_video_reader_instance.__iter__.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)
    ]
    mock_video_reader.return_value.__enter__.return_value = mock_video_reader_instance
    
    mock_video_writer_instance = MagicMock()
    mock_video_writer.return_value.__enter__.return_value = mock_video_writer_instance
    
    # Call the method
    pose_tool.visualization(
        video_path=sample_video_path,
        poses_path=sample_poses_path,
        visualization_path=sample_visualization_path,
        confidence_threshold=0.8
    )
    
    # Assertions
    mock_tar_check.assert_called_once_with(path=sample_poses_path)
    mock_tar_read.assert_called_once_with(sample_poses_path, verbose=False)
    mock_video_reader.assert_called_once()
    mock_video_writer.assert_called_once()
    
    # Check that write was called for each frame
    assert mock_video_writer_instance.write.call_count == 3


@patch("psifx.io.tar.TarReader.read")
@patch("psifx.io.tar.TarReader.check")
@patch("psifx.io.video.VideoReader")
@patch("psifx.io.video.VideoWriter")
def test_visualization_missing_edges(mock_video_writer, mock_video_reader, mock_tar_check, 
                                   mock_tar_read, pose_tool, sample_video_path, 
                                   sample_poses_path, sample_visualization_path):
    """Test visualization when edges.json is missing."""
    # Setup mocks - create poses data without edges
    poses_data = {
        "000000000000000.json": json.dumps({
            "pose_keypoints_2d": [100.0, 200.0, 0.9, 150.0, 250.0, 0.8, 200.0, 300.0, 0.7],
            "face_keypoints_2d": [110.0, 210.0, 0.95, 160.0, 260.0, 0.85]
        })
    }
    mock_tar_read.return_value = poses_data
    
    # Mock video reader and writer
    mock_video_reader_instance = MagicMock()
    mock_video_reader_instance.frame_rate = 30
    mock_video_reader_instance.__iter__.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(1)
    ]
    mock_video_reader.return_value.__enter__.return_value = mock_video_reader_instance
    
    mock_video_writer_instance = MagicMock()
    mock_video_writer.return_value.__enter__.return_value = mock_video_writer_instance
    
    # Call the method
    pose_tool.visualization(
        video_path=sample_video_path,
        poses_path=sample_poses_path,
        visualization_path=sample_visualization_path
    )
    
    # Assertions
    mock_tar_check.assert_called_once_with(path=sample_poses_path)
    mock_tar_read.assert_called_once_with(sample_poses_path, verbose=False)
    mock_video_reader.assert_called_once()
    mock_video_writer.assert_called_once()
    
    # Check that write was called for each frame
    assert mock_video_writer_instance.write.call_count == 1


@patch("psifx.io.tar.TarReader.read")
@patch("psifx.io.tar.TarReader.check")
@patch("psifx.io.video.VideoReader")
@patch("psifx.io.video.VideoWriter")
def test_visualization_confidence_threshold(mock_video_writer, mock_video_reader, mock_tar_check, 
                                          mock_tar_read, pose_tool, sample_video_path, 
                                          sample_poses_path, sample_visualization_path, sample_poses_data):
    """Test visualization with different confidence thresholds."""
    # Setup mocks
    mock_tar_read.return_value = sample_poses_data
    
    # Mock video reader and writer
    mock_video_reader_instance = MagicMock()
    mock_video_reader_instance.frame_rate = 30
    mock_video_reader_instance.__iter__.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)
    ]
    mock_video_reader.return_value.__enter__.return_value = mock_video_reader_instance
    
    mock_video_writer_instance = MagicMock()
    mock_video_writer.return_value.__enter__.return_value = mock_video_writer_instance
    
    # Test with high confidence threshold (should filter out some keypoints)
    pose_tool.visualization(
        video_path=sample_video_path,
        poses_path=sample_poses_path,
        visualization_path=sample_visualization_path,
        confidence_threshold=0.9
    )
    
    # Assertions
    mock_tar_check.assert_called_once_with(path=sample_poses_path)
    mock_tar_read.assert_called_once_with(sample_poses_path, verbose=False)
    mock_video_reader.assert_called_once()
    mock_video_writer.assert_called_once()
    
    # Check that write was called for each frame
    assert mock_video_writer_instance.write.call_count == 3


def test_visualization_invalid_confidence_threshold(pose_tool, sample_video_path, 
                                                  sample_poses_path, sample_visualization_path):
    """Test error handling for invalid confidence threshold."""
    # Test with confidence threshold < 0
    with pytest.raises(AssertionError):
        pose_tool.visualization(
            video_path=sample_video_path,
            poses_path=sample_poses_path,
            visualization_path=sample_visualization_path,
            confidence_threshold=-0.1
        )
    
    # Test with confidence threshold > 1
    with pytest.raises(AssertionError):
        pose_tool.visualization(
            video_path=sample_video_path,
            poses_path=sample_poses_path,
            visualization_path=sample_visualization_path,
            confidence_threshold=1.1
        )