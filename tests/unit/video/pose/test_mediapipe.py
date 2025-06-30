"""Tests for psifx.video.pose.mediapipe module."""
import os
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import psifx.video.pose.mediapipe.skeleton as skeleton
# Create a mock for skeleton module
class MockSkeleton:
    N_POSE_LANDMARKS = 33
    N_FACE_LANDMARKS = 468
    N_LEFT_HAND_LANDMARKS = 21
    POSE_EDGES = [(0, 1), (1, 2)]
    FACE_EDGES = [(0, 1), (1, 2)]
    LEFT_HAND_EDGES = [(0, 1), (1, 2)]
    RIGHT_HAND_EDGES = [(0, 1), (1, 2)]

from psifx.video.pose.mediapipe.tool import MediaPipePoseEstimationTool, MediaPipePoseEstimationAndSegmentationTool


@pytest.fixture
def mediapipe_tool():
    """Create a MediaPipePoseEstimationTool instance for testing."""
    return MediaPipePoseEstimationTool(
        model_complexity=1,
        smooth=True,
        device="cpu",
        overwrite=True,
        verbose=False
    )


@pytest.fixture
def mediapipe_segmentation_tool():
    """Create a MediaPipePoseEstimationAndSegmentationTool instance for testing."""
    return MediaPipePoseEstimationAndSegmentationTool(
        model_complexity=1,
        smooth=True,
        mask_threshold=0.1,
        device="cpu",
        overwrite=True,
        verbose=False
    )


@pytest.fixture
def sample_video_path(temp_dir):
    """Return a path for a sample video file."""
    return Path(temp_dir) / "sample_video.mp4"


@pytest.fixture
def sample_poses_path(temp_dir):
    """Return a path for a sample poses archive."""
    return Path(temp_dir) / "sample_poses.tar.xz"


@pytest.fixture
def sample_masks_path(temp_dir):
    """Return a path for a sample masks video file."""
    return Path(temp_dir) / "sample_masks.mp4"


@pytest.fixture
def mock_mediapipe_results():
    """Create mock MediaPipe results for testing."""
    # Create a mock landmark
    class MockLandmark:
        def __init__(self, x, y, z, visibility):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility
    
    # Create a mock results object
    class MockResults:
        def __init__(self):
            # Create pose landmarks
            self.pose_landmarks = MagicMock()
            self.pose_landmarks.landmark = [
                MockLandmark(0.1, 0.2, 0.3, 0.9),
                MockLandmark(0.2, 0.3, 0.4, 0.8),
                MockLandmark(0.3, 0.4, 0.5, 0.7)
            ]
            
            # Create face landmarks
            self.face_landmarks = MagicMock()
            self.face_landmarks.landmark = [
                MockLandmark(0.15, 0.25, 0.35, 0.95),
                MockLandmark(0.25, 0.35, 0.45, 0.85)
            ]
            
            # Create hand landmarks
            self.left_hand_landmarks = MagicMock()
            self.left_hand_landmarks.landmark = [
                MockLandmark(0.4, 0.5, 0.6, 0.75),
                MockLandmark(0.5, 0.6, 0.7, 0.65)
            ]
            
            self.right_hand_landmarks = MagicMock()
            self.right_hand_landmarks.landmark = [
                MockLandmark(0.45, 0.55, 0.65, 0.8),
                MockLandmark(0.55, 0.65, 0.75, 0.7)
            ]
            
            # Create segmentation mask for segmentation tool
            self.segmentation_mask = np.random.random((100, 100)).astype(np.float32)
    
    return MockResults()


def test_mediapipe_tool_init():
    """Test MediaPipePoseEstimationTool initialization."""
    # Test with default parameters
    tool = MediaPipePoseEstimationTool()
    assert tool.device == "cpu"
    assert tool.model_complexity == 2
    assert tool.smooth is True
    assert not tool.overwrite
    assert tool.verbose is True
    
    # Test with custom parameters
    tool = MediaPipePoseEstimationTool(
        model_complexity=1,
        smooth=False,
        device="cpu",
        overwrite=True,
        verbose=False
    )
    assert tool.device == "cpu"
    assert tool.model_complexity == 1
    assert not tool.smooth
    assert tool.overwrite
    assert not tool.verbose
    
    # Test with invalid device
    with pytest.raises(AssertionError):
        MediaPipePoseEstimationTool(device="cuda")


def test_mediapipe_segmentation_tool_init():
    """Test MediaPipePoseEstimationAndSegmentationTool initialization."""
    # Test with default parameters
    tool = MediaPipePoseEstimationAndSegmentationTool()
    assert tool.device == "cpu"
    assert tool.model_complexity == 2
    assert tool.smooth is True
    assert tool.mask_threshold == 0.1
    assert not tool.overwrite
    assert tool.verbose is True
    
    # Test with custom parameters
    tool = MediaPipePoseEstimationAndSegmentationTool(
        model_complexity=1,
        smooth=False,
        mask_threshold=0.5,
        device="cpu",
        overwrite=True,
        verbose=False
    )
    assert tool.device == "cpu"
    assert tool.model_complexity == 1
    assert not tool.smooth
    assert tool.mask_threshold == 0.5
    assert tool.overwrite
    assert not tool.verbose
    
    # Test with invalid mask_threshold
    with pytest.raises(AssertionError):
        MediaPipePoseEstimationAndSegmentationTool(mask_threshold=-0.1)
    
    with pytest.raises(AssertionError):
        MediaPipePoseEstimationAndSegmentationTool(mask_threshold=1.1)


def test_process_part(mediapipe_tool, mock_mediapipe_results):
    """Test the process_part method."""
    # Test with valid landmarks
    size = (100, 200)  # (height, width)
    result = mediapipe_tool.process_part(
        landmarks=mock_mediapipe_results.pose_landmarks,
        size=size,
        n_points=3
    )
    
    # Check that the result is a list of floats
    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)
    
    # Check that the coordinates are scaled correctly
    # Original x=0.1, y=0.2 should be scaled to x=19.9, y=19.9
    assert len(result) == 9  # 3 points * 3 values (x, y, visibility)
    assert result[0] == pytest.approx(0.1 * (200 - 1))  # x coordinate of first point
    assert result[1] == pytest.approx(0.2 * (100 - 1))  # y coordinate of first point
    assert result[2] == pytest.approx(0.9)  # visibility of first point
    
    # Test with None landmarks
    result = mediapipe_tool.process_part(
        landmarks=None,
        size=size,
        n_points=3
    )
    
    # Check that the result is a list of zeros
    assert isinstance(result, list)
    assert len(result) == 9  # 3 points * 3 values
    assert all(x == 0.0 for x in result)


def test_process_pose(mediapipe_tool, mock_mediapipe_results):
    """Test the process_pose method."""
    size = (100, 200)  # (height, width)
    result = mediapipe_tool.process_pose(
        results=mock_mediapipe_results,
        size=size
    )
    
    # Check that the result contains all expected parts
    assert "pose_keypoints_2d" in result
    assert "face_keypoints_2d" in result
    assert "hand_left_keypoints_2d" in result
    assert "hand_right_keypoints_2d" in result

    # Check that each part contains the correct number of values
    assert len(result["pose_keypoints_2d"]) == skeleton.N_POSE_LANDMARKS * 3
    assert len(result["face_keypoints_2d"]) == skeleton.N_FACE_LANDMARKS * 3
    assert len(result["hand_left_keypoints_2d"]) == skeleton.N_LEFT_HAND_LANDMARKS * 3
    assert len(result["hand_right_keypoints_2d"]) == skeleton.N_LEFT_HAND_LANDMARKS * 3


@patch("psifx.io.tar.TarWriter.write")
@patch("psifx.io.tar.TarWriter.check")
@patch("psifx.io.video.VideoReader")
@patch("mediapipe.python.solutions.holistic.Holistic")
def test_inference(mock_holistic, mock_video_reader, mock_tar_check, 
                  mock_tar_write, mediapipe_tool, sample_video_path, 
                  sample_poses_path, mock_mediapipe_results):
    """Test the inference method."""
    # Setup mocks
    mock_holistic_instance = MagicMock()
    mock_holistic_instance.process.return_value = mock_mediapipe_results
    mock_holistic.return_value.__enter__.return_value = mock_holistic_instance
    
    # Mock video reader
    mock_video_reader_instance = MagicMock()
    mock_video_reader_instance.__iter__.return_value = [
        np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)
    ]
    mock_video_reader.return_value.__enter__.return_value = mock_video_reader_instance
    
    # Call the method
    mediapipe_tool.inference(
        video_path=sample_video_path,
        poses_path=sample_poses_path
    )
    
    # Assertions
    mock_tar_check.assert_called_once_with(path=sample_poses_path, overwrite=True)
    mock_holistic.assert_called_once_with(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        refine_face_landmarks=True
    )
    mock_video_reader.assert_called_once_with(path=sample_video_path)
    
    # Check that process was called for each frame
    assert mock_holistic_instance.process.call_count == 3
    
    # Check that tar.write was called with the correct data
    mock_tar_write.assert_called_once()
    # The first argument should be a dictionary
    args, _ = mock_tar_write.call_args
    assert isinstance(args[0], dict)
    # The dictionary should contain edges.json and 3 frame entries
    assert "edges.json" in args[0]
    assert len(args[0]) == 4  # edges.json + 3 frames


@patch("psifx.io.tar.TarWriter.write")
@patch("psifx.io.tar.TarWriter.check")
@patch("psifx.io.video.VideoReader")
@patch("psifx.io.video.VideoWriter")
@patch("mediapipe.python.solutions.holistic.Holistic")
def test_segmentation_inference(mock_holistic, mock_video_writer, mock_video_reader, 
                               mock_tar_check, mock_tar_write, mediapipe_segmentation_tool, 
                               sample_video_path, sample_poses_path, sample_masks_path, 
                               mock_mediapipe_results):
    """Test the inference method of the segmentation tool."""
    # Setup mocks
    mock_holistic_instance = MagicMock()
    mock_holistic_instance.process.return_value = mock_mediapipe_results
    mock_holistic.return_value.__enter__.return_value = mock_holistic_instance
    
    # Mock video reader and writer
    mock_video_reader_instance = MagicMock()
    mock_video_reader_instance.frame_rate = 30
    mock_video_reader_instance.__iter__.return_value = [
        np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)
    ]
    mock_video_reader.return_value.__enter__.return_value = mock_video_reader_instance
    
    mock_video_writer_instance = MagicMock()
    mock_video_writer.return_value.__enter__.return_value = mock_video_writer_instance
    
    # Call the method
    mediapipe_segmentation_tool.inference(
        video_path=sample_video_path,
        poses_path=sample_poses_path,
        masks_path=sample_masks_path
    )
    
    # Assertions
    mock_tar_check.assert_called_once_with(path=sample_poses_path, overwrite=True)
    mock_holistic.assert_called_once_with(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=True,
        smooth_segmentation=True,
        refine_face_landmarks=True
    )
    mock_video_reader.assert_called_once_with(path=sample_video_path)
    mock_video_writer.assert_called_once()
    
    # Check that process was called for each frame
    assert mock_holistic_instance.process.call_count == 3
    
    # Check that mask_writer.write was called for each frame
    assert mock_video_writer_instance.write.call_count == 3
    
    # Check that tar.write was called with the correct data
    mock_tar_write.assert_called_once()
    # The first argument should be a dictionary
    args, _ = mock_tar_write.call_args
    assert isinstance(args[0], dict)
    # The dictionary should contain edges.json and 3 frame entries
    assert "edges.json" in args[0]
    assert len(args[0]) == 4  # edges.json + 3 frames


def test_process_mask(mediapipe_segmentation_tool):
    """Test the process_mask method."""
    # Create a sample mask
    mask = np.random.random((100, 100)).astype(np.float32)
    size = (100, 100)
    
    # Test with threshold = 0.5
    result = mediapipe_segmentation_tool.process_mask(
        mask=mask,
        size=size,
        threshold=0.5
    )
    
    # Check that the result is a binary mask
    assert result.shape == (100, 100, 3)
    assert np.all(np.isin(result, [0, 255]))
    
    # Test with None mask
    result = mediapipe_segmentation_tool.process_mask(
        mask=None,
        size=size,
        threshold=0.5
    )
    
    # Check that the result is all zeros
    assert result.shape == (100, 100, 3)
    assert np.all(result == 0)