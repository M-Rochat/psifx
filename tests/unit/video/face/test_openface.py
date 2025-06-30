"""Tests for psifx.video.face.openface module."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

import psifx.video.face.openface.tool
from psifx.video.face.openface.tool import OpenFaceTool


@pytest.fixture
def openface_tool():
    """Create an OpenFaceTool instance for testing."""
    return OpenFaceTool(
        device="cpu",
        overwrite=True,
        verbose=False
    )


@pytest.fixture
def sample_video_path(temp_dir):
    """Return a path for a sample video file."""
    return Path(temp_dir) / "sample_video.mp4"


@pytest.fixture
def sample_features_path(temp_dir):
    """Return a path for a sample features archive."""
    return Path(temp_dir) / "sample_features.tar.xz"


@pytest.fixture
def sample_visualization_path(temp_dir):
    """Return a path for a sample visualization file."""
    return Path(temp_dir) / "sample_visualization.mp4"


@pytest.fixture
def sample_openface_output():
    """Create sample OpenFace output data for testing."""
    # Create a DataFrame with sample OpenFace output
    data = {
        'frame': [0, 1, 2],
        'timestamp': [0.0, 0.033, 0.067],
        'confidence': [0.9, 0.85, 0.95],
        'success': [1, 1, 1],
        'pose_Tx': [10.0, 11.0, 12.0],
        'pose_Ty': [20.0, 21.0, 22.0],
        'pose_Tz': [30.0, 31.0, 32.0],
        'pose_Rx': [1.0, 1.1, 1.2],
        'pose_Ry': [2.0, 2.1, 2.2],
        'pose_Rz': [3.0, 3.1, 3.2],
        'gaze_0_x': [0.1, 0.11, 0.12],
        'gaze_0_y': [0.2, 0.21, 0.22],
        'gaze_0_z': [0.3, 0.31, 0.32],
        'gaze_1_x': [0.15, 0.16, 0.17],
        'gaze_1_y': [0.25, 0.26, 0.27],
        'gaze_1_z': [0.35, 0.36, 0.37],
        'gaze_angle_x': [0.05, 0.06, 0.07],
        'gaze_angle_y': [0.02, 0.03, 0.04],
        'p_scale': [1.0, 1.1, 1.2],
        'p_rx': [0.01, 0.02, 0.03],
        'p_ry': [0.04, 0.05, 0.06],
        'p_rz': [0.07, 0.08, 0.09],
        'p_tx': [5.0, 5.1, 5.2],
        'p_ty': [6.0, 6.1, 6.2]
    }

    for i in range(34):
        data[f'p_{i}'] = [0.1 + 0.01 * i, 0.11 + 0.01 * i, 0.12 + 0.01 * i]

    for i in range(56):
        data[f'eye_lmk_x_{i}'] = [300 + i, 301 + i, 302 + i]
        data[f'eye_lmk_y_{i}'] = [400 + i, 401 + i, 402 + i]

    for i in range(56):
        data[f'eye_lmk_X_{i}'] = [500 + i, 501 + i, 502 + i]
        data[f'eye_lmk_Y_{i}'] = [600 + i, 601 + i, 602 + i]
        data[f'eye_lmk_Z_{i}'] = [700 + i, 701 + i, 702 + i]

    # Add landmark columns
    for i in range(68):
        data[f'x_{i}'] = [100 + i, 101 + i, 102 + i]
        data[f'y_{i}'] = [200 + i, 201 + i, 202 + i]

    for i in range(68):
        data[f'X_{i}'] = [800 + i, 801 + i, 802 + i]
        data[f'Y_{i}'] = [900 + i, 901 + i, 902 + i]
        data[f'Z_{i}'] = [1000 + i, 1001 + i, 1002 + i]

    for au in ['01', '02', '04', '05', '06', '07', '09', '10', '12', '14', '15', '17', '20', '23', '25', '26', '28', '45']:
        data[f'AU{au}_r'] = [0.1, 0.2, 0.3]
        data[f'AU{au}_c'] = [0, 1, 0]

    return pd.DataFrame(data)


def test_openface_tool_init():
    """Test OpenFaceTool initialization."""
    with patch("psifx.video.face.openface.tool.EXECUTABLE_PATH", "/mock/path/to/FeatureExtraction"):
        # Test with default parameters
        tool = OpenFaceTool()
        assert tool.device == "cpu"
        assert not tool.overwrite
        assert tool.verbose is True

        # Test with custom parameters
        tool = OpenFaceTool(
            device="cuda",
            overwrite=True,
            verbose=False
        )
        assert tool.device == "cuda"
        assert tool.overwrite
        assert not tool.verbose


@patch("subprocess.run")
@patch("psifx.io.tar.TarWriter.write")
@patch("psifx.io.tar.TarWriter.check")
@patch("pandas.read_csv")
def test_inference(mock_read_csv, mock_tar_check, mock_tar_write,
                  mock_subprocess_run, openface_tool, sample_video_file,
                  sample_features_path, sample_openface_output):
    """Test the inference method."""
    # Setup mocks
    mock_read_csv.return_value = sample_openface_output
    mock_subprocess_run.return_value = MagicMock(returncode=0)

    # Call the method
    with patch("psifx.video.face.openface.tool.EXECUTABLE_PATH", "/mock/path/to/FeatureExtraction"):
        openface_tool.inference(
            video_path=sample_video_file,
            features_path=sample_features_path
        )

    # Assertions
    mock_tar_check.assert_called_once_with(path=sample_features_path, overwrite=True)
    mock_subprocess_run.assert_called_once()

    # Check that subprocess.run was called with the correct command
    args, kwargs = mock_subprocess_run.call_args
    print(args, kwargs)
    args = (kwargs['args'],)
    print(args)
    assert "-f" in args[0]  # Check that the video path is in the command
    assert "-out_dir" in args[0]  # Check that the output directory is in the command

    # Check that tar.write was called
    mock_tar_write.assert_called_once()


@patch("psifx.io.tar.TarReader.read")
@patch("psifx.io.tar.TarReader.check")
@patch("psifx.io.video.VideoReader")
@patch("psifx.io.video.VideoWriter")
def test_visualization(mock_video_writer, mock_video_reader, mock_tar_check,
                      mock_tar_read, openface_tool, sample_video_path,
                      sample_features_path, sample_visualization_path):
    """Test the visualization method."""
    # Setup mocks - create features data
    features_data = {
        "metadata.json": json.dumps({
            "frame_rate": 30.0,
            "width": 640,
            "height": 480
        }),
        "000000000000000.json": json.dumps({
            "frame": 0,
            "timestamp": 0.0,
            "confidence": 0.9,
            "success": 1,
            "pose": [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
            "gaze_0": [0.1, 0.2, 0.3],
            "gaze_1": [0.15, 0.25, 0.35],
            "landmarks_2d": [[100, 200], [101, 201], [102, 202]],  # Simplified for test
            "landmarks_3d": [[100, 200, 300], [101, 201, 301], [102, 202, 302]],  # Simplified for test
            "action_units": {"01": [0.1, 0], "02": [0.2, 1]}  # Simplified for test
        }),
        "000000000000001.json": json.dumps({
            "frame": 1,
            "timestamp": 0.033,
            "confidence": 0.85,
            "success": 1,
            "pose": [11.0, 21.0, 31.0, 1.1, 2.1, 3.1],
            "gaze_0": [0.11, 0.21, 0.31],
            "gaze_1": [0.16, 0.26, 0.36],
            "landmarks_2d": [[101, 201], [102, 202], [103, 203]],  # Simplified for test
            "landmarks_3d": [[101, 201, 301], [102, 202, 302], [103, 203, 303]],  # Simplified for test
            "action_units": {"01": [0.2, 1], "02": [0.3, 0]}  # Simplified for test
        })
    }
    mock_tar_read.return_value = features_data

    # Mock video reader and writer
    mock_video_reader_instance = MagicMock()
    mock_video_reader_instance.frame_rate = 30
    mock_video_reader_instance.__iter__.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)
    ]
    mock_video_reader.return_value.__enter__.return_value = mock_video_reader_instance

    mock_video_writer_instance = MagicMock()
    mock_video_writer.return_value.__enter__.return_value = mock_video_writer_instance

    # Call the method
    openface_tool.visualization(
        video_path=sample_video_path,
        features_path=sample_features_path,
        visualization_path=sample_visualization_path
    )

    # Assertions
    mock_tar_check.assert_called_once_with(path=sample_features_path)
    mock_tar_read.assert_called_once_with(sample_features_path, verbose=False)
    mock_video_reader.assert_called_once()
    mock_video_writer.assert_called_once()

    # Check that write was called for each frame
    assert mock_video_writer_instance.write.call_count == 2


@patch("psifx.io.tar.TarReader.read")
@patch("psifx.io.tar.TarReader.check")
@patch("psifx.io.video.VideoReader")
@patch("psifx.io.video.VideoWriter")
def test_visualization_with_projection_params(mock_video_writer, mock_video_reader,
                                            mock_tar_check, mock_tar_read, openface_tool,
                                            sample_video_path, sample_features_path,
                                            sample_visualization_path):
    """Test visualization with custom projection parameters."""
    # Setup mocks - create features data
    features_data = {
        "metadata.json": json.dumps({
            "frame_rate": 30.0,
            "width": 640,
            "height": 480
        }),
        "000000000000000.json": json.dumps({
            "frame": 0,
            "timestamp": 0.0,
            "confidence": 0.9,
            "success": 1,
            "pose": [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
            "gaze_0": [0.1, 0.2, 0.3],
            "gaze_1": [0.15, 0.25, 0.35],
            "landmarks_2d": [[100, 200], [101, 201], [102, 202]],  # Simplified for test
            "landmarks_3d": [[100, 200, 300], [101, 201, 301], [102, 202, 302]],  # Simplified for test
            "action_units": {"01": [0.1, 0], "02": [0.2, 1]}  # Simplified for test
        })
    }
    mock_tar_read.return_value = features_data

    # Mock video reader and writer
    mock_video_reader_instance = MagicMock()
    mock_video_reader_instance.frame_rate = 30
    mock_video_reader_instance.__iter__.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(1)
    ]
    mock_video_reader.return_value.__enter__.return_value = mock_video_reader_instance

    mock_video_writer_instance = MagicMock()
    mock_video_writer.return_value.__enter__.return_value = mock_video_writer_instance

    # Call the method with custom projection parameters
    openface_tool.visualization(
        video_path=sample_video_path,
        features_path=sample_features_path,
        visualization_path=sample_visualization_path,
        depth=5.0,
        f_x=1200.0,
        f_y=1200.0,
        c_x=320.0,
        c_y=240.0
    )

    # Assertions
    mock_tar_check.assert_called_once_with(path=sample_features_path)
    mock_tar_read.assert_called_once_with(sample_features_path, verbose=False)
    mock_video_reader.assert_called_once()
    mock_video_writer.assert_called_once()

    # Check that write was called for each frame
    assert mock_video_writer_instance.write.call_count == 1


@patch("psifx.io.tar.TarReader.read")
@patch("psifx.io.tar.TarReader.check")
@patch("psifx.io.video.VideoReader")
@patch("psifx.io.video.VideoWriter")
def test_visualization_with_failed_frames(mock_video_writer, mock_video_reader,
                                         mock_tar_check, mock_tar_read, openface_tool,
                                         sample_video_path, sample_features_path,
                                         sample_visualization_path):
    """Test visualization with frames where face detection failed."""
    # Setup mocks - create features data with a failed frame
    features_data = {
        "metadata.json": json.dumps({
            "frame_rate": 30.0,
            "width": 640,
            "height": 480
        }),
        "000000000000000.json": json.dumps({
            "frame": 0,
            "timestamp": 0.0,
            "confidence": 0.9,
            "success": 1,
            "pose": [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
            "gaze_0": [0.1, 0.2, 0.3],
            "gaze_1": [0.15, 0.25, 0.35],
            "landmarks_2d": [[100, 200], [101, 201], [102, 202]],  # Simplified for test
            "landmarks_3d": [[100, 200, 300], [101, 201, 301], [102, 202, 302]],  # Simplified for test
            "action_units": {"01": [0.1, 0], "02": [0.2, 1]}  # Simplified for test
        }),
        "000000000000001.json": json.dumps({
            "frame": 1,
            "timestamp": 0.033,
            "confidence": 0.0,
            "success": 0,  # Failed frame
            "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "gaze_0": [0.0, 0.0, 0.0],
            "gaze_1": [0.0, 0.0, 0.0],
            "landmarks_2d": [[0, 0], [0, 0], [0, 0]],  # Simplified for test
            "landmarks_3d": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # Simplified for test
            "action_units": {"01": [0.0, 0], "02": [0.0, 0]}  # Simplified for test
        })
    }
    mock_tar_read.return_value = features_data

    # Mock video reader and writer
    mock_video_reader_instance = MagicMock()
    mock_video_reader_instance.frame_rate = 30
    mock_video_reader_instance.__iter__.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)
    ]
    mock_video_reader.return_value.__enter__.return_value = mock_video_reader_instance

    mock_video_writer_instance = MagicMock()
    mock_video_writer.return_value.__enter__.return_value = mock_video_writer_instance

    # Call the method
    openface_tool.visualization(
        video_path=sample_video_path,
        features_path=sample_features_path,
        visualization_path=sample_visualization_path
    )

    # Assertions
    mock_tar_check.assert_called_once_with(path=sample_features_path)
    mock_tar_read.assert_called_once_with(sample_features_path, verbose=False)
    mock_video_reader.assert_called_once()
    mock_video_writer.assert_called_once()

    # Check that write was called for each frame
    assert mock_video_writer_instance.write.call_count == 2