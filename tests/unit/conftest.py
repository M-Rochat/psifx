"""
Shared pytest fixtures for psifx unit tests.
"""
import pytest
import tempfile
import cv2
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for unit test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_video_file(tmp_path):
    """Create a temporary sample .mp4 video file for testing."""
    output_path = tmp_path / "test_video.mp4"
    frame_size = (640, 480)
    fps = 24
    num_frames = 10

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    # Write a few colored frames
    for i in range(num_frames):
        frame = np.full((frame_size[1], frame_size[0], 3), fill_value=i*20 % 255, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return output_path
