"""Tests for psifx.io.video module."""
import os
import pytest
import numpy as np
from pathlib import Path


from psifx.io.video import VideoReader, VideoWriter

@pytest.fixture
def sample_video_file(temp_dir):
    """Create a sample video file for testing."""
    try:
        # Check if we have ffmpeg installed
        import skvideo.io
        
        # Create a small test video file
        video_path = Path(temp_dir) / "test_video.mp4"
        
        # Create a simple video with 10 frames
        frames = []
        for i in range(10):
            # Create a colored frame (100x100 pixels)
            frame = np.ones((100, 100, 3), dtype=np.uint8) * (25 * i)
            frames.append(frame)
        
        # Write the video
        writer = VideoWriter(video_path)
        for frame in frames:
            writer.write(frame)
        writer.close()
        
        return video_path
    except (ImportError, Exception) as e:
        pytest.skip(f"Skipping video tests: {str(e)}")

def test_video_reader_init(sample_video_file):
    """Test VideoReader initialization."""
    # Test with existing file
    reader = VideoReader(sample_video_file)
    assert reader.num_frames > 0
    assert reader.frame_rate is not None
    reader.close()
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        VideoReader("nonexistent_video.mp4")

def test_video_reader_len(sample_video_file):
    """Test VideoReader __len__ method."""
    reader = VideoReader(sample_video_file)
    assert len(reader) > 0
    reader.close()

def test_video_writer_init(temp_dir, sample_video_file):
    """Test VideoWriter methods."""
    # Read frames from sample video

    with VideoReader(sample_video_file) as reader:
        frames = [frame for frame in reader]

    # Test with new file
    video_path = Path(temp_dir) / "output_video.mp4"
    with VideoWriter(video_path) as writer:
        for frame in frames:
            writer.write(frame)

    assert video_path.exists()


    # Test with existing file and overwrite=False
    with pytest.raises(FileExistsError):
        with VideoWriter(video_path, overwrite=False):
            pass
    
    # Test with existing file and overwrite=True
    with VideoWriter(video_path, overwrite=True) as writer:
        for frame in frames[:2]:
            writer.write(frame)

    # Assert that the file has been overwritten with two frames
    assert video_path.exists()
    with VideoReader(video_path) as reader:
        assert len(reader) == 2

    # Test creating parent directories
    nested_path = Path(temp_dir) / "nested" / "dir" / "output_video.mp4"
    with VideoWriter(nested_path) as writer:
        for frame in frames:
            writer.write(frame)
    assert nested_path.exists()