"""Tests for psifx.video.face.tool module."""
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from psifx.video.face.tool import FaceAnalysisTool


@pytest.fixture
def face_tool():
    """Create a FaceAnalysisTool instance for testing."""
    class TestFaceAnalysisTool(FaceAnalysisTool):
        """Test implementation of FaceAnalysisTool."""
        def inference(self, video_path, features_path):
            """Override abstract method for testing."""
            pass
        
        def visualization(self, video_path, features_path, visualization_path):
            """Override abstract method for testing."""
            pass
    
    return TestFaceAnalysisTool(device="cpu", overwrite=True, verbose=False)


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


def test_face_tool_init():
    """Test FaceAnalysisTool initialization."""
    class TestFaceAnalysisTool(FaceAnalysisTool):
        """Test implementation of FaceAnalysisTool."""
        def __init__(self, device="cpu", overwrite=False, verbose=True):
            super().__init__(device=device, overwrite=overwrite, verbose=verbose)
            
        def inference(self, video_path, features_path):
            """Override abstract method for testing."""
            pass
        
        def visualization(self, video_path, features_path, visualization_path):
            """Override abstract method for testing."""
            pass
    
    # Test with default parameters
    tool = TestFaceAnalysisTool()
    assert tool.device == "cpu"
    assert not tool.overwrite
    assert tool.verbose is True
    
    # Test with custom parameters
    tool = TestFaceAnalysisTool(device="cuda", overwrite=True, verbose=False)
    assert tool.device == "cuda"
    assert tool.overwrite
    assert not tool.verbose


def test_inference_abstract_method():
    """Test that the base inference method raises NotImplementedError."""
    tool = FaceAnalysisTool(device="cpu")
    with pytest.raises(NotImplementedError):
        tool.inference("video.mp4", "features.tar.xz")


def test_visualization_abstract_method():
    """Test that the base visualization method raises NotImplementedError."""
    tool = FaceAnalysisTool(device="cpu")
    with pytest.raises(NotImplementedError):
        tool.visualization("video.mp4", "features.tar.xz", "visualization.mp4")