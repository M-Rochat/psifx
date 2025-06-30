"""Tests for psifx.audio.manipulation.tool module."""
import os
import pytest
import tempfile
from pathlib import Path
import numpy as np
from pydub import AudioSegment
import ffmpeg
from psifx.audio.manipulation.tool import ManipulationTool


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a temporary sample .wav audio file for testing."""
    output_path = tmp_path / "test_audio.wav"
    
    # Create a simple sine wave
    sample_rate = 44100
    duration_sec = 2
    frequency = 440  # A4 note
    
    # Generate sine wave
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create AudioSegment
    audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=1  # mono
    )
    
    # Export to file
    audio.export(output_path, format="wav")
    
    return output_path


@pytest.fixture
def sample_stereo_audio_file(tmp_path):
    """Create a temporary sample stereo .wav audio file for testing."""
    output_path = tmp_path / "test_stereo_audio.wav"
    
    # Create a simple sine wave for left channel
    sample_rate = 44100
    duration_sec = 2
    frequency_left = 440  # A4 note
    frequency_right = 880  # A5 note
    
    # Generate sine waves for left and right channels
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    left_data = np.sin(2 * np.pi * frequency_left * t) * 0.5
    right_data = np.sin(2 * np.pi * frequency_right * t) * 0.5
    
    # Convert to 16-bit PCM
    left_data = (left_data * 32767).astype(np.int16)
    right_data = (right_data * 32767).astype(np.int16)
    
    # Interleave channels
    stereo_data = np.empty((left_data.size + right_data.size), dtype=np.int16)
    stereo_data[0::2] = left_data
    stereo_data[1::2] = right_data
    
    # Create AudioSegment
    audio = AudioSegment(
        stereo_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=2  # stereo
    )
    
    # Export to file
    audio.export(output_path, format="wav")
    
    return output_path


@pytest.fixture
def sample_video_file(tmp_path):
    """Create a temporary sample .mp4 video file for testing."""
    output_path = tmp_path / "test_video.mp4"
    
    # Create a simple video
    width, height = 320, 240
    framerate = 30
    duration = 2  # seconds
    
    # Create a video with a colored rectangle
    subprocess_args = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'color=c=red:s={width}x{height}:r={framerate}:d={duration}',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file if it exists
        str(output_path)
    ]
    
    import subprocess
    subprocess.run(subprocess_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    return output_path


class TestManipulationTool:
    """Test the ManipulationTool class."""
    
    def test_init(self):
        """Test initialization of ManipulationTool."""
        tool = ManipulationTool(overwrite=True, verbose=False)
        assert tool.overwrite is True
        assert tool.verbose is False
        assert tool.device == "cpu"
        
        tool = ManipulationTool()
        assert tool.overwrite is False
        assert tool.verbose is True
        assert tool.device == "cpu"
        
    def test_extraction(self, tmp_path, monkeypatch):
        """Test extraction method."""
        # Setup
        tool = ManipulationTool(overwrite=True, verbose=False)
        video_path = tmp_path / "test_video.mp4"
        audio_path = tmp_path / "extracted_audio.wav"
        
        # Create a dummy video file
        with open(video_path, 'wb') as f:
            f.write(b'dummy video content')
        
        # Create a mock for the ffmpeg chain
        class MockAudio:
            def output(self, *args, **kwargs):
                return MockOutput()
                
        class MockOutput:
            def overwrite_output(self):
                return MockOverwrite()
                
        class MockOverwrite:
            def run(self, *args, **kwargs):
                # Create a dummy audio file to simulate extraction
                with open(audio_path, 'wb') as f:
                    f.write(b'dummy audio content')
                return None
        
        class MockInput:
            @property
            def audio(self):
                return MockAudio()
        
        # Mock the ffmpeg.input function
        def mock_input(*args, **kwargs):
            return MockInput()
        
        # Apply the mock
        import ffmpeg
        monkeypatch.setattr(ffmpeg, 'input', mock_input)
        
        # Execute
        tool.extraction(
            video_path=video_path,
            audio_path=audio_path
        )
        
        # Verify
        assert audio_path.exists()
        
        # Test file exists error when overwrite=False
        tool = ManipulationTool(overwrite=False, verbose=False)
        with pytest.raises(FileExistsError):
            tool.extraction(
                video_path=video_path,
                audio_path=audio_path
            )
    
    def test_convert(self, sample_audio_file, tmp_path):
        """Test convert method."""
        # Setup
        tool = ManipulationTool(overwrite=True, verbose=False)
        mono_audio_path = tmp_path / "converted_audio.wav"
        
        # Execute
        tool.convert(
            audio_path=sample_audio_file,
            mono_audio_path=mono_audio_path
        )
        
        # Verify
        assert mono_audio_path.exists()
        
        # Load the converted audio and check properties
        converted_audio = AudioSegment.from_file(mono_audio_path)
        assert converted_audio.channels == 1  # Should be mono
        
        # Test file exists error when overwrite=False
        tool = ManipulationTool(overwrite=False, verbose=False)
        with pytest.raises(FileExistsError):
            tool.convert(
                audio_path=sample_audio_file,
                mono_audio_path=mono_audio_path
            )
    
    def test_split(self, sample_stereo_audio_file, tmp_path):
        """Test split method."""
        # Setup
        tool = ManipulationTool(overwrite=True, verbose=False)
        left_audio_path = tmp_path / "left_audio.wav"
        right_audio_path = tmp_path / "right_audio.wav"
        
        # Execute
        tool.split(
            stereo_audio_path=sample_stereo_audio_file,
            left_audio_path=left_audio_path,
            right_audio_path=right_audio_path
        )
        
        # Verify
        assert left_audio_path.exists()
        assert right_audio_path.exists()
        
        # Load the split audio files and check properties
        left_audio = AudioSegment.from_file(left_audio_path)
        right_audio = AudioSegment.from_file(right_audio_path)
        
        assert left_audio.channels == 1  # Should be mono
        assert right_audio.channels == 1  # Should be mono
        
        # Test with non-stereo input
        with pytest.raises(ValueError, match="Input audio is not stereo."):
            tool.split(
                stereo_audio_path=left_audio_path,  # This is mono, should fail
                left_audio_path=tmp_path / "new_left.wav",
                right_audio_path=tmp_path / "new_right.wav"
            )
        
        # Test file exists error when overwrite=False
        tool = ManipulationTool(overwrite=False, verbose=False)
        with pytest.raises(FileExistsError):
            tool.split(
                stereo_audio_path=sample_stereo_audio_file,
                left_audio_path=left_audio_path,
                right_audio_path=right_audio_path
            )
    
    def test_mixdown(self, sample_audio_file, tmp_path):
        """Test mixdown method."""
        # Setup
        tool = ManipulationTool(overwrite=True, verbose=False)
        
        # Create a second audio file
        second_audio_path = tmp_path / "second_audio.wav"
        original_audio = AudioSegment.from_file(sample_audio_file)
        original_audio.export(second_audio_path, format="wav")
        
        mixed_audio_path = tmp_path / "mixed_audio.wav"
        
        # Execute
        tool.mixdown(
            mono_audio_paths=[sample_audio_file, second_audio_path],
            mixed_audio_path=mixed_audio_path
        )
        
        # Verify
        assert mixed_audio_path.exists()
        
        # Load the mixed audio and check properties
        mixed_audio = AudioSegment.from_file(mixed_audio_path)
        assert mixed_audio.channels == 1  # Should be mono
        
        # Test file exists error when overwrite=False
        tool = ManipulationTool(overwrite=False, verbose=False)
        with pytest.raises(FileExistsError):
            tool.mixdown(
                mono_audio_paths=[sample_audio_file, second_audio_path],
                mixed_audio_path=mixed_audio_path
            )
        
        # Test with non-mono input
        stereo_audio_path = tmp_path / "stereo_for_mixdown.wav"
        
        # Create a stereo audio file by duplicating the mono file
        # First, create a new stereo audio file
        (
            ffmpeg
            .input(str(sample_audio_file))
            .output(str(stereo_audio_path), ac=2)  # ac=2 means 2 audio channels (stereo)
            .overwrite_output()
            .run(quiet=True)
        )
        
        with pytest.raises(AssertionError):
            tool.mixdown(
                mono_audio_paths=[sample_audio_file, stereo_audio_path],
                mixed_audio_path=tmp_path / "should_fail.wav"
            )
    
    def test_normalization(self, sample_audio_file, tmp_path):
        """Test normalization method."""
        # Setup
        tool = ManipulationTool(overwrite=True, verbose=False)
        normalized_audio_path = tmp_path / "normalized_audio.wav"
        
        # Execute
        tool.normalization(
            audio_path=sample_audio_file,
            normalized_audio_path=normalized_audio_path
        )
        
        # Verify
        assert normalized_audio_path.exists()
        
        # Load the normalized audio and check properties
        normalized_audio = AudioSegment.from_file(normalized_audio_path)
        assert abs(normalized_audio.max_dBFS) < 0.1  # Should be close to 0 dBFS
        
        # Test file exists error when overwrite=False
        tool = ManipulationTool(overwrite=False, verbose=False)
        with pytest.raises(FileExistsError):
            tool.normalization(
                audio_path=sample_audio_file,
                normalized_audio_path=normalized_audio_path
            )
    
    def test_trim(self, sample_audio_file, tmp_path):
        """Test trim method."""
        # Setup
        tool = ManipulationTool(overwrite=True, verbose=False)
        trimmed_audio_path = tmp_path / "trimmed_audio.wav"
        
        # Get original duration
        original_audio = AudioSegment.from_file(sample_audio_file)
        original_duration_sec = len(original_audio) / 1000.0
        
        # Execute - trim first half
        start_time = 0.0
        end_time = original_duration_sec / 2
        
        tool.trim(
            audio_path=sample_audio_file,
            trimmed_audio_path=trimmed_audio_path,
            start_time=start_time,
            end_time=end_time
        )
        
        # Verify
        assert trimmed_audio_path.exists()
        
        # Load the trimmed audio and check duration
        trimmed_audio = AudioSegment.from_file(trimmed_audio_path)
        trimmed_duration_sec = len(trimmed_audio) / 1000.0
        assert abs(trimmed_duration_sec - (end_time - start_time)) < 0.1
        
        # Test trim with only start_time
        trimmed_audio_path2 = tmp_path / "trimmed_audio2.wav"
        tool.trim(
            audio_path=sample_audio_file,
            trimmed_audio_path=trimmed_audio_path2,
            start_time=original_duration_sec / 2
        )
        
        # Verify
        assert trimmed_audio_path2.exists()
        trimmed_audio2 = AudioSegment.from_file(trimmed_audio_path2)
        assert abs(len(trimmed_audio2) / 1000.0 - original_duration_sec / 2) < 0.1
        
        # Test trim with only end_time
        trimmed_audio_path3 = tmp_path / "trimmed_audio3.wav"
        tool.trim(
            audio_path=sample_audio_file,
            trimmed_audio_path=trimmed_audio_path3,
            end_time=original_duration_sec / 2
        )
        
        # Verify
        assert trimmed_audio_path3.exists()
        trimmed_audio3 = AudioSegment.from_file(trimmed_audio_path3)
        assert abs(len(trimmed_audio3) / 1000.0 - original_duration_sec / 2) < 0.1
        
        # Test error cases
        with pytest.raises(ValueError, match="At least one of start_time or end_time must be specified."):
            tool.trim(
                audio_path=sample_audio_file,
                trimmed_audio_path=tmp_path / "should_fail.wav"
            )
        
        with pytest.raises(ValueError, match="Start time must be non-negative."):
            tool.trim(
                audio_path=sample_audio_file,
                trimmed_audio_path=tmp_path / "should_fail.wav",
                start_time=-1.0
            )
        
        with pytest.raises(ValueError, match="End time exceeds audio duration"):
            tool.trim(
                audio_path=sample_audio_file,
                trimmed_audio_path=tmp_path / "should_fail.wav",
                end_time=original_duration_sec + 1.0
            )
        
        with pytest.raises(ValueError, match="Start time must be less than end time."):
            tool.trim(
                audio_path=sample_audio_file,
                trimmed_audio_path=tmp_path / "should_fail.wav",
                start_time=1.0,
                end_time=0.5
            )
        
        # Test file exists error when overwrite=False
        tool = ManipulationTool(overwrite=False, verbose=False)
        with pytest.raises(FileExistsError):
            tool.trim(
                audio_path=sample_audio_file,
                trimmed_audio_path=trimmed_audio_path,
                start_time=0.5
            )