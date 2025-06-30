"""Tests for psifx.io.vtt module."""
import os
import pytest
from pathlib import Path
from psifx.io.vtt import VTTReader, VTTWriter, seconds2timestamp, timestamp2seconds

def test_seconds2timestamp():
    """Test seconds2timestamp function."""
    assert seconds2timestamp(0) == "00:00:00.000"
    assert seconds2timestamp(1.5) == "00:00:01.500"
    assert seconds2timestamp(61.25) == "00:01:01.250"
    assert seconds2timestamp(3661.75) == "01:01:01.750"

def test_timestamp2seconds():
    """Test timestamp2seconds function."""
    assert timestamp2seconds("00:00:00.000") == 0
    assert timestamp2seconds("00:00:01.500") == 1.5
    assert timestamp2seconds("00:01:01.250") == 61.25
    assert timestamp2seconds("01:01:01.750") == 3661.75

def test_vtt_reader_check():
    """Test VTTReader.check method."""
    # Test with non-vtt file
    with pytest.raises(NameError):
        VTTReader.check("file.srt")
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        VTTReader.check("nonexistent.vtt")

def test_vtt_reader_read(temp_dir):
    """Test VTTReader.read method."""
    # Create a test VTT file
    test_file = Path(temp_dir) / "test.vtt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        f.write("00:00:01.000 --> 00:00:05.000\n")
        f.write("This is the first subtitle\n\n")
        f.write("00:00:06.000 --> 00:00:10.000\n")
        f.write("<v Speaker1>This is the second subtitle with a speaker\n\n")
    
    # Test reading the file
    result = VTTReader.read(test_file)
    
    assert len(result) == 2
    
    # Check first segment
    assert result[0]["start"] == 1.0
    assert result[0]["end"] == 5.0
    assert result[0]["speaker"] is None
    assert result[0]["text"] == "This is the first subtitle"
    
    # Check second segment
    assert result[1]["start"] == 6.0
    assert result[1]["end"] == 10.0
    assert result[1]["speaker"] == "Speaker1"
    assert result[1]["text"] == "This is the second subtitle with a speaker"
    
    # Test with verbose=False
    result = VTTReader.read(test_file, verbose=False)
    assert len(result) == 2

def test_vtt_writer_check():
    """Test VTTWriter.check method."""
    # Test with non-vtt file
    with pytest.raises(NameError):
        VTTWriter.check("file.srt")
    
    # Test with existing file and overwrite=False
    test_file = "test_existing.vtt"
    try:
        # Create the file
        with open(test_file, 'w') as f:
            f.write("WEBVTT\n")
        
        # Check should raise FileExistsError
        with pytest.raises(FileExistsError):
            VTTWriter.check(test_file, overwrite=False)
        
        # Check with overwrite=True should not raise
        VTTWriter.check(test_file, overwrite=True)
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

def test_vtt_writer_write(temp_dir):
    """Test VTTWriter.write method."""
    segments = [
        {
            "start": 1.0,
            "end": 5.0,
            "text": "This is the first subtitle",
        },
        {
            "start": 6.0,
            "end": 10.0,
            "speaker": "Speaker1",
            "text": "This is the second subtitle with a speaker",
        }
    ]
    
    test_file = Path(temp_dir) / "output.vtt"
    
    # Test writing to a new file
    VTTWriter.write(segments, test_file)
    assert test_file.exists()
    
    # Test overwrite=False (should raise an error)
    with pytest.raises(FileExistsError):
        VTTWriter.write(segments, test_file, overwrite=False)
    
    # Test overwrite=True
    VTTWriter.write(segments, test_file, overwrite=True)
    
    # Test with verbose=False
    VTTWriter.write(segments, test_file, overwrite=True, verbose=False)
    
    # Test creating parent directories
    nested_file = Path(temp_dir) / "nested" / "dir" / "output.vtt"
    VTTWriter.write(segments, nested_file)
    assert nested_file.exists()
    
    # Read back the file to verify content
    result = VTTReader.read(test_file)
    assert len(result) == 2
    assert result[0]["start"] == 1.0
    assert result[0]["end"] == 5.0
    assert result[0]["text"] == "This is the first subtitle"
    assert result[1]["start"] == 6.0
    assert result[1]["end"] == 10.0
    assert result[1]["speaker"] == "Speaker1"
    assert result[1]["text"] == "This is the second subtitle with a speaker"