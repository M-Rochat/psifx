"""Tests for psifx.io.rttm module."""
import os
import pytest
import pandas as pd
from pathlib import Path
from psifx.io.rttm import RTTMReader, RTTMWriter, COLUMN_NAMES

def test_rttm_reader_check():
    """Test RTTMReader.check method."""
    # Test with non-rttm file
    with pytest.raises(NameError):
        RTTMReader.check("file.txt")
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        RTTMReader.check("nonexistent.rttm")

def test_rttm_reader_read(temp_dir):
    """Test RTTMReader.read method."""
    # Create a test RTTM file
    test_file = Path(temp_dir) / "test.rttm"
    
    # Sample RTTM content
    rttm_content = [
        "SPEAKER file1 1 0.000 1.500 <NA> <NA> speaker1 0.900 <NA>",
        "SPEAKER file1 1 1.750 2.000 <NA> <NA> speaker2 0.850 <NA>",
        "SPEAKER file1 1 4.000 1.250 <NA> <NA> speaker1 0.750 <NA>"
    ]
    
    with open(test_file, 'w') as f:
        f.write("\n".join(rttm_content))
    
    # Test reading the file
    result = RTTMReader.read(test_file)
    
    # Verify content
    assert len(result) == 3
    
    # Check first segment
    assert result[0]["type"] == "SPEAKER"
    assert result[0]["file_stem"] == "file1"
    assert result[0]["channel"] == 1
    assert result[0]["start"] == 0.0
    assert result[0]["duration"] == 1.5
    assert result[0]["speaker_name"] == "speaker1"
    assert result[0]["confidence_score"] == 0.9
    
    # Check second segment
    assert result[1]["type"] == "SPEAKER"
    assert result[1]["file_stem"] == "file1"
    assert result[1]["channel"] == 1
    assert result[1]["start"] == 1.75
    assert result[1]["duration"] == 2.0
    assert result[1]["speaker_name"] == "speaker2"
    assert result[1]["confidence_score"] == 0.85
    
    # Test with verbose=False
    result = RTTMReader.read(test_file, verbose=False)
    assert len(result) == 3

def test_rttm_writer_check():
    """Test RTTMWriter.check method."""
    # Test with non-rttm file
    with pytest.raises(NameError):
        RTTMWriter.check("file.txt")
    
    # Test with existing file and overwrite=False
    test_file = "test_existing.rttm"
    try:
        # Create the file
        with open(test_file, 'w') as f:
            f.write("SPEAKER file1 1 0.000 1.500 <NA> <NA> speaker1 0.900 <NA>")
        
        # Check should raise FileExistsError
        with pytest.raises(FileExistsError):
            RTTMWriter.check(test_file, overwrite=False)
        
        # Check with overwrite=True should not raise
        RTTMWriter.check(test_file, overwrite=True)
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

def test_rttm_writer_write(temp_dir):
    """Test RTTMWriter.write method."""
    # Sample segments
    segments = [
        {
            "type": "SPEAKER",
            "file_stem": "file1",
            "channel": 1,
            "start": 0.0,
            "duration": 1.5,
            "orthography": "<NA>",
            "speaker_type": "<NA>",
            "speaker_name": "speaker1",
            "confidence_score": 0.9,
            "signal_lookahead_time": "<NA>"
        },
        {
            "type": "SPEAKER",
            "file_stem": "file1",
            "channel": 1,
            "start": 1.75,
            "duration": 2.0,
            "orthography": "<NA>",
            "speaker_type": "<NA>",
            "speaker_name": "speaker2",
            "confidence_score": 0.85,
            "signal_lookahead_time": "<NA>"
        }
    ]
    
    test_file = Path(temp_dir) / "output.rttm"
    
    # Test writing to a new file
    RTTMWriter.write(segments, test_file)
    assert test_file.exists()
    
    # Verify the content by reading it back
    result = RTTMReader.read(test_file)
    assert len(result) == 2
    
    # Check first segment
    assert result[0]["type"] == "SPEAKER"
    assert result[0]["file_stem"] == "file1"
    assert result[0]["channel"] == 1
    assert result[0]["start"] == 0.0
    assert result[0]["duration"] == 1.5
    assert result[0]["speaker_name"] == "speaker1"
    assert result[0]["confidence_score"] == 0.9
    
    # Test overwrite=False (should raise an error)
    with pytest.raises(FileExistsError):
        RTTMWriter.write(segments, test_file, overwrite=False)
    
    # Test overwrite=True
    new_segments = [
        {
            "type": "SPEAKER",
            "file_stem": "file2",
            "channel": 1,
            "start": 0.5,
            "duration": 2.5,
            "orthography": "<NA>",
            "speaker_type": "<NA>",
            "speaker_name": "speaker3",
            "confidence_score": 0.95,
            "signal_lookahead_time": "<NA>"
        }
    ]
    RTTMWriter.write(new_segments, test_file, overwrite=True)
    
    # Verify the content was updated
    result = RTTMReader.read(test_file)
    assert len(result) == 1
    assert result[0]["file_stem"] == "file2"
    assert result[0]["speaker_name"] == "speaker3"
    
    # Test creating parent directories
    nested_file = Path(temp_dir) / "nested" / "dir" / "output.rttm"
    RTTMWriter.write(segments, nested_file)
    assert nested_file.exists()