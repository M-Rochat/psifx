"""Tests for psifx.io.wav module."""
import os
import pytest
from pathlib import Path
from psifx.io.wav import WAVReader, WAVWriter

def test_wav_reader_check():
    """Test WAVReader.check method."""
    # Test with non-wav file
    with pytest.raises(NameError):
        WAVReader.check("file.mp3")
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        WAVReader.check("nonexistent.wav")

def test_wav_writer_check():
    """Test WAVWriter.check method."""
    # Test with non-wav file
    with pytest.raises(NameError):
        WAVWriter.check("file.mp3")
    
    # Test with existing file and overwrite=False
    test_file = "test_existing.wav"
    try:
        # Create the file
        with open(test_file, 'w') as f:
            f.write("dummy content")
        
        # Check should raise FileExistsError
        with pytest.raises(FileExistsError):
            WAVWriter.check(test_file, overwrite=False)
        
        # Check with overwrite=True should not raise
        WAVWriter.check(test_file, overwrite=True)
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)