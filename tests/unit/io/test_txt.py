"""Tests for psifx.io.txt module."""
import os
import pytest
from pathlib import Path
from psifx.io.txt import TxtReader, TxtWriter

def test_txt_reader_check():
    """Test TxtReader.check method."""
    # Test with non-txt file
    with pytest.raises(NameError):
        TxtReader.check("file.csv")
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        TxtReader.check("nonexistent.txt")

def test_txt_reader_read(temp_dir):
    """Test TxtReader.read method."""
    # Create a test TXT file
    test_content = "This is a test file.\nIt has multiple lines.\nThis is the third line."
    test_file = Path(temp_dir) / "test.txt"
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    # Test reading the file
    result = TxtReader.read(test_file)
    assert result == test_content
    
    # Test with verbose=False
    result = TxtReader.read(test_file, verbose=False)
    assert result == test_content

def test_txt_writer_check():
    """Test TxtWriter.check method."""
    # Test with non-txt file
    with pytest.raises(NameError):
        TxtWriter.check("file.csv")
    
    # Test with existing file and overwrite=False
    test_file = "test_existing.txt"
    try:
        # Create the file
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        # Check should raise FileExistsError
        with pytest.raises(FileExistsError):
            TxtWriter.check(test_file, overwrite=False)
        
        # Check with overwrite=True should not raise
        TxtWriter.check(test_file, overwrite=True)
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

def test_txt_writer_write(temp_dir):
    """Test TxtWriter.write method."""
    test_content = "This is a test file.\nIt has multiple lines.\nThis is the third line."
    test_file = Path(temp_dir) / "output.txt"
    
    # Test writing to a new file
    TxtWriter.write(test_content, test_file)
    assert test_file.exists()
    
    # Verify the content
    with open(test_file, 'r') as f:
        result = f.read()
    assert result == test_content
    
    # Test overwrite=False (should raise an error)
    with pytest.raises(FileExistsError):
        TxtWriter.write("New content", test_file, overwrite=False)
    
    # Test overwrite=True
    new_content = "This is new content."
    TxtWriter.write(new_content, test_file, overwrite=True)
    
    # Verify the content was updated
    with open(test_file, 'r') as f:
        result = f.read()
    assert result == new_content
    
    # Test creating parent directories
    nested_file = Path(temp_dir) / "nested" / "dir" / "output.txt"
    TxtWriter.write(test_content, nested_file)
    assert nested_file.exists()
    with open(nested_file, 'r') as f:
        result = f.read()
    assert result == test_content