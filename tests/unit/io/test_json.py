"""Tests for psifx.io.json module."""
import json
import os
import pytest
from pathlib import Path
from psifx.io.json import JSONReader, JSONWriter

def test_json_reader_check():
    """Test JSONReader.check method."""
    # Test with non-json file
    with pytest.raises(NameError):
        JSONReader.check("file.txt")
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        JSONReader.check("nonexistent.json")

def test_json_reader_read(temp_dir):
    """Test JSONReader.read method."""
    # Create a test JSON file
    test_data = {"key": "value", "list": [1, 2, 3]}
    test_file = Path(temp_dir) / "test.json"
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    # Test reading the file
    result = JSONReader.read(test_file)
    assert result == test_data
    
    # Test with verbose=False
    result = JSONReader.read(test_file, verbose=False)
    assert result == test_data

def test_json_writer_check():
    """Test JSONWriter.check method."""
    # Test with non-json file
    with pytest.raises(NameError):
        JSONWriter.check("file.txt")

def test_json_writer_write(temp_dir):
    """Test JSONWriter.write method."""
    test_data = {"key": "value", "list": [1, 2, 3]}
    test_file = Path(temp_dir) / "output.json"
    
    # Test writing to a new file
    JSONWriter.write(test_data, test_file)
    assert test_file.exists()
    
    # Verify the content
    with open(test_file, 'r') as f:
        result = json.load(f)
    assert result == test_data
    
    # Test overwrite=False (should raise an error)
    with pytest.raises(FileExistsError):
        JSONWriter.write({"new": "data"}, test_file, overwrite=False)
    
    # Test overwrite=True
    new_data = {"new": "data"}
    JSONWriter.write(new_data, test_file, overwrite=True)
    
    # Verify the content was updated
    with open(test_file, 'r') as f:
        result = json.load(f)
    assert result == new_data
    
    # Test with verbose=False
    JSONWriter.write(test_data, test_file, overwrite=True, verbose=False)
    
    # Test creating parent directories
    nested_file = Path(temp_dir) / "nested" / "dir" / "output.json"
    JSONWriter.write(test_data, nested_file)
    assert nested_file.exists()