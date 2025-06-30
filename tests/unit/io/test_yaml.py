"""Tests for psifx.io.yaml module."""
import os
import yaml
import pytest
from pathlib import Path
from psifx.io.yaml import YAMLReader, YAMLWriter

def test_yaml_reader_check():
    """Test YAMLReader.check method."""
    # Test with non-yaml file
    with pytest.raises(NameError):
        YAMLReader.check("file.txt")
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        YAMLReader.check("nonexistent.yaml")
    
    # Test with .yml extension (should not raise)
    test_file = "test.yml"
    try:
        with open(test_file, 'w') as f:
            f.write("key: value\n")
        YAMLReader.check(test_file)
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def test_yaml_reader_read(temp_dir):
    """Test YAMLReader.read method."""
    # Create a test YAML file
    test_data = {
        "name": "Test Project",
        "version": "1.0.0",
        "dependencies": ["dep1", "dep2", "dep3"],
        "config": {
            "debug": True,
            "timeout": 30
        }
    }
    test_file = Path(temp_dir) / "test.yaml"
    
    with open(test_file, 'w') as f:
        yaml.safe_dump(test_data, f)
    
    # Test reading the file
    result = YAMLReader.read(test_file)
    assert result == test_data
    
    # Test with verbose=False
    result = YAMLReader.read(test_file, verbose=False)
    assert result == test_data
    
    # Test with .yml extension
    yml_file = Path(temp_dir) / "test.yml"
    with open(yml_file, 'w') as f:
        yaml.safe_dump(test_data, f)
    
    result = YAMLReader.read(yml_file)
    assert result == test_data

def test_yaml_writer_check():
    """Test YAMLWriter.check method."""
    # Test with non-yaml file
    with pytest.raises(NameError):
        YAMLWriter.check("file.txt")
    
    # Test with existing file and overwrite=False
    test_file = "test_existing.yaml"
    try:
        # Create the file
        with open(test_file, 'w') as f:
            f.write("key: value\n")
        
        # Check should raise FileExistsError
        with pytest.raises(FileExistsError):
            YAMLWriter.check(test_file, overwrite=False)
        
        # Check with overwrite=True should not raise
        YAMLWriter.check(test_file, overwrite=True)
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
    
    # Test with .yml extension
    test_file = "test.yml"
    YAMLWriter.check(test_file)  # Should not raise
    
def test_yaml_writer_write(temp_dir):
    """Test YAMLWriter.write method."""
    test_data = {
        "name": "Test Project",
        "version": "1.0.0",
        "dependencies": ["dep1", "dep2", "dep3"],
        "config": {
            "debug": True,
            "timeout": 30
        }
    }
    test_file = Path(temp_dir) / "output.yaml"
    
    # Test writing to a new file
    YAMLWriter.write(test_data, test_file)
    assert test_file.exists()
    
    # Verify the content
    with open(test_file, 'r') as f:
        result = yaml.safe_load(f)
    assert result == test_data
    
    # Test overwrite=False (should raise an error)
    with pytest.raises(FileExistsError):
        YAMLWriter.write({"new": "data"}, test_file, overwrite=False)
    
    # Test overwrite=True
    new_data = {"new": "data", "list": [1, 2, 3]}
    YAMLWriter.write(new_data, test_file, overwrite=True)
    
    # Verify the content was updated
    with open(test_file, 'r') as f:
        result = yaml.safe_load(f)
    assert result == new_data
    
    # Test with verbose=False
    YAMLWriter.write(test_data, test_file, overwrite=True, verbose=False)
    
    # Test creating parent directories
    nested_file = Path(temp_dir) / "nested" / "dir" / "output.yaml"
    YAMLWriter.write(test_data, nested_file)
    assert nested_file.exists()
    
    # Test with .yml extension
    yml_file = Path(temp_dir) / "test.yml"
    YAMLWriter.write(test_data, yml_file)
    assert yml_file.exists()
    with open(yml_file, 'r') as f:
        result = yaml.safe_load(f)
    assert result == test_data