"""Tests for psifx.io.csv module."""
import os
import pytest
import pandas as pd
from pathlib import Path
from psifx.io.csv import CsvReader, CsvWriter

def test_csv_reader_check():
    """Test CsvReader.check method."""
    # Test with non-csv file
    with pytest.raises(NameError):
        CsvReader.check("file.txt")
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        CsvReader.check("nonexistent.csv")

def test_csv_reader_read(temp_dir):
    """Test CsvReader.read method."""
    # Create a test CSV file
    test_data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['New York', 'London', 'Paris']
    })
    test_file = Path(temp_dir) / "test.csv"
    
    test_data.to_csv(test_file, index=False)
    
    # Test reading the file
    result = CsvReader.read(test_file)
    pd.testing.assert_frame_equal(result, test_data)
    
    # Test with verbose=False
    result = CsvReader.read(test_file, verbose=False)
    pd.testing.assert_frame_equal(result, test_data)

def test_csv_writer_check():
    """Test CsvWriter.check method."""
    # Test with non-csv file
    with pytest.raises(NameError):
        CsvWriter.check("file.txt")
    
    # Test with existing file and overwrite=False
    test_file = "test_existing.csv"
    try:
        # Create the file
        with open(test_file, 'w') as f:
            f.write("dummy,data\n1,2\n")
        
        # Check should raise FileExistsError
        with pytest.raises(FileExistsError):
            CsvWriter.check(test_file, overwrite=False)
        
        # Check with overwrite=True should not raise
        CsvWriter.check(test_file, overwrite=True)
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

def test_csv_writer_write(temp_dir):
    """Test CsvWriter.write method."""
    test_data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['New York', 'London', 'Paris']
    })
    test_file = Path(temp_dir) / "output.csv"
    
    # Test writing to a new file
    CsvWriter.write(test_data, test_file)
    assert test_file.exists()
    
    # Verify the content
    result = pd.read_csv(test_file)
    pd.testing.assert_frame_equal(result, test_data)
    
    # Test overwrite=False (should raise an error)
    with pytest.raises(FileExistsError):
        CsvWriter.write(pd.DataFrame({"new": ["data"]}), test_file, overwrite=False)
    
    # Test overwrite=True
    new_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    CsvWriter.write(new_data, test_file, overwrite=True)
    
    # Verify the content was updated
    result = pd.read_csv(test_file)
    pd.testing.assert_frame_equal(result, new_data)
    
    # Test creating parent directories
    nested_file = Path(temp_dir) / "nested" / "dir" / "output.csv"
    CsvWriter.write(test_data, nested_file)
    assert nested_file.exists()
    result = pd.read_csv(nested_file)
    pd.testing.assert_frame_equal(result, test_data)