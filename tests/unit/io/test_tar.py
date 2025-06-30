"""Tests for psifx.io.tar module."""
import os
import pytest
from pathlib import Path
from psifx.io.tar import TarReader, TarWriter

def test_tar_reader_check():
    """Test TarReader.check method."""
    # Test with non-tar file
    with pytest.raises(NameError):
        TarReader.check("file.zip")
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        TarReader.check("nonexistent.tar")
    
    # Test with .tar.gz extension (should not raise)
    test_file = "test.tar.gz"
    try:
        with open(test_file, 'w') as f:
            f.write("dummy content")
        TarReader.check(test_file)
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def test_tar_reader_read(temp_dir):
    """Test TarReader.read method."""
    import tarfile
    import io
    
    # Create a test TAR file
    test_data = {
        "file1.txt": "This is file 1 content",
        "file2.txt": "This is file 2 content",
        "file3.txt": "This is file 3 content"
    }
    
    test_file = Path(temp_dir) / "test.tar"
    
    with tarfile.open(test_file, mode="w") as tar:
        for key, value in test_data.items():
            tarinfo = tarfile.TarInfo(name=f"test/{key}")
            tarinfo.size = len(value)
            tar.addfile(tarinfo, io.BytesIO(value.encode()))
    
    # Test reading the file
    result = TarReader.read(test_file)
    
    # Verify content
    assert len(result) == 3
    for key, value in test_data.items():
        assert key in result
        assert result[key].decode() == value
    
    # Test with verbose=False
    result = TarReader.read(test_file, verbose=False)
    assert len(result) == 3

def test_tar_writer_check():
    """Test TarWriter.check method."""
    # Test with non-tar file
    with pytest.raises(NameError):
        TarWriter.check("file.zip")
    
    # Test with existing file and overwrite=False
    test_file = "test_existing.tar"
    try:
        # Create the file
        with open(test_file, 'w') as f:
            f.write("dummy content")
        
        # Check should raise FileExistsError
        with pytest.raises(FileExistsError):
            TarWriter.check(test_file, overwrite=False)
        
        # Check with overwrite=True should not raise
        TarWriter.check(test_file, overwrite=True)
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
    
    # Test with .tar.gz extension
    test_file = "test.tar.gz"
    TarWriter.check(test_file)  # Should not raise

def test_tar_writer_write(temp_dir):
    """Test TarWriter.write method."""
    test_data = {
        "file1.txt": "This is file 1 content",
        "file2.txt": "This is file 2 content",
        "file3.txt": "This is file 3 content"
    }
    
    test_file = Path(temp_dir) / "output.tar"
    
    # Test writing to a new file
    TarWriter.write(test_data, test_file)
    assert test_file.exists()
    
    # Verify the content by reading it back
    result = TarReader.read(test_file)
    assert len(result) == 3
    for key, value in test_data.items():
        assert key in result
        assert result[key].decode() == value
    
    # Test overwrite=False (should raise an error)
    with pytest.raises(FileExistsError):
        TarWriter.write({"new": "data"}, test_file, overwrite=False)
    
    # Test overwrite=True
    new_data = {
        "new_file.txt": "This is new content"
    }
    TarWriter.write(new_data, test_file, overwrite=True)
    
    # Verify the content was updated
    result = TarReader.read(test_file)
    assert len(result) == 1
    assert "new_file.txt" in result
    assert result["new_file.txt"].decode() == "This is new content"
    
    # Test with verbose=False
    TarWriter.write(test_data, test_file, overwrite=True, verbose=False)
    
    # Test creating parent directories
    nested_file = Path(temp_dir) / "nested" / "dir" / "output.tar"
    TarWriter.write(test_data, nested_file)
    assert nested_file.exists()
    
    # Test with compression (.tar.gz)
    gz_file = Path(temp_dir) / "output.tar.gz"
    TarWriter.write(test_data, gz_file)
    assert gz_file.exists()
    
    # Verify the content of the compressed file
    result = TarReader.read(gz_file)
    assert len(result) == 3
    for key, value in test_data.items():
        assert key in result
        assert result[key].decode() == value