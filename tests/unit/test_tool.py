"""Tests for psifx.tool module."""
from psifx.tool import Tool

def test_tool_initialization():
    """Test Tool class initialization with different parameters."""
    # Test with default parameters
    tool = Tool(device="cpu")
    assert tool.device == "cpu"
    assert tool.overwrite is False
    assert tool.verbose is True
    
    # Test with custom parameters
    tool = Tool(device="cuda", overwrite=True, verbose=False)
    assert tool.device == "cuda"
    assert tool.overwrite is True
    assert tool.verbose is False
    
    # Test with verbose as integer
    tool = Tool(device="cpu", verbose=2)
    assert tool.device == "cpu"
    assert tool.verbose == 2