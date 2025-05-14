import os
import pytest

@pytest.mark.structure
def test_example_readme_exists():
    assert os.path.isfile("../example/README.md"), f"Missing example readme"
