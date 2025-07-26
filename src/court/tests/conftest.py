"""Pytest configuration and shared fixtures for court detection tests."""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add src to Python path 
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def project_root_path():
    """Provide project root path."""
    return project_root


@pytest.fixture(scope="session") 
def court_module_path():
    """Provide court module path."""
    return project_root / "src" / "court"


def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line("markers", "validation: validation tests")
    config.addinivalue_line("markers", "unit: unit tests") 
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "slow: slow running tests")
    config.addinivalue_line("markers", "gpu: tests requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify collected test items."""
    # Add markers based on directory structure
    for item in items:
        if "validation" in str(item.fspath):
            item.add_marker(pytest.mark.validation)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)