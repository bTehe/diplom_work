import pytest
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def mock_streamlit():
    """Fixture to mock Streamlit functionality"""
    import streamlit as st
    with pytest.MonkeyPatch.context() as m:
        # Mock common Streamlit functions
        m.setattr(st, 'header', lambda x: None)
        m.setattr(st, 'info', lambda x: None)
        m.setattr(st, 'error', lambda x: None)
        m.setattr(st, 'success', lambda x: None)
        m.setattr(st, 'markdown', lambda x, **kwargs: None)
        yield st

@pytest.fixture
def test_upload_folder(tmp_path):
    """Fixture to create a temporary upload folder for testing"""
    upload_folder = tmp_path / "test_uploads"
    upload_folder.mkdir()
    return str(upload_folder)

@pytest.fixture
def sample_csv_file(tmp_path):
    """Fixture to create a sample CSV file for testing"""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2\n1,2\n3,4\n5,6")
    return str(csv_file) 