import unittest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages.upload_page import upload_page
from utils.config import UPLOAD_FOLDER

class TestUploadPage(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Create a mock for streamlit
        self.st_patcher = patch('streamlit.st')
        self.mock_st = self.st_patcher.start()
        
        # Create a mock for file uploader
        self.mock_st.file_uploader.return_value = None

    def tearDown(self):
        """Clean up after each test"""
        self.st_patcher.stop()

    @patch('os.makedirs')
    def test_upload_folder_creation(self, mock_makedirs):
        """Test if upload folder is created"""
        upload_page()
        mock_makedirs.assert_called_once_with(UPLOAD_FOLDER, exist_ok=True)

    @patch('pathlib.Path')
    def test_file_upload_handling(self, mock_path):
        """Test file upload handling"""
        # Mock a file upload
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        mock_file.getbuffer.return_value = b"test,data\n1,2"
        self.mock_st.file_uploader.return_value = mock_file

        # Mock path operations
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__.return_value = "test_path"

        # Mock file operations
        mock_open = patch('builtins.open', MagicMock())
        mock_open.start()

        upload_page()

        # Verify file uploader was called with correct parameters
        self.mock_st.file_uploader.assert_called_once_with(
            "Оберіть CSV або PCAP файл:",
            type=["csv", "pcap"]
        )

        mock_open.stop()

    def test_no_file_uploaded(self):
        """Test behavior when no file is uploaded"""
        upload_page()
        self.mock_st.info.assert_called_once_with("Завантажте файл, щоб розпочати.")

    @patch('utils.data_loader.load_csv_preview')
    def test_csv_preview(self, mock_load_csv_preview):
        """Test CSV preview functionality"""
        # Mock a CSV file upload
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        mock_file.getbuffer.return_value = b"test,data\n1,2"
        self.mock_st.file_uploader.return_value = mock_file

        # Mock preview loading
        mock_load_csv_preview.return_value = "<table>test</table>"

        upload_page()

        # Verify preview was requested
        mock_load_csv_preview.assert_called_once()

if __name__ == '__main__':
    unittest.main() 