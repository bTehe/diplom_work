import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages.dashboard_page import dashboard_page, count_files

class TestDashboardPage(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.st_patcher = patch('streamlit.st')
        self.mock_st = self.st_patcher.start()
        
        # Mock columns
        self.mock_cols = [MagicMock() for _ in range(5)]
        self.mock_st.columns.return_value = self.mock_cols

    def tearDown(self):
        """Clean up after each test"""
        self.st_patcher.stop()

    def test_count_files_empty_directory(self):
        """Test counting files in empty directory"""
        with patch('pathlib.Path.exists', return_value=False):
            count = count_files(Path("nonexistent"))
            self.assertEqual(count, 0)

    def test_count_files_with_pattern(self):
        """Test counting files with specific pattern"""
        mock_files = [
            MagicMock(spec=Path),
            MagicMock(spec=Path),
            MagicMock(spec=Path)
        ]
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.glob', return_value=mock_files):
            count = count_files(Path("test"), "*.txt")
            self.assertEqual(count, 3)

    @patch('pathlib.Path.mkdir')
    def test_dashboard_page_folder_creation(self, mock_mkdir):
        """Test if all required folders are created"""
        dashboard_page()
        self.assertEqual(mock_mkdir.call_count, 5)  # Should create 5 folders

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_dashboard_metrics_display(self, mock_glob, mock_exists):
        """Test if metrics are displayed correctly"""
        # Mock file counting
        mock_exists.return_value = True
        mock_glob.return_value = [MagicMock() for _ in range(2)]  # 2 files in each directory

        dashboard_page()

        # Verify metrics were displayed
        for col in self.mock_cols:
            col.metric.assert_called_once()

    def test_dashboard_title_and_description(self):
        """Test if title and description are displayed"""
        dashboard_page()
        
        self.mock_st.title.assert_called_once_with("🛡️ Система виявлення кібер-атак")
        self.mock_st.write.assert_called_once()

    def test_dashboard_info_message(self):
        """Test if info message is displayed"""
        dashboard_page()
        
        self.mock_st.info.assert_called_once()
        info_message = self.mock_st.info.call_args[0][0]
        self.assertIn("Порада:", info_message)
        self.assertIn("Preprocessing", info_message)
        self.assertIn("Training", info_message)

if __name__ == '__main__':
    unittest.main() 