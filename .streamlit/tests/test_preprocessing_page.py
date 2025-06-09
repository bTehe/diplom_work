import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages.preprocessing_page import preprocessing_page
from utils.config import UPLOAD_FOLDER, PROCESSED_FOLDER

class TestPreprocessingPage(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.st_patcher = patch('streamlit.st')
        self.mock_st = self.st_patcher.start()
        
        # Mock multiselect
        self.mock_st.multiselect.return_value = []
        
        # Mock button
        self.mock_st.button.return_value = False

    def tearDown(self):
        """Clean up after each test"""
        self.st_patcher.stop()

    @patch('utils.preprocessing.list_raw_files')
    def test_file_selection_display(self, mock_list_files):
        """Test if file selection is displayed correctly"""
        # Mock available files
        mock_list_files.return_value = ["file1.csv", "file2.csv"]
        
        preprocessing_page()
        
        # Verify multiselect was called with correct parameters
        self.mock_st.multiselect.assert_called_once_with(
            "Виберіть один або кілька RAW CSV-файлів",
            options=["file1.csv", "file2.csv"],
            help="Використовуйте Ctrl/Cmd для множинного вибору"
        )

    @patch('utils.preprocessing.list_raw_files')
    @patch('utils.preprocessing.preprocess_files_combined')
    def test_preprocessing_with_no_selection(self, mock_preprocess, mock_list_files):
        """Test preprocessing with no files selected"""
        # Mock button press
        self.mock_st.button.return_value = True
        
        preprocessing_page()
        
        # Verify error message
        self.mock_st.error.assert_called_once_with(
            "Будь ласка, виберіть принаймні один файл для обробки."
        )
        # Verify preprocessing was not called
        mock_preprocess.assert_not_called()

    @patch('utils.preprocessing.list_raw_files')
    @patch('utils.preprocessing.preprocess_files_combined')
    def test_preprocessing_with_selection(self, mock_preprocess, mock_list_files):
        """Test preprocessing with files selected"""
        # Mock file selection
        self.mock_st.multiselect.return_value = ["file1.csv"]
        self.mock_st.button.return_value = True
        
        # Mock preprocessing result
        mock_summary = MagicMock()
        mock_summary.filename = "file1.csv"
        mock_summary.initial_rows = 100
        mock_summary.after_clean_rows = 95
        mock_summary.missing_before = 5
        mock_summary.negatives_before = 2
        mock_summary.processed_path = "processed/file1.csv"
        mock_summary.df_head = "<table>test</table>"
        mock_preprocess.return_value = [mock_summary]
        
        preprocessing_page()
        
        # Verify preprocessing was called with correct parameters
        mock_preprocess.assert_called_once_with(
            filenames=["file1.csv"],
            raw_dir=UPLOAD_FOLDER,
            processed_dir=PROCESSED_FOLDER
        )
        
        # Verify success message
        self.mock_st.success.assert_called_once_with("✅ Попередня обробка завершена")

    @patch('utils.preprocessing.list_raw_files')
    @patch('utils.preprocessing.preprocess_files_combined')
    def test_preprocessing_error_handling(self, mock_preprocess, mock_list_files):
        """Test error handling during preprocessing"""
        # Mock file selection
        self.mock_st.multiselect.return_value = ["file1.csv"]
        self.mock_st.button.return_value = True
        
        # Mock preprocessing error
        mock_preprocess.side_effect = Exception("Test error")
        
        preprocessing_page()
        
        # Verify error message
        self.mock_st.error.assert_called_once()
        error_message = self.mock_st.error.call_args[0][0]
        self.assertIn("Помилка під час попередньої обробки", error_message)

    @patch('utils.preprocessing.list_raw_files')
    @patch('utils.preprocessing.preprocess_files_combined')
    def test_preprocessing_results_display(self, mock_preprocess, mock_list_files):
        """Test display of preprocessing results"""
        # Mock file selection and preprocessing
        self.mock_st.multiselect.return_value = ["file1.csv"]
        self.mock_st.button.return_value = True
        
        # Mock preprocessing result
        mock_summary = MagicMock()
        mock_summary.filename = "file1.csv"
        mock_summary.initial_rows = 100
        mock_summary.after_clean_rows = 95
        mock_summary.missing_before = 5
        mock_summary.negatives_before = 2
        mock_summary.processed_path = "processed/file1.csv"
        mock_summary.df_head = "<table>test</table>"
        mock_preprocess.return_value = [mock_summary]
        
        preprocessing_page()
        
        # Verify results were displayed
        self.mock_st.subheader.assert_called()
        self.mock_st.markdown.assert_called()

if __name__ == '__main__':
    unittest.main() 