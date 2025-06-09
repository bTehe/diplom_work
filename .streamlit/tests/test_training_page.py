import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json
import base64
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages.training_page import training_page, _show_results
from utils.config import PROCESSED_FOLDER, MODEL_FOLDER, TEMP_FOLDER

class TestTrainingPage(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.st_patcher = patch('streamlit.st')
        self.mock_st = self.st_patcher.start()
        
        # Mock selectbox
        self.mock_st.selectbox.return_value = ""
        
        # Mock number inputs
        self.mock_st.number_input.side_effect = [64, 20, 0.001, 64, 5, 2, 128, 1]
        
        # Mock sliders
        self.mock_st.slider.side_effect = [0.3, 0.1]
        
        # Mock checkbox
        self.mock_st.checkbox.return_value = True
        
        # Mock button
        self.mock_st.button.return_value = False

    def tearDown(self):
        """Clean up after each test"""
        self.st_patcher.stop()

    @patch('os.listdir')
    def test_file_selection_display(self, mock_listdir):
        """Test if file selection is displayed correctly"""
        # Mock available CSV files
        mock_listdir.return_value = ["train.csv", "val.csv", "test.csv"]
        
        training_page()
        
        # Verify selectbox was called three times with correct parameters
        self.assertEqual(self.mock_st.selectbox.call_count, 3)
        calls = self.mock_st.selectbox.call_args_list
        self.assertEqual(calls[0][0][0], "Навчальний CSV")
        self.assertEqual(calls[1][0][0], "Валідаційний CSV")
        self.assertEqual(calls[2][0][0], "Тестовий CSV")

    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.remove')
    @patch('threading.Thread')
    def test_training_start(self, mock_thread, mock_remove, mock_exists, mock_listdir):
        """Test training start process"""
        # Mock file selection
        self.mock_st.selectbox.side_effect = ["train.csv", "val.csv", "test.csv"]
        self.mock_st.button.return_value = True
        
        # Mock file existence
        mock_exists.return_value = True
        
        training_page()
        
        # Verify old progress file was removed
        mock_remove.assert_called_once()
        
        # Verify training thread was started
        mock_thread.assert_called_once()
        thread_args = mock_thread.call_args[1]
        self.assertTrue(thread_args['daemon'])

    @patch('os.listdir')
    @patch('os.path.exists')
    def test_training_without_files(self, mock_exists, mock_listdir):
        """Test training attempt without selecting files"""
        # Mock empty file selection
        self.mock_st.selectbox.side_effect = ["", "", ""]
        self.mock_st.button.return_value = True
        
        training_page()
        
        # Verify error message
        self.mock_st.error.assert_called_once_with(
            "Будь ласка, виберіть файли CSV для навчання, валідації та тестування."
        )

    def test_show_results(self):
        """Test results display"""
        # Mock metrics data
        metrics = {
            'hardware': 'GPU',
            'batch_size': 64,
            'summary': '<p>Test summary</p>',
            'test_loss': 0.1234,
            'test_accuracy': 0.9876,
            'loss_plot': base64.b64encode(b'test_image').decode(),
            'accuracy_plot': base64.b64encode(b'test_image').decode(),
            'history': {
                'loss': [0.5, 0.4, 0.3],
                'val_loss': [0.6, 0.5, 0.4],
                'accuracy': [0.7, 0.8, 0.9],
                'val_accuracy': [0.65, 0.75, 0.85]
            },
            'model_path': 'test_model.h5'
        }
        
        _show_results(metrics)
        
        # Verify results were displayed
        self.mock_st.subheader.assert_called()
        self.mock_st.write.assert_called()
        self.mock_st.markdown.assert_called()
        self.mock_st.line_chart.assert_called_once()

    @patch('os.path.exists')
    def test_model_download_button(self, mock_exists):
        """Test model download button display"""
        # Mock metrics with model path
        metrics = {
            'model_path': 'test_model.h5',
            'test_loss': 0.1234,
            'test_accuracy': 0.9876
        }
        
        # Mock file existence
        mock_exists.return_value = True
        
        _show_results(metrics)
        
        # Verify download button was created
        self.mock_st.download_button.assert_called_once()

if __name__ == '__main__':
    unittest.main() 