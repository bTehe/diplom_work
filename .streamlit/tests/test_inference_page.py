import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import base64
from datetime import datetime
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages.inference_page import inference_page, _show_inference_results
from utils.config import PROCESSED_FOLDER, MODEL_FOLDER, INFERENCE_FOLDER

class TestInferencePage(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.st_patcher = patch('streamlit.st')
        self.mock_st = self.st_patcher.start()
        
        # Mock selectbox
        self.mock_st.selectbox.return_value = ""
        
        # Mock checkbox
        self.mock_st.checkbox.return_value = True
        
        # Mock button
        self.mock_st.button.return_value = False
        
        # Mock columns
        self.mock_cols = [MagicMock(), MagicMock()]
        self.mock_st.columns.return_value = self.mock_cols

    def tearDown(self):
        """Clean up after each test"""
        self.st_patcher.stop()

    @patch('os.listdir')
    @patch('utils.inference.list_model_files')
    def test_file_selection_display(self, mock_list_models, mock_listdir):
        """Test if file selection is displayed correctly"""
        # Mock available files
        mock_listdir.return_value = ["test1.csv", "test2.csv"]
        mock_list_models.return_value = ["model1.h5", "model2.h5"]
        
        inference_page()
        
        # Verify selectbox was called twice with correct parameters
        self.assertEqual(self.mock_st.selectbox.call_count, 2)
        calls = self.mock_st.selectbox.call_args_list
        self.assertEqual(calls[0][0][0], "Оброблений CSV-файл")
        self.assertEqual(calls[1][0][0], "Файл моделі (.h5)")

    @patch('os.listdir')
    @patch('utils.inference.list_model_files')
    @patch('utils.inference.infer_on_file')
    def test_inference_with_selection(self, mock_infer, mock_list_models, mock_listdir):
        """Test inference with file selection"""
        # Mock file selection
        self.mock_st.selectbox.side_effect = ["test.csv", "model.h5"]
        self.mock_st.button.return_value = True
        
        # Mock inference result
        mock_result = {
            'true_labels': [0, 1, 0],
            'predictions': [0, 1, 1],
            'prediction_probs': [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]],
            'metrics': {
                'accuracy': 0.67,
                'precision': 0.75,
                'recall': 0.67
            },
            'model_used': 'model.h5',
            'hardware': 'GPU'
        }
        mock_infer.return_value = mock_result
        
        inference_page()
        
        # Verify inference was called with correct parameters
        mock_infer.assert_called_once_with(
            test_file="test.csv",
            processed_dir=PROCESSED_FOLDER,
            model_path=os.path.join(MODEL_FOLDER, "model.h5"),
            use_gpu=True
        )

    @patch('os.listdir')
    @patch('utils.inference.list_model_files')
    def test_inference_without_selection(self, mock_list_models, mock_listdir):
        """Test inference attempt without selecting files"""
        # Mock empty file selection
        self.mock_st.selectbox.side_effect = ["", ""]
        self.mock_st.button.return_value = True
        
        inference_page()
        
        # Verify error message
        self.mock_st.error.assert_called_once_with(
            "Будь ласка, виберіть і CSV-файл, і файл моделі."
        )

    @patch('os.listdir')
    @patch('utils.inference.list_model_files')
    @patch('utils.inference.infer_on_file')
    def test_inference_error_handling(self, mock_infer, mock_list_models, mock_listdir):
        """Test error handling during inference"""
        # Mock file selection
        self.mock_st.selectbox.side_effect = ["test.csv", "model.h5"]
        self.mock_st.button.return_value = True
        
        # Mock inference error
        mock_infer.side_effect = Exception("Test error")
        
        inference_page()
        
        # Verify error message
        self.mock_st.error.assert_called()
        error_message = self.mock_st.error.call_args[0][0]
        self.assertIn("Не вдалося виконати інференс", error_message)

    def test_show_inference_results(self):
        """Test display of inference results"""
        # Mock inference result
        result = {
            'model_used': 'test_model.h5',
            'hardware': 'GPU',
            'true_labels': [0, 1, 0, 1],
            'predictions': [0, 1, 1, 1],
            'metrics': {
                'accuracy': 0.75,
                'precision': 0.8,
                'recall': 0.75,
                'f1': 0.77,
                'weighted_f1': 0.76,
                'top_3_accuracy': 0.9,
                'top_5_accuracy': 0.95,
                'log_loss': 0.3,
                'roc_auc_micro': 0.85,
                'roc_auc_macro': 0.84,
                'pr_auc_micro': 0.83,
                'pr_auc_macro': 0.82
            },
            'confusion_matrix_plot': base64.b64encode(b'test_image').decode(),
            'roc_plot': base64.b64encode(b'test_image').decode(),
            'pr_plot': base64.b64encode(b'test_image').decode(),
            'calibration_plot': base64.b64encode(b'test_image').decode()
        }
        
        _show_inference_results(result)
        
        # Verify results were displayed
        self.mock_st.success.assert_called_once_with("✅ Інференс завершено")
        self.mock_st.subheader.assert_called()
        self.mock_st.write.assert_called()
        self.mock_st.table.assert_called()

    def test_show_inference_results_with_errors(self):
        """Test display of inference results with classification errors"""
        # Mock inference result with errors
        result = {
            'true_labels': [0, 1, 0, 1],
            'predictions': [1, 1, 0, 0],  # Two errors
            'metrics': {'accuracy': 0.5}
        }
        
        _show_inference_results(result)
        
        # Verify error table was displayed
        self.mock_st.table.assert_called()
        table_data = self.mock_st.table.call_args[0][0]
        self.assertIsInstance(table_data, pd.DataFrame)
        self.assertEqual(len(table_data), 2)  # Two errors

    def test_show_inference_results_no_errors(self):
        """Test display of inference results with no classification errors"""
        # Mock inference result with no errors
        result = {
            'true_labels': [0, 1, 0, 1],
            'predictions': [0, 1, 0, 1],  # No errors
            'metrics': {'accuracy': 1.0}
        }
        
        _show_inference_results(result)
        
        # Verify no error table was displayed
        self.mock_st.write.assert_called_with("Помилок класифікації не знайдено!")

if __name__ == '__main__':
    unittest.main() 