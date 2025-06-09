import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages.reports_page import reports_page
from utils.config import INFERENCE_FOLDER, REPORT_FOLDER

class TestReportsPage(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.st_patcher = patch('streamlit.st')
        self.mock_st = self.st_patcher.start()
        
        # Mock form
        self.mock_form = MagicMock()
        self.mock_st.form.return_value.__enter__.return_value = self.mock_form
        
        # Mock selectbox
        self.mock_form.selectbox.return_value = ""
        
        # Mock form submit button
        self.mock_form.form_submit_button.return_value = False

    def tearDown(self):
        """Clean up after each test"""
        self.st_patcher.stop()

    @patch('utils.reports.list_inference_results')
    def test_no_inference_results(self, mock_list_results):
        """Test behavior when no inference results are found"""
        mock_list_results.return_value = []
        
        reports_page()
        
        self.mock_st.info.assert_called_once_with(
            "Файли результатів inference не знайдено."
        )

    @patch('utils.reports.list_inference_results')
    @patch('utils.reports.list_reports')
    def test_report_generation_form(self, mock_list_reports, mock_list_results):
        """Test report generation form display"""
        # Mock available results and reports
        mock_list_results.return_value = ["result1.csv", "result2.csv"]
        mock_list_reports.return_value = {'csv': [], 'pdf': []}
        
        reports_page()
        
        # Verify form was created
        self.mock_st.form.assert_called_once_with("generate_report")
        
        # Verify selectboxes were created
        self.assertEqual(self.mock_form.selectbox.call_count, 2)
        calls = self.mock_form.selectbox.call_args_list
        self.assertEqual(calls[0][0][0], "Виберіть файл результатів inference:")
        self.assertEqual(calls[1][0][0], "Тип звіту:")

    @patch('utils.reports.list_inference_results')
    @patch('utils.reports.list_reports')
    @patch('utils.reports.save_csv_report')
    def test_csv_report_generation(self, mock_save_csv, mock_list_reports, mock_list_results):
        """Test CSV report generation"""
        # Mock available results and reports
        mock_list_results.return_value = ["result1.csv"]
        mock_list_reports.return_value = {'csv': [], 'pdf': []}
        
        # Mock form submission
        self.mock_form.selectbox.side_effect = ["result1.csv", "CSV"]
        self.mock_form.form_submit_button.return_value = True
        
        # Mock report saving
        mock_save_csv.return_value = "report.csv"
        
        reports_page()
        
        # Verify report was saved
        mock_save_csv.assert_called_once_with(
            INFERENCE_FOLDER,
            REPORT_FOLDER,
            "result1.csv"
        )
        
        # Verify success message
        self.mock_st.success.assert_called_once_with(
            'Звіт "report.csv" успішно створено.'
        )

    @patch('utils.reports.list_inference_results')
    @patch('utils.reports.list_reports')
    @patch('utils.reports.save_pdf_report')
    def test_pdf_report_generation(self, mock_save_pdf, mock_list_reports, mock_list_results):
        """Test PDF report generation"""
        # Mock available results and reports
        mock_list_results.return_value = ["result1.csv"]
        mock_list_reports.return_value = {'csv': [], 'pdf': []}
        
        # Mock form submission
        self.mock_form.selectbox.side_effect = ["result1.csv", "PDF"]
        self.mock_form.form_submit_button.return_value = True
        
        # Mock report saving
        mock_save_pdf.return_value = "report.pdf"
        
        reports_page()
        
        # Verify report was saved
        mock_save_pdf.assert_called_once_with(
            INFERENCE_FOLDER,
            REPORT_FOLDER,
            "result1.csv"
        )
        
        # Verify success message
        self.mock_st.success.assert_called_once_with(
            'Звіт "report.pdf" успішно створено.'
        )

    @patch('utils.reports.list_inference_results')
    @patch('utils.reports.list_reports')
    @patch('utils.reports.save_csv_report')
    def test_report_generation_error(self, mock_save_csv, mock_list_reports, mock_list_results):
        """Test error handling during report generation"""
        # Mock available results and reports
        mock_list_results.return_value = ["result1.csv"]
        mock_list_reports.return_value = {'csv': [], 'pdf': []}
        
        # Mock form submission
        self.mock_form.selectbox.side_effect = ["result1.csv", "CSV"]
        self.mock_form.form_submit_button.return_value = True
        
        # Mock report saving error
        mock_save_csv.side_effect = Exception("Test error")
        
        reports_page()
        
        # Verify error message
        self.mock_st.error.assert_called_once()
        error_message = self.mock_st.error.call_args[0][0]
        self.assertIn("Помилка при створенні звіту", error_message)

    @patch('utils.reports.list_inference_results')
    @patch('utils.reports.list_reports')
    def test_report_download_buttons(self, mock_list_reports, mock_list_results):
        """Test report download buttons display"""
        # Mock available results and reports
        mock_list_results.return_value = ["result1.csv"]
        mock_list_reports.return_value = {
            'csv': ['report1.csv', 'report2.csv'],
            'pdf': ['report1.pdf']
        }
        
        # Mock file reading
        with patch('builtins.open', MagicMock()) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"test data"
            
            reports_page()
            
            # Verify download buttons were created
            self.mock_st.download_button.assert_called()
            calls = self.mock_st.download_button.call_args_list
            
            # Verify CSV download buttons
            csv_calls = [call for call in calls if call[1]['mime'] == 'text/csv']
            self.assertEqual(len(csv_calls), 2)
            
            # Verify PDF download buttons
            pdf_calls = [call for call in calls if call[1]['mime'] == 'application/pdf']
            self.assertEqual(len(pdf_calls), 1)

    @patch('utils.reports.list_inference_results')
    @patch('utils.reports.list_reports')
    def test_no_reports_available(self, mock_list_reports, mock_list_results):
        """Test display when no reports are available"""
        # Mock available results but no reports
        mock_list_results.return_value = ["result1.csv"]
        mock_list_reports.return_value = {'csv': [], 'pdf': []}
        
        reports_page()
        
        # Verify "no reports" messages
        self.mock_st.write.assert_called_with("CSV-звітів немає.")
        self.mock_st.write.assert_called_with("PDF-звітів немає.")

if __name__ == '__main__':
    unittest.main() 