import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages.monitoring_page import monitoring_page
from utils.config import INFERENCE_FOLDER

class TestMonitoringPage(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.st_patcher = patch('streamlit.st')
        self.mock_st = self.st_patcher.start()
        
        # Mock tabs
        self.mock_tabs = [MagicMock(), MagicMock(), MagicMock()]
        self.mock_st.tabs.return_value = self.mock_tabs
        
        # Mock columns
        self.mock_cols = [MagicMock(), MagicMock()]
        self.mock_st.columns.return_value = self.mock_cols
        
        # Mock selectbox
        self.mock_st.selectbox.return_value = ""
        
        # Mock slider
        self.mock_st.slider.return_value = 0.0
        
        # Mock button
        self.mock_st.button.return_value = False

    def tearDown(self):
        """Clean up after each test"""
        self.st_patcher.stop()

    @patch('utils.monitoring.list_incident_files')
    def test_no_incident_files(self, mock_list_files):
        """Test behavior when no incident files are found"""
        mock_list_files.return_value = []
        
        monitoring_page()
        
        self.mock_st.info.assert_called_once_with(
            "У каталозі інференсу не знайдено файлів інцидентів."
        )

    @patch('utils.monitoring.list_incident_files')
    @patch('utils.monitoring.load_incidents')
    def test_incident_file_selection(self, mock_load, mock_list_files):
        """Test incident file selection"""
        # Mock available files
        mock_list_files.return_value = ["incident1.csv", "incident2.csv"]
        
        # Mock incident data
        mock_df = pd.DataFrame({
            'pred': [0, 1, 0, 1],
            'confidence': [0.9, 0.8, 0.7, 0.6],
            'true': [0, 1, 0, 1]
        })
        mock_load.return_value = mock_df
        
        monitoring_page()
        
        # Verify file selection was displayed
        self.mock_st.selectbox.assert_called_with(
            "Оберіть файл інцидентів:",
            options=[""] + ["incident1.csv", "incident2.csv"],
            index=0
        )

    @patch('utils.monitoring.list_incident_files')
    @patch('utils.monitoring.load_incidents')
    def test_incident_filtering(self, mock_load, mock_list_files):
        """Test incident filtering functionality"""
        # Mock file selection
        mock_list_files.return_value = ["incident1.csv"]
        self.mock_st.selectbox.side_effect = ["incident1.csv", "Усі"]
        self.mock_st.button.return_value = True
        
        # Mock incident data
        mock_df = pd.DataFrame({
            'pred': [0, 1, 0, 1],
            'confidence': [0.9, 0.8, 0.7, 0.6],
            'true': [0, 1, 0, 1]
        })
        mock_load.return_value = mock_df
        
        monitoring_page()
        
        # Verify filtering controls were displayed
        self.mock_st.slider.assert_called()

    @patch('utils.monitoring.list_incident_files')
    @patch('utils.monitoring.load_incidents')
    @patch('utils.monitoring.get_incident_statistics')
    def test_statistics_display(self, mock_stats, mock_load, mock_list_files):
        """Test statistics display"""
        # Mock file selection
        mock_list_files.return_value = ["incident1.csv"]
        self.mock_st.selectbox.return_value = "incident1.csv"
        
        # Mock incident data
        mock_df = pd.DataFrame({
            'pred': [0, 1, 0, 1],
            'confidence': [0.9, 0.8, 0.7, 0.6],
            'true': [0, 1, 0, 1]
        })
        mock_load.return_value = mock_df
        
        # Mock statistics
        mock_stats.return_value = {
            'total_incidents': 4,
            'avg_confidence': 0.75,
            'min_confidence': 0.6,
            'max_confidence': 0.9,
            'class_distribution': {0: 2, 1: 2}
        }
        
        monitoring_page()
        
        # Verify statistics were displayed
        self.mock_st.metric.assert_called()
        self.mock_st.bar_chart.assert_called_once()

    @patch('utils.monitoring.list_incident_files')
    @patch('utils.monitoring.load_incidents')
    @patch('utils.monitoring.get_anomaly_incidents')
    def test_anomaly_detection(self, mock_anomalies, mock_load, mock_list_files):
        """Test anomaly detection functionality"""
        # Mock file selection
        mock_list_files.return_value = ["incident1.csv"]
        self.mock_st.selectbox.return_value = "incident1.csv"
        
        # Mock incident data
        mock_df = pd.DataFrame({
            'pred': [0, 1, 0, 1],
            'confidence': [0.9, 0.8, 0.7, 0.6],
            'true': [0, 1, 0, 1]
        })
        mock_load.return_value = mock_df
        
        # Mock anomalies
        mock_anomalies.return_value = pd.DataFrame({
            'pred': [0, 1],
            'confidence': [0.3, 0.4],
            'true': [1, 0]
        })
        
        monitoring_page()
        
        # Verify anomaly detection controls were displayed
        self.mock_st.slider.assert_called()
        self.mock_st.dataframe.assert_called()

    @patch('utils.monitoring.list_incident_files')
    @patch('utils.monitoring.load_incidents')
    @patch('utils.monitoring.export_incidents')
    def test_incident_export(self, mock_export, mock_load, mock_list_files):
        """Test incident export functionality"""
        # Mock file selection
        mock_list_files.return_value = ["incident1.csv"]
        self.mock_st.selectbox.side_effect = ["incident1.csv", "csv"]
        self.mock_st.button.side_effect = [True, True]  # Filter and Export buttons
        
        # Mock incident data
        mock_df = pd.DataFrame({
            'pred': [0, 1, 0, 1],
            'confidence': [0.9, 0.8, 0.7, 0.6],
            'true': [0, 1, 0, 1]
        })
        mock_load.return_value = mock_df
        
        # Mock export
        mock_export.return_value = "exported_incidents.csv"
        
        monitoring_page()
        
        # Verify export was called
        mock_export.assert_called_once()
        self.mock_st.success.assert_called_once()

    @patch('utils.monitoring.list_incident_files')
    @patch('utils.monitoring.load_incidents')
    def test_incident_loading_error(self, mock_load, mock_list_files):
        """Test error handling during incident loading"""
        # Mock file selection
        mock_list_files.return_value = ["incident1.csv"]
        self.mock_st.selectbox.return_value = "incident1.csv"
        
        # Mock loading error
        mock_load.side_effect = Exception("Test error")
        
        monitoring_page()
        
        # Verify error message
        self.mock_st.error.assert_called_once()
        error_message = self.mock_st.error.call_args[0][0]
        self.assertIn("Помилка завантаження інцидентів", error_message)

if __name__ == '__main__':
    unittest.main() 