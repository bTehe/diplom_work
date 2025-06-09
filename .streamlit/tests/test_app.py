import unittest
import streamlit as st
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import main

class TestApp(unittest.TestCase):
    @patch('streamlit.set_page_config')
    @patch('streamlit.sidebar')
    @patch('streamlit.option_menu')
    def test_main_page_config(self, mock_option_menu, mock_sidebar, mock_set_page_config):
        """Test if the main page configuration is set correctly"""
        # Mock the option_menu return value
        mock_option_menu.return_value = "Dashboard"
        
        # Call the main function
        main()
        
        # Verify that set_page_config was called with correct parameters
        mock_set_page_config.assert_called_once_with(
            page_title="My ML App",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    @patch('streamlit.sidebar')
    @patch('streamlit.option_menu')
    def test_navigation_options(self, mock_option_menu, mock_sidebar):
        """Test if all navigation options are present"""
        # Mock the option_menu return value
        mock_option_menu.return_value = "Dashboard"
        
        # Call the main function
        main()
        
        # Verify that option_menu was called with correct options
        mock_option_menu.assert_called_once()
        call_args = mock_option_menu.call_args[1]
        self.assertEqual(
            call_args['options'],
            ["Dashboard", "Upload", "Preprocessing", "Training", 
             "Inference", "Monitoring", "Reports"]
        )

if __name__ == '__main__':
    unittest.main() 