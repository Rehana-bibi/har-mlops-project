# test_integration.py

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import sys
from io import StringIO
from main import main

class TestMLPipeline(unittest.TestCase):
    """Integration tests for the complete ML pipeline"""

    def setUp(self):
        """Set up test environment"""
        # Create sample dataset
        self.test_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'Activity': np.random.randint(0, 2, 100)
        })
        
        # Create temporary directory and save test data
        self.test_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.test_dir, 'test_data.csv')
        self.test_data.to_csv(self.test_csv_path, index=False)

    def test_complete_pipeline(self):
        """Test the entire ML pipeline"""
        # Redirect stdout to capture output
        captured_output = StringIO()
        sys.stdout = captured_output

        # Run main with test data
        sys.argv = ['main.py', '--data_path', self.test_csv_path]
        try:
            main()
        except SystemExit:
            pass

        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Check if key outputs are present
        output = captured_output.getvalue()
        self.assertIn("Training Random Forest Model", output)
        self.assertIn("Training Logistic Regression Model", output)
        self.assertIn("Random Forest Results", output)
        self.assertIn("Logistic Regression Results", output)
        
        # Check if feature importance plot was created
        self.assertTrue(os.path.exists('feature_importance.png'))

    def test_error_handling(self):
        """Test error handling for non-existent file"""
        with self.assertRaises(SystemExit):
            sys.argv = ['main.py', '--data_path', 'nonexistent.csv']
            main()

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
        if os.path.exists('feature_importance.png'):
            os.remove('feature_importance.png')
        os.rmdir(self.test_dir)

if __name__ == '__main__':
    unittest.main()