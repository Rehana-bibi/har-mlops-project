# test_unit.py

import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os
import tempfile

class TestMLComponents(unittest.TestCase):
    """Unit tests for individual components"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample data
        self.X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        self.y = pd.Series([0, 1, 0, 1, 0])
        
        # Create sample CSV
        self.test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'Activity': [0, 1, 0, 1, 0]
        })
        
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.test_dir, 'test_data.csv')
        self.test_data.to_csv(self.test_csv_path, index=False)

    def test_data_loading(self):
        """Test data loading functionality"""
        loaded_data = pd.read_csv(self.test_csv_path)
        self.assertEqual(len(loaded_data), 5)
        self.assertTrue('Activity' in loaded_data.columns)
        self.assertEqual(len(loaded_data.columns), 3)

    def test_scaling(self):
        """Test data scaling"""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.X)
        self.assertAlmostEqual(np.mean(scaled_data), 0, places=10)
        self.assertAlmostEqual(np.std(scaled_data), 1, places=10)

    def test_random_forest(self):
        """Test Random Forest model"""
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(self.X, self.y)
        predictions = rf_model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(all(isinstance(pred, (np.integer, int)) for pred in predictions))

    def test_logistic_regression(self):
        """Test Logistic Regression model"""
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(self.X, self.y)
        predictions = lr_model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(all(isinstance(pred, (np.integer, int)) for pred in predictions))

    def tearDown(self):
        """Clean up test files"""
        os.remove(self.test_csv_path)
        os.rmdir(self.test_dir)

if __name__ == '__main__':
    unittest.main()