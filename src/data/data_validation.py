# src/data/data_validation.py
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path) as conf_file:
            self.config = yaml.safe_load(conf_file)
            
    def check_data_quality(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, bool]:
        """Perform data quality checks"""
        quality_checks = {
            'missing_values': self._check_missing_values(X),
            'shape_consistency': self._check_shape_consistency(X, y),
            'value_ranges': self._check_value_ranges(X),
            'label_validity': self._check_labels(y)
        }
        
        return quality_checks
        
    def _check_missing_values(self, X: pd.DataFrame) -> bool:
        """Check for missing values in features"""
        return X.isnull().sum().sum() == 0
        
    def _check_shape_consistency(self, X: pd.DataFrame, y: pd.DataFrame) -> bool:
        """Check if features and labels have consistent shapes"""
        return len(X) == len(y)
        
    def _check_value_ranges(self, X: pd.DataFrame) -> bool:
        """Check if values are within expected ranges"""
        return (X.values >= -1).all() and (X.values <= 1).all()
        
    def _check_labels(self, y: pd.DataFrame) -> bool:
        """Check if labels are valid"""
        valid_labels = set(range(1, 7))  # HAR dataset has 6 activities
        return set(y[0].unique()).issubset(valid_labels)
        
    def validate_dataset(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                        X_test: pd.DataFrame, y_test: pd.DataFrame) -> bool:
        """Validate both training and test datasets"""
        try:
            train_checks = self.check_data_quality(X_train, y_train)
            test_checks = self.check_data_quality(X_test, y_test)
            
            all_checks_passed = all(train_checks.values()) and all(test_checks.values())
            
            if all_checks_passed:
                logger.info("All data validation checks passed")
            else:
                failed_checks = {k: v for k, v in {**train_checks, **test_checks}.items() if not v}
                logger.warning(f"Failed checks: {failed_checks}")
                
            return all_checks_passed
            
        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
            raise

if __name__ == "__main__":
    from make_dataset import DataLoader
    
    # Load data
    data_loader = DataLoader()
    X_train, y_train, X_test, y_test = data_loader.load_raw_data()
    
    # Validate data
    validator = DataValidator()
    is_valid = validator.validate_dataset(X_train, y_train, X_test, y_test)
