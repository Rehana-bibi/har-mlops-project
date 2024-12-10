# src/data/make_dataset.py
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from azure.ml.core import Run
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path) as conf_file:
            self.config = yaml.safe_load(conf_file)
        self.run = Run.get_context()
        
    def load_raw_data(self):
        """Load raw data from files"""
        try:
            # Load training data
            X_train = pd.read_csv(
                Path(self.config['data']['raw_data_path']) / self.config['data']['train_data']['features'],
                delim_whitespace=True, header=None
            )
            y_train = pd.read_csv(
                Path(self.config['data']['raw_data_path']) / self.config['data']['train_data']['labels'],
                header=None
            )
            
            # Load test data
            X_test = pd.read_csv(
                Path(self.config['data']['raw_data_path']) / self.config['data']['test_data']['features'],
                delim_whitespace=True, header=None
            )
            y_test = pd.read_csv(
                Path(self.config['data']['raw_data_path']) / self.config['data']['test_data']['labels'],
                header=None
            )
            
            logger.info("Raw data loaded successfully")
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise
            
    def preprocess_data(self, X_train, y_train, X_test, y_test):
        """Preprocess the data"""
        try:
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save processed data
            processed_path = Path(self.config['data']['processed_data_path'])
            np.save(processed_path / 'X_train_processed.npy', X_train_scaled)
            np.save(processed_path / 'X_test_processed.npy', X_test_scaled)
            np.save(processed_path / 'y_train_processed.npy', y_train.values)
            np.save(processed_path / 'y_test_processed.npy', y_test.values)
            
            # Log to Azure ML
            self.run.log('data_processed', True)
            logger.info("Data preprocessing completed")
            
            return X_train_scaled, y_train, X_test_scaled, y_test
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

if __name__ == "__main__":
    data_loader = DataLoader()
    X_train, y_train, X_test, y_test = data_loader.load_raw_data()
    X_train_proc, y_train_proc, X_test_proc, y_test_proc = data_loader.preprocess_data(
        X_train, y_train, X_test, y_test
    )
