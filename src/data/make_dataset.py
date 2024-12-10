
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from azure.storage.blob import BlobServiceClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HARDataLoader:
    def __init__(self, config_path: str = 'config/model_config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Azure Blob Storage
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.config['azure_storage_connection_string']
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.config['container_name']
        )

    def read_data(self, data_path: str) -> tuple:
        """Read the HAR dataset files"""
        try:
            # Read training data
            X_train = pd.read_csv(f"{data_path}/train/X_train.txt", delimiter=r"\s+", header=None)
            y_train = pd.read_csv(f"{data_path}/train/y_train.txt", header=None)
            
            # Read test data
            X_test = pd.read_csv(f"{data_path}/test/X_test.txt", delimiter=r"\s+", header=None)
            y_test = pd.read_csv(f"{data_path}/test/y_test.txt", header=None)
            
            # Read feature names
            features = pd.read_csv(f"{data_path}/features.txt", 
                                 delimiter=r"\s+", 
                                 header=None, 
                                 names=['index', 'feature_name'])
            
            # Set feature names
            X_train.columns = features['feature_name']
            X_test.columns = features['feature_name']
            
            logger.info(f"Data loaded successfully. Training set shape: {X_train.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error reading data: {str(e)}")
            raise

    def preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """Preprocess the data"""
        try:
            # Handle missing values
            X_train = X_train.fillna(X_train.mean())
            X_test = X_test.fillna(X_test.mean())
            
            # Normalize features
            X_train_normalized = (X_train - X_train.mean()) / X_train.std()
            X_test_normalized = (X_test - X_test.mean()) / X_test.std()
            
            logger.info("Data preprocessing completed successfully")
            return X_train_normalized, X_test_normalized
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def save_to_azure(self, data: pd.DataFrame, filename: str) -> None:
        """Save processed data to Azure Blob Storage"""
        try:
            # Convert to parquet for efficient storage
            data.to_parquet(filename)
            
            # Upload to Azure
            with open(filename, "rb") as data_file:
                self.container_client.upload_blob(
                    name=f"processed/{filename}",
                    data=data_file,
                    overwrite=True
                )
            logger.info(f"Data saved to Azure: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving to Azure: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize data loader
    data_loader = HARDataLoader()
    
    # Load data
    X_train, X_test, y_train, y_test = data_loader.read_data("data/raw")
    
    # Preprocess data
    X_train_processed, X_test_processed = data_loader.preprocess_data(X_train, X_test)
    
    # Save processed data to Azure
    data_loader.save_to_azure(X_train_processed, "X_train_processed.parquet")
    data_loader.save_to_azure(X_test_processed, "X_test_processed.parquet")
    data_loader.save_to_azure(y_train, "y_train.parquet")
    data_loader.save_to_azure(y_test, "y_test.parquet")
>>>>>>> Stashed changes
