# Feature extraction (time and frequency domain features)
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import List, Tuple
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from processed sensor data for HAR prediction."""
    
    def __init__(self, config_path: str = '../config/model_config.yaml'):
        """Initialize feature extractor with configuration."""
        self.config = self._load_config(config_path)
        self.processed_dir = Path(self.config['data_paths']['processed_data_dir'])
        self.features_dir = Path(self.config['data_paths']['features_dir'])
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed training and test data."""
        train_data = pd.read_csv(self.processed_dir / 'train_processed.csv')
        test_data = pd.read_csv(self.processed_dir / 'test_processed.csv')
        return train_data, test_data
    
    def extract_time_features(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Extract time-domain features from sensor data."""
        features = pd.DataFrame()
        
        for col in columns:
            # Statistical features
            features[f'{col}_mean'] = data[col].mean()
            features[f'{col}_std'] = data[col].std()
            features[f'{col}_mad'] = data[col].mad()
            features[f'{col}_max'] = data[col].max()
            features[f'{col}_min'] = data[col].min()
            features[f'{col}_range'] = data[col].max() - data[col].min()
            features[f'{col}_skew'] = stats.skew(data[col])
            features[f'{col}_kurtosis'] = stats.kurtosis(data[col])
            
        return features
    
    def extract_frequency_features(self, data: pd.DataFrame, columns: List[str], 
                                 sample_rate: int = 50) -> pd.DataFrame:
        """Extract frequency-domain features from sensor data."""
        features = pd.DataFrame()
        
        for col in columns:
            # Perform FFT
            fft_values = np.fft.fft(data[col])
            fft_freq = np.fft.fftfreq(len(data[col]), 1/sample_rate)
            
            # Get magnitude spectrum
            magnitude_spectrum = np.abs(fft_values)
            
            # Extract frequency domain features
            features[f'{col}_freq_mean'] = np.mean(magnitude_spectrum)
            features[f'{col}_freq_std'] = np.std(magnitude_spectrum)
            features[f'{col}_dom_freq'] = fft_freq[np.argmax(magnitude_spectrum)]
            
        return features
    
    def build_features(self):
        """Main method to build all features."""
        logger.info("Loading processed data...")
        train_data, test_data = self.load_processed_data()
        
        # Separate sensor columns and activity labels
        sensor_columns = [col for col in train_data.columns if col != 'activity']
        
        logger.info("Extracting features...")
        # Extract features for training data
        train_features = pd.DataFrame()
        train_features = pd.concat([
            self.extract_time_features(train_data, sensor_columns),
            self.extract_frequency_features(train_data, sensor_columns)
        ], axis=1)
        train_features['activity'] = train_data['activity']
        
        # Extract features for test data
        test_features = pd.DataFrame()
        test_features = pd.concat([
            self.extract_time_features(test_data, sensor_columns),
            self.extract_frequency_features(test_data, sensor_columns)
        ], axis=1)
        test_features['activity'] = test_data['activity']
        
        logger.info("Saving features...")
        train_features.to_csv(self.features_dir / 'train_features.csv', index=False)
        test_features.to_csv(self.features_dir / 'test_features.csv', index=False)
        
        logger.info(f"Features built: {train_features.shape[1]} features created")
        return train_features, test_features

def main():
    """Run the feature building process."""
    feature_extractor = FeatureExtractor()
    train_features, test_features = feature_extractor.build_features()
    
if __name__ == '__main__':
    main()
