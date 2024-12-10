# Model training (using RandomForest classifier)
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train machine learning model for HAR prediction."""
    
    def __init__(self, config_path: str = '../config/model_config.yaml'):
        """Initialize model trainer with configuration."""
        self.config = self._load_config(config_path)
        self.features_dir = Path(self.config['data_paths']['features_dir'])
        self.models_dir = Path(self.config['data_paths']['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_features(self):
        """Load feature data."""
        train_features = pd.read_csv(self.features_dir / 'train_features.csv')
        test_features = pd.read_csv(self.features_dir / 'test_features.csv')
        
        # Separate features and labels
        X_train = train_features.drop('activity', axis=1)
        y_train = train_features['activity']
        X_test = test_features.drop('activity', axis=1)
        y_test = test_features['activity']
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self):
        """Train the machine learning model."""
        logger.info("Loading feature data...")
        X_train, X_test, y_train, y_test = self.load_features()
        
        logger.info("Preprocessing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("Training model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test_scaled)
        logger.info("\n" + classification_report(y_test, y_pred))
        
        logger.info("Saving model and scaler...")
        joblib.dump(model, self.models_dir / 'har_model.joblib')
        joblib.dump(scaler, self.models_dir / 'scaler.joblib')
        
        return model, scaler

def main():
    """Run the model training process."""
    trainer = ModelTrainer()
    model, scaler = trainer.train_model()
    
if __name__ == '__main__':
    main()
