import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
# Model evaluation (with detailed metrics and visualizations)
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate trained HAR model performance."""
    
    def __init__(self, config_path: str = '../config/model_config.yaml'):
        """Initialize model evaluator with configuration."""
        self.config = self._load_config(config_path)
        self.features_dir = Path(self.config['data_paths']['features_dir'])
        self.models_dir = Path(self.config['data_paths']['models_dir'])
        self.results_dir = Path(self.config['data_paths']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model_and_data(self):
        """Load trained model, scaler, and test data."""
        # Load model and scaler
        model = joblib.load(self.models_dir / 'har_model.joblib')
        scaler = joblib.load(self.models_dir / 'scaler.joblib')
        
        # Load test features
        test_features = pd.read_csv(self.features_dir / 'test_features.csv')
        X_test = test_features.drop('activity', axis=1)
        y_test = test_features['activity']
        
        return model, scaler, X_test, y_test
    
    def plot_confusion_matrix(self, y_true, y_pred, labels):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png')
        plt.close()
    
    def evaluate_model(self):
        """Perform detailed model evaluation."""
        logger.info("Loading model and test data...")
        model, scaler, X_test, y_test = self.load_model_and_data()
        
        logger.info("Preprocessing test data...")
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("Generating predictions...")
        y_pred = model.predict(X_test_scaled)
        
        # Generate classification report
        logger.info("Classification Report:")
        report = classification_report(y_test, y_pred)
        logger.info("\n" + report)
        
        # Save classification report
        with open(self.results_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        # Plot confusion matrix
        logger.info("Generating confusion matrix plot...")
        activity_labels = sorted(y_test.unique())
        self.plot_confusion_matrix(y_test, y_pred, activity_labels)
        
        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save feature importance
            feature_importance.to_csv(
                self.results_dir / 'feature_importance.csv', 
                index=False
            )
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            sns.barplot(data=feature_importance.head(20), 
                       x='importance', y='feature')
            plt.title('Top 20 Most Important Features')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'feature_importance.png')
            plt.close()

def main():
    """Run the model evaluation process."""
    evaluator = ModelEvaluator()
    evaluator.evaluate_model()
    
if __name__ == '__main__':
    main()
