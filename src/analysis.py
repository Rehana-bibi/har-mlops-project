import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(base_path):
    """Load and prepare HAR dataset"""
    # Load activity labels and features
    activity_labels = pd.read_csv(base_path / 'activity_labels.txt', 
                                sep=' ', 
                                header=None, 
                                names=['id', 'activity'])

    features = pd.read_csv(base_path / 'features.txt',
                          sep=' ',
                          header=None,
                          names=['id', 'feature'])

    # Load training data
    X_train = pd.read_csv(base_path / 'train/X_train.txt', 
                          sep='\s+', 
                          header=None,
                          names=features['feature'])

    y_train = pd.read_csv(base_path / 'train/y_train.txt',
                          header=None,
                          names=['activity'])

    # Load test data
    X_test = pd.read_csv(base_path / 'test/X_test.txt',
                         sep='\s+',
                         header=None,
                         names=features['feature'])

    y_test = pd.read_csv(base_path / 'test/y_test.txt',
                         header=None,
                         names=['activity'])
    
    return X_train, X_test, y_train, y_test, activity_labels, features

def print_dataset_info(X_train, X_test, features, activity_labels):
    """Print basic information about the dataset"""
    print("Dataset Overview:")
    print("-" * 50)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {len(features)}")
    print(f"\nActivities in the dataset:")
    print(activity_labels)

def plot_activity_distribution(y_train, activity_labels):
    """Plot distribution of activities in training set"""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=y_train, x='activity')
    plt.title('Distribution of Activities in Training Set')
    plt.xticks(range(6), activity_labels['activity'], rotation=45)
    plt.tight_layout()
    plt.savefig('activity_distribution.png')
    plt.close()

def check_data_quality(X_train, X_test):
    """Check for missing values and print feature statistics"""
    print("\nFeature Statistics:")
    print("-" * 50)
    print(X_train.describe())

    print("\nMissing Values Check:")
    print("-" * 50)
    print("Training set missing values:", X_train.isnull().sum().sum())
    print("Test set missing values:", X_test.isnull().sum().sum())

def plot_correlation_matrix(X_train):
    """Plot correlation matrix for first 10 features"""
    plt.figure(figsize=(12, 8))
    selected_features = X_train.iloc[:, :10]
    correlation_matrix = selected_features.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of First 10 Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def plot_activity_signals(X_train, y_train, activity_labels, activity_id=1, n_samples=100):
    """Plot accelerometer and gyroscope signals for a specific activity"""
    activity_data = X_train[y_train['activity'] == activity_id].iloc[:n_samples]
    activity_name = activity_labels.loc[activity_id-1, 'activity']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot accelerometer data
    acc_columns = [col for col in activity_data.columns if 'tBodyAcc-mean()' in col]
    activity_data[acc_columns].plot(ax=ax1)
    ax1.set_title(f'Accelerometer Signals for {activity_name}')
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Acceleration')
    ax1.legend(['X-axis', 'Y-axis', 'Z-axis'])
    
    # Plot gyroscope data
    gyro_columns = [col for col in activity_data.columns if 'tBodyGyro-mean()' in col]
    activity_data[gyro_columns].plot(ax=ax2)
    ax2.set_title(f'Gyroscope Signals for {activity_name}')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Angular velocity')
    ax2.legend(['X-axis', 'Y-axis', 'Z-axis'])
    
    plt.tight_layout()
    plt.savefig(f'activity_{activity_name}_signals.png')
    plt.close()

def save_processed_data(X_train, X_test, y_train, y_test, output_path):
    """Save processed datasets"""
    output_path.mkdir(exist_ok=True)
    X_train.to_csv(output_path / 'X_train_processed.csv', index=False)
    X_test.to_csv(output_path / 'X_test_processed.csv', index=False)
    y_train.to_csv(output_path / 'y_train_processed.csv', index=False)
    y_test.to_csv(output_path / 'y_test_processed.csv', index=False)

def main():
    # Set paths
    base_path = Path('/Users/macbookair/Desktop/har-mlops-project/data/raw/UCI HAR Dataset')
    output_path = Path('/Users/macbookair/Desktop/har-mlops-project/data/processed')
    
    # Load data
    X_train, X_test, y_train, y_test, activity_labels, features = load_data(base_path)
    
    # Run analysis
    print_dataset_info(X_train, X_test, features, activity_labels)
    plot_activity_distribution(y_train, activity_labels)
    check_data_quality(X_train, X_test)
    plot_correlation_matrix(X_train)
    
    # Plot signals for walking activity
    plot_activity_signals(X_train, y_train, activity_labels, activity_id=1)
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, output_path)
    
    print("\nAnalysis complete! Check the output directory for plots and processed data.")

if __name__ == "__main__":
    main()