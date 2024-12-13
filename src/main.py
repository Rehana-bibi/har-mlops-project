import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import argparse
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Enable MLflow autologging
mlflow.sklearn.autolog()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the input data", default="./data/har.csv")
    args = parser.parse_args()

    # Debug information
    print("Current working directory:", os.getcwd())
    print("Attempting to read file from:", os.path.abspath(args.data_path))

    # Check if file exists
    if not os.path.exists(args.data_path):
        print(f"Error: File not found at {args.data_path}")
        print("Please provide correct path using --data_path argument")
        exit(1)

    # Load the dataset
    try:
        df = pd.read_csv(args.data_path)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        exit(1)

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        print("Warning: Dataset contains missing values. Handling missing values.")
        df = df.fillna(df.mean())

    # Start MLflow run after data loading validation
    with mlflow.start_run() as run:
        # Enable autologging
        mlflow.sklearn.autolog(log_models=True)

        # Rest of your model training code goes here...

        # Print dataset info
        print("First 5 Rows of Dataset:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
        print("\nDataset Description:")
        print(df.describe())

        # Prepare features and target
        X = df.drop(['Activity'], axis=1)
        y = df['Activity']

        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train and evaluate Random Forest
        print("\nTraining Random Forest Model...\n\n")
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)

        print("\nRandom Forest Results:\n")
        rf_accuracy = accuracy_score(y_test, rf_preds)
        rf_f1 = f1_score(y_test, rf_preds, average='weighted')
        print(classification_report(y_test, rf_preds))
        print(f"Accuracy: {rf_accuracy:.2f}")
        print(f"F1 Score: {rf_f1:.2f}")

        # Random Forest Cross-Validation
        rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
        rf_mean_accuracy = rf_scores.mean()
        print("\nRandom Forest Cross-Validation Scores:", rf_scores)
        print("Mean CV Accuracy:", rf_mean_accuracy)

        # Add MLflow logging for Random Forest
        mlflow.log_metric("rf_accuracy", rf_accuracy)
        mlflow.log_metric("rf_f1_score", rf_f1)
        mlflow.log_metric("rf_cv_mean_accuracy", rf_mean_accuracy)
        
        # Log Random Forest model
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            artifact_path="random_forest_model",
            registered_model_name="har_activity_recognition_rf"
        )

        # Train and evaluate Logistic Regression
        print("\n\nTraining Logistic Regression Model...\n\n")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        lr_preds = lr_model.predict(X_test)

        print("\n\nLogistic Regression Results:\n")
        lr_accuracy = accuracy_score(y_test, lr_preds)
        lr_f1 = f1_score(y_test, lr_preds, average='weighted')
        print(classification_report(y_test, lr_preds))
        print(f"Accuracy: {lr_accuracy:.2f}")
        print(f"F1 Score: {lr_f1:.2f}")

        # Logistic Regression Cross-Validation
        lr_scores = cross_val_score(lr_model, X, y, cv=5, scoring='accuracy')
        lr_mean_accuracy = lr_scores.mean()
        print("\n\nLogistic Regression Cross-Validation Scores:", lr_scores)
        print("Mean CV Accuracy:", lr_mean_accuracy)

        # Add MLflow logging for Logistic Regression
        mlflow.log_metric("lr_accuracy", lr_accuracy)
        mlflow.log_metric("lr_f1_score", lr_f1)
        mlflow.log_metric("lr_cv_mean_accuracy", lr_mean_accuracy)
        
        # Log Logistic Regression model
        mlflow.sklearn.log_model(
            sk_model=lr_model,
            artifact_path="logistic_regression_model",
            registered_model_name="har_activity_recognition_lr"
        )

        # Add this after both models are trained and logged
        print(f"\nMLflow Run Summary:")
        print(f"Run ID: {run.info.run_id}")
        print(f"Models registered as:")
        


    # Visualize top 10 feature importance (Random Forest)
    top_features = pd.Series(rf_model.feature_importances_, index=X.columns).nlargest(10)
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.barh(top_features.index, top_features.values, color=colors)
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # # Extract a valid sample input
    # sample_input = df.drop(['Activity'], axis=1).iloc[0].tolist()
    # print("\nSample Input Format:")
    # print({"data": [sample_input]})

if __name__ == "__main__":
    main()