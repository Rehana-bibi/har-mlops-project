# Step 1: Import libraries
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



# Enable MLflow autologging
mlflow.sklearn.autolog()

# Add argument parser to accept data_path
# parser = argparse.ArgumentParser()
# parser.add_argument("--data_path", type=str, help="Path to the input data")
# args = parser.parse_args()
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to the input data", default="./data/har.csv")
args = parser.parse_args()


# Add these debug lines after your argparse setup
print("Current working directory:", os.getcwd())
print("Attempting to read file from:", os.path.abspath(args.data_path))


# Check if file exists at the path
if not os.path.exists(args.data_path):
    print(f"Error: File not found at {args.data_path}")
    print("Please provide correct path using --data_path argument")
    exit(1)


# Step 2: Load the dataset using the provided data_path
try:
    df = pd.read_csv(args.data_path)
except Exception as e:
    print(f"Error reading dataset: {e}")
    exit(1)

# Handle missing values (if any)
if df.isnull().sum().sum() > 0:
    print("Warning: Dataset contains missing values. Handling missing values.")
    df = df.fillna(df.mean())  # Impute with mean, or choose another strategy



# Confirm the dataset is loaded correctly
print("First 5 Rows of Dataset:")
print(df.head())

# Step 3: Understand the dataset
print("\nDataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# Step 4: Preprocess the data
# Separate features and target variable
X = df.drop(['Activity'], axis=1)  # Features
y = df['Activity']                # Target


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Fit and evaluate models
# Start MLflow tracking
mlflow.start_run()

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Step 6: Evaluate performance
print("\nRandom Forest Results:")
rf_accuracy = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds, average='weighted')
print(classification_report(y_test, rf_preds))
print(f"Accuracy: {rf_accuracy:.2f}")
print(f"F1 Score: {rf_f1:.2f}")

# Log metrics for Random Forest
mlflow.log_metric("rf_accuracy", rf_accuracy)
mlflow.log_metric("rf_f1_score", rf_f1)

print("\nLogistic Regression Results:")
lr_accuracy = accuracy_score(y_test, lr_preds)
lr_f1 = f1_score(y_test, lr_preds, average='weighted')
print(classification_report(y_test, lr_preds))
print(f"Accuracy: {lr_accuracy:.2f}")
print(f"F1 Score: {lr_f1:.2f}")

# Log metrics for Logistic Regression
mlflow.log_metric("lr_accuracy", lr_accuracy)
mlflow.log_metric("lr_f1_score", lr_f1)

# Save the models to MLflow
mlflow.sklearn.log_model(rf_model, "random_forest_model")
mlflow.sklearn.log_model(lr_model, "logistic_regression_model")

# Step 7: Cross-Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
mean_accuracy = scores.mean()
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", mean_accuracy)

# Log cross-validation metric
mlflow.log_metric("rf_mean_cv_accuracy", mean_accuracy)

# End MLflow tracking
mlflow.end_run()

# Step 8: Visualize top 10 feature importance (Random Forest)
top_features = pd.Series(rf_model.feature_importances_, index=X.columns).nlargest(10)
plt.figure(figsize=(10, 6))
plt.barh(top_features.index, top_features.values)  # Using barh for horizontal bars
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

