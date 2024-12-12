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
df = pd.read_csv(args.data_path)

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
print(classification_report(y_test, rf_preds))
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.2f}")
print(f"F1 Score: {f1_score(y_test, rf_preds, average='weighted'):.2f}")

print("\nLogistic Regression Results:")
print(classification_report(y_test, lr_preds))
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.2f}")
print(f"F1 Score: {f1_score(y_test, lr_preds, average='weighted'):.2f}")

from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

# Step 8: Visualize top 10 feature importance (Random Forest)
top_features = pd.Series(rf_model.feature_importances_, index=X.columns).nlargest(10)
plt.figure(figsize=(10, 6))
plt.barh(top_features.index, top_features.values)  # Using barh for horizontal bars
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()




