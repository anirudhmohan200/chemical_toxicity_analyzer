import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
dataset_path = r'C:\Users\user\chemical_toxicity_app\rdkit_toxicity_dataset.csv'
df = pd.read_csv(dataset_path)

print("Dataset loaded successfully.")
print(df.head())

# Features and Target
# Based on the new dataset: 'MolWt', 'LogP' are features, 'toxic' is target (0 or 1)
X = df[['MolWt', 'LogP']]
y = df['toxic']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Model
# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train Model
print("Training model...")
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Training Completed.")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Save Model
joblib.dump(model, 'toxicity_model.pkl')
print("Model saved to 'toxicity_model.pkl'")
