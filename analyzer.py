import pandas as pd
import numpy as np
import joblib
import os

def get_user_input():
    print("\n--- Chemical Toxicity Analyzer ---")
    print("Please enter the following molecular descriptors:")
    
    try:
        molwt = float(input("1. Molecular Weight: ").strip())
        logp = float(input("2. LogP (Partition Coefficient): ").strip())
        
        return np.array([[molwt, logp]])
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None

def analyze_toxicity(features):
    model_path = 'toxicity_model.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run train_model.py first.")
        return

    model = joblib.load(model_path)
    
    # Predict Class and Probability (optional, but good for confidence)
    prediction = model.predict(features)[0]
    # probability = model.predict_proba(features)[0] # Optional: if we want to show confidence
    
    status = "TOXIC" if prediction == 1 else "Non-Toxic"
    
    print("\n--- Analysis Result ---")
    print(f"Prediction: {status}")
    print("-------------------------")

if __name__ == "__main__":
    while True:
        features = get_user_input()
        if features is not None:
            analyze_toxicity(features)
        
        cont = input("\nAnalyze another chemical? (y/n): ").lower()
        if cont != 'y':
            break
