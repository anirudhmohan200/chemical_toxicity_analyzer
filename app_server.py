from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model
MODEL_PATH = 'toxicity_model.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train_model.py first.")

model = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in order: MolWt, LogP
        # Ensure we handle potential missing or invalid data gracefully
        try:
            molwt = float(data['molwt'])
            logp = float(data['logp'])
        except (KeyError, ValueError, TypeError) as e:
             return jsonify({'success': False, 'error': "Invalid input: Please provide numeric 'molwt' and 'logp'."})


        features = [molwt, logp]
        
        # Predict
        prediction = model.predict([features])[0]
        
        # Classification Logic
        # 1 = Toxic, 0 = Non-Toxic (based on usual dataset conventions, assumed from 'toxic' col)
        
        status = "TOXIC" if prediction == 1 else "Safe / Non-Toxic"
        
        return jsonify({
            'success': True,
            'prediction': str(prediction), # Returns "1" or "0"
            'status': status
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
