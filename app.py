 
# Flask web app for predictions
from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from flask_cors import CORS
 
app = Flask(__name__)
CORS(app)
 
# Paths to model and scaler
model_path = os.path.join("models", "model.joblib")
scaler_path = os.path.join("models", "scaler.joblib")
 
# Load model and scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
 
# Features to include (same as in training)
features_to_include = [
    'systolic_bp', 'diastolic_bp', 'cholesterol_total', 'cholesterol_hdl', 'cholesterol_ldl',
    'triglycerides', 'blood_sugar', 'heart_rate', 'smoking_status', 'bmi'
]
 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()
        if not data or "input" not in data:
            return jsonify({'error': 'No valid input received'}), 400
 
        input_array = data["input"]
 
        # Ensure the input has the correct length
        if len(input_array) != len(features_to_include):
            return jsonify({'error': 'Invalid input data format'}), 400
 
        # Map input values to expected feature names
        input_data = {features_to_include[i]: input_array[i] for i in range(len(features_to_include))}
        input_df = pd.DataFrame([input_data])
 
        # Ensure all input data are numeric
        input_df = input_df.astype(float)
 
        # Scale the input data
        input_scaled = scaler.transform(input_df)
 
        # Predict the risk category
        prediction = model.predict(input_scaled)
        risk_category = "You have high risk of CVD ADVICE: Please consult your Cardiologist for a comprehensive evaluation" if prediction[0] == 1 else "You have low risk of CVD ADVICE: Maintain a healthy lifestyle to keep risks low."
 
        return jsonify({'prediction': risk_category})
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500
 
if __name__ == '__main__':
    app.run(debug=True)
 
 