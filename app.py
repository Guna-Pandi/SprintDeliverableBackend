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

# Health advice function based on smoking status and risk category
def health_advice(smoking_status, prediction):
    if smoking_status == 0:  # Non-Smoker
        if prediction == "Low Risk (Non-Smoker)":
            return "Your health is in good shape! Continue maintaining a balanced diet, regular physical activity, and a smoke-free lifestyle."
        else:
            return "Your health risk is high. Please consult a healthcare provider for a full assessment. Consider reducing stress, improving diet, and increasing physical activity."
    
    elif smoking_status == 1:  # Occasional Smoker
        if prediction == "Low Risk (Occasional Smoker)":
            return "You are in good health, but as an occasional smoker, it's important to quit smoking to lower your long-term risk. Maintain a healthy lifestyle with regular exercise."
        else:
            return "Your health risk is concerning. Smoking contributes significantly to your health issues. Consider quitting smoking and speaking with your doctor about possible interventions."

    elif smoking_status == 2:  # Regular Smoker
        if prediction == "Low Risk (Regular Smoker)":
            return "You are at a low health risk, but smoking remains a major factor in increasing long-term health problems. Consider quitting smoking and adopting a healthier lifestyle."
        else:
            return "You have a high health risk due to smoking and other factors. Please see a healthcare provider immediately to discuss smoking cessation and other lifestyle changes."

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
        prediction = model.predict(input_scaled)[0]
        
        # Create the risk category label
        if prediction == 0:
            if input_data['smoking_status'] == 0:
                risk_category = "Low Risk (Non-Smoker)"
            elif input_data['smoking_status'] == 1:
                risk_category = "Low Risk (Occasional Smoker)"
            else:
                risk_category = "Low Risk (Regular Smoker)"
        else:
            if input_data['smoking_status'] == 0:
                risk_category = "High Risk (Non-Smoker)"
            elif input_data['smoking_status'] == 1:
                risk_category = "High Risk (Occasional Smoker)"
            else:
                risk_category = "High Risk (Regular Smoker)"

        # Get personalized health advice based on smoking status and risk category
        advice = health_advice(input_data['smoking_status'], risk_category)

        # Return the prediction and advice
        return jsonify({'prediction': risk_category, 'advice': advice})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
