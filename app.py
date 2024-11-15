import pandas as pd
from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS  # Import CORS for handling cross-origin requests

app = Flask(__name__)
CORS(app)  # Allow all origins to access this app

# Load the model and scaler
model_path = os.path.join("models", "model.joblib")
scaler_path = os.path.join("models", "scaler.joblib")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")

# List of features expected by the model
features_to_include = [
    'sbp', 'ldl', 'dbp', 'tgc', 'tc', 'bs', 'hdl', 'bmi', 'smoke', 'heart_rate'
]

@app.route('/predict', methods=['POST'])  # Ensure it's POST
def predict():
    try:
        # Get JSON data sent by React
        data = request.get_json()
        if not data or "input" not in data:
            return jsonify({'error': 'No valid input received'}), 400

        input_array = data["input"]  # Extract the input values from the request
        
        # Check if the input array length matches the expected features
        if len(input_array) != len(features_to_include):
            return jsonify({'error': 'Invalid input data format'}), 400

        # Process categorical input ("smoke")
        smoke_mapping = {'Non-Smoker': 0, 'Smoker': 1}
        input_array[8] = smoke_mapping.get(input_array[8], -1)  # Encode 'smoke'

        if input_array[8] == -1:
            return jsonify({'error': 'Invalid value for smoke'}), 400

        # Map input values to the expected feature names
        input_data = {features_to_include[idx]: input_array[idx] for idx in range(len(features_to_include))}
        input_df = pd.DataFrame([input_data])

        print("Input DataFrame:", input_df)  # Debugging log

        # Ensure the data is in the correct format (float)
        input_df = input_df.astype(float)

        # Scale the input data using the pre-trained scaler
        input_data_scaled = scaler.transform(input_df)

        # Make prediction using the pre-trained model
        prediction = model.predict(input_data_scaled)

        # Interpret prediction
        risk_category = "High Risk" if prediction[0] == 1 else "Low Risk"

        return jsonify({'prediction': risk_category})

    except Exception as e:
        print("Error during prediction:", str(e))  # Log the error
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
