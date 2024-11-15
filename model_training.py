import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib

# Load patient demographic data
try:
    demographic_df = pd.read_csv('data/Patient_demographics_data.csv')
    diagnosis_df = pd.read_csv('data/Patient_diagnosis_data.csv')   
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: One or more data files were not found.")
    exit()

# Ensure both dataframes have the necessary columns for merging
# Assuming that both files can be merged on a common 'patient_id' column (adjust as needed)
if 'patient_id' not in demographic_df.columns or 'patient_id' not in diagnosis_df.columns:
    print("Error: Missing 'patient_id' column in one of the datasets.")
    exit()

# Merge both dataframes on the 'patient_id' column
merged_df = pd.merge(demographic_df, diagnosis_df, on='patient_id')

# Data Preprocessing (Assume the dataset has these columns, modify if necessary)
required_columns = ['systolic_bp', 'diastolic_bp', 'cholesterol_total', 'cholesterol_hdl', 'cholesterol_ldl', 
                    'triglycerides', 'blood_sugar', 'heart_rate', 'bmi', 'risk_category']  # Add your target column

# Check if the required columns are in the merged dataset
for col in required_columns[:-1]:  # Exclude 'risk_category' while checking
    if col not in merged_df.columns:
        print(f"Error: Missing column '{col}' in the dataset.")
        exit()

# Create 'risk_category' column if missing (for example, based on cholesterol and blood pressure)
def categorize_risk(row):
    # Example business logic for risk categorization
    if row['cholesterol_total'] > 240 or row['systolic_bp'] > 140:
        return 'High Risk'
    else:
        return 'Low Risk'

# Apply the function to create the 'risk_category' column
merged_df['risk_category'] = merged_df.apply(categorize_risk, axis=1)

# Prepare the features (X) and target variable (y)
X = merged_df[required_columns[:-1]]  # All columns except the target
y = merged_df['risk_category']   # The target column (risk category)

# Convert the target to numeric (1 for high risk, 0 for low risk)
y = y.apply(lambda x: 1 if x == 'High Risk' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Standardize the features
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_res_scaled, y_train_res)

# Evaluate the model (optional, for your understanding)
accuracy = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and scaler using joblib
joblib.dump(model, 'models/model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')

print("Model and scaler saved successfully.")
