import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib
import os
 
# Paths to data files
demographic_path = os.path.join('data', 'Patient_demographics_data.csv')
diagnosis_path = os.path.join('data', 'Patient_diagnosis_data.csv')
 
# Load data
try:
    demographic_df = pd.read_csv(demographic_path)
    diagnosis_df = pd.read_csv(diagnosis_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: One or more data files were not found.")
    exit()
 
# Merge datasets on 'patient_id'
if 'patient_id' not in demographic_df.columns or 'patient_id' not in diagnosis_df.columns:
    print("Error: Missing 'patient_id' column in one of the datasets.")
    exit()
 
merged_df = pd.merge(demographic_df, diagnosis_df, on='patient_id')
 
# Define the features for prediction
features_to_include = [
    'systolic_bp', 'diastolic_bp', 'cholesterol_total', 'cholesterol_hdl', 'cholesterol_ldl',
    'triglycerides', 'blood_sugar', 'heart_rate', 'smoking_status', 'bmi'
]
 
# Add a target column if it's not already there
if 'risk_category' not in merged_df.columns:
    def categorize_risk(row):
        risk_score = 0
 
        # Add points for high values in each health indicator (based on thresholds)
        if row['systolic_bp'] > 160:
            risk_score += 1
        if row['diastolic_bp'] > 100:
            risk_score += 1
        if row['cholesterol_total'] > 280 :
            risk_score += 1
        if row['cholesterol_hdl'] < 35 :
            risk_score += 1
        if row['cholesterol_ldl'] > 180:
            risk_score += 1
        if row['triglycerides'] > 200:
            risk_score += 1
        if row['blood_sugar'] > 130:
            risk_score += 1
        if row['bmi'] > 35:
            risk_score += 1
        if row['heart_rate'] > 100:
            risk_score += 1
        if row['smoking_status'] in [1,2]:
            risk_score += 1
 
        # If risk score is high, categorize as High Risk, otherwise Low Risk
        if risk_score >= 2:
            return 'High Risk'
        else:
            return'Low Risk'
   
    merged_df['risk_category'] = merged_df.apply(categorize_risk, axis=1)
 
# Check for missing columns
missing_columns = [col for col in features_to_include if col not in merged_df.columns]
if missing_columns:
    print(f"Error: Missing columns {missing_columns} in the dataset.")
    exit()
 
# Separate numeric and categorical columns
numeric_columns = ['systolic_bp', 'diastolic_bp', 'cholesterol_total', 'cholesterol_hdl',
                   'cholesterol_ldl', 'triglycerides', 'blood_sugar', 'heart_rate', 'bmi']
categorical_columns = ['smoking_status']
 
# Impute missing values for numeric columns (mean strategy)
numeric_imputer = SimpleImputer(strategy='mean')
merged_df[numeric_columns] = numeric_imputer.fit_transform(merged_df[numeric_columns])
 
# Impute missing values for categorical columns (most frequent strategy)
categorical_imputer = SimpleImputer(strategy='most_frequent')
merged_df[categorical_columns] = categorical_imputer.fit_transform(merged_df[categorical_columns])
 
# Encode 'smoking_status' using LabelEncoder for three categories
if 'smoking_status' in merged_df.columns:
    le = LabelEncoder()
    merged_df['smoking_status'] = le.fit_transform(merged_df['smoking_status'])
 
# Prepare features and target
X = merged_df[features_to_include]
y = merged_df['risk_category'].map({'Low Risk': 0, 'High Risk': 1})
 
# Check if y contains both classes
if len(np.unique(y)) < 2:
    print("Error: The target variable `y` does not contain both classes.")
    exit()
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
# Check if both classes are in the training set
if len(np.unique(y_train)) < 2:
    print("Error: `y_train` contains only one class. Adjusting the train-test split...")
 
# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
 
# Check class distribution after SMOTE
print("Class distribution after SMOTE:")
print(pd.Series(y_train_res).value_counts())
 
# Standardize features
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)
 
# Train the model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_res_scaled, y_train_res)
 
# Save the model and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
 
# Evaluate the model
y_pred = model.predict(X_test_scaled)
 
# Print classification report for more detailed performance metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))
 
# Print model accuracy
accuracy = model.score(X_test_scaled, y_test)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Model and scaler saved successfully.")