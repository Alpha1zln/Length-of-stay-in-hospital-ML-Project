import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Correct path using raw string
path = r"C:\Users\shreyas1bst\Desktop\6th_sem_proj\analytics vidhya dataset\pickle file\RmvdCol"
model_filename = "decision_tree_unq_model.pkl"

# Load model
model_path = os.path.join(path, model_filename)
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Length of Stay Prediction")

# Define mappings for Label Encoding
label_mappings = {
    "Hospital_type_code": {'c': 2, 'e': 4, 'b': 1, 'a': 0, 'f': 5, 'd': 3, 'g': 6},
    "Department": {'radiotherapy': 3, 'anesthesia': 1, 'gynecology': 2, 'TB & Chest disease': 0, 'surgery': 4},
    "Ward_Type": {'R': 2, 'S': 3, 'Q': 1, 'P': 0, 'T': 4, 'U': 5},
    "Ward_Facility_Code": {'F': 5, 'E': 4, 'D': 3, 'B': 1, 'A': 0, 'C': 2},
    "Type of Admission": {'Emergency': 0, 'Trauma': 1, 'Urgent': 2},
    "Severity of Illness": {'Extreme': 0, 'Moderate': 2, 'Minor': 1},
    "Age": {'0-10': 0, '11-20': 1, '21-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, '71-80': 7, '81-90': 8, '91-100': 9}
}

# Reverse mapping for prediction output
stay_dict_rev = {0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50', 
                 5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100', 10: 'More than 100 Days'}

# Mean and standard deviation for Standard Scaling (must match training data)
scaler_params = {
    "Type of Admission": {"mean": 1.04, "std": 0.85},  # Example values, replace with actual from training
    "Admission_Deposit": {"mean": 5000, "std": 2000}   # Example values, replace with actual from training
}

# Streamlit Input Fields
input_data = {}

# Categorical Inputs
for feature, mapping in label_mappings.items():
    selected_value = st.selectbox(f"{feature}:", list(mapping.keys()))
    input_data[feature] = mapping[selected_value]

# Numerical Input for Admission Deposit
input_data["Admission_Deposit"] = st.number_input("Admission_Deposit:", min_value=1000.0, max_value=10000.0, value=5000.0)

# Convert input_data to DataFrame
input_df = pd.DataFrame([input_data])

# Apply Standard Scaling
for feature in ["Type of Admission", "Admission_Deposit"]:
    mean_val = scaler_params[feature]["mean"]
    std_val = scaler_params[feature]["std"]
    input_df[feature] = (input_df[feature] - mean_val) / std_val

# Ensure correct feature order
input_df = input_df[['Hospital_type_code', 'Department', 'Ward_Type', 'Ward_Facility_Code',
                     'Type of Admission', 'Severity of Illness', 'Age', 'Admission_Deposit']]

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    stay_class = stay_dict_rev[prediction]  # Convert numerical class back to human-readable stay duration
    st.success(f"Predicted Stay Class: {stay_class} days")
