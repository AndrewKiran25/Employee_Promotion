import os
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(
    repo_id="Andrew2505/Employee-Promotion",
    filename="best_model_v1.joblib",
    repo_type="model"
)

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Employee Promotion Classification App")
st.write("The Employee Promotion Classification App is an internal tool for employee's that classify whether staff's will be promoted or not.")
st.write("Kindly enter the employee details to check whether they are likely to promoted or not.")

# Collect user input
no_of_trainings = st.number_input("Number of Trainings", min_value=0, value=3)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
previous_year_rating = st.number_input("Previous Year Rating", min_value=1, max_value=5, value=3)
length_of_service = st.number_input("Length of Service", min_value=0, value=5)
awards_won_str = st.selectbox("Awards Won?", ["Yes", "No"])
avg_training_score = st.number_input("Average Training Score", min_value=0, max_value=1000, value=75)
department = st.selectbox("Department", ('Sales & Marketing', 'Operations', 'Technology', 'Analytics',
       'R&D', 'Procurement', 'Finance', 'HR', 'Legal'))
region = st.selectbox('region', ('region_7', 'region_22', 'region_19', 'region_23', 'region_26',
       'region_2', 'region_20', 'region_34', 'region_1', 'region_4',
       'region_29', 'region_31', 'region_15', 'region_14', 'region_11',
       'region_5', 'region_28', 'region_17', 'region_13', 'region_16',
       'region_25', 'region_10', 'region_27', 'region_30', 'region_12',
       'region_21', 'region_32', 'region_6', 'region_33', 'region_8',
       'region_24', 'region_3', 'region_9', 'region_18'))
gender = st.selectbox('gender', ('m', 'f'))
recruitment_channel = st.selectbox('recruitment_channel', ('sourcing', 'other', 'referred'))

# Convert 'awards_won' to numerical format (0 or 1)
awards_won = 1 if awards_won_str == "Yes" else 0

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'no_of_trainings': no_of_trainings,
    'age': age,
    'previous_year_rating': previous_year_rating,
    'length_of_service': length_of_service,
    'awards_won': awards_won,
    'avg_training_score': avg_training_score,
    'department': department,
    'region': region,
    'gender': gender,
    'recruitment_channel': recruitment_channel
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Promoted" if prediction == 1 else "Not Promoted"
    st.write(f"Based on the information provided, the employee is likely to {result}.")
