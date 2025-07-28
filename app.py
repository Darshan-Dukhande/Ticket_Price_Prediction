import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and training feature list
model = joblib.load("best_random_forest_model.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Flight Price Predictor", layout="centered")
st.title("✈️ Flight Price Prediction App")

st.markdown("Enter flight details to estimate the ticket price.")

# User inputs
airline = st.selectbox("Airline", ['IndiGo', 'Air India', 'SpiceJet', 'Other'])
source = st.selectbox("Source", ['Delhi', 'Kolkata', 'Mumbai', 'Chennai'])
destination = st.selectbox("Destination", ['Cochin', 'Delhi', 'New Delhi', 'Banglore'])
stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])
day = st.number_input("Journey Day", 1, 31, value=1)
month = st.number_input("Journey Month", 1, 12, value=1)
year = 2019  # fixed as dataset is from 2019
dep_hour = st.slider("Departure Hour", 0, 23, value=10)
dep_min = st.slider("Departure Minute", 0, 59, value=0)
arr_hour = st.slider("Arrival Hour", 0, 23, value=13)
arr_min = st.slider("Arrival Minute", 0, 59, value=0)
duration = st.number_input("Duration (in minutes)", 30, 1440, value=180)
additional_info = st.selectbox("Additional Info", ['No info', 'Other'])

# Manual One-Hot Encoding
input_dict = {
    'Date': day,
    'Month': month,
    'Year': year,
    'Arrival_Time_Hour': arr_hour,
    'Arrival_Time_Min': arr_min,
    'Dep_Time_Hour': dep_hour,
    'Dep_Time_Min': dep_min,
    'Duration': duration,
    'Total_Stops': stops
}

# One-hot encoded fields (set all possible values)
all_dummies = {
    'Airline_Air India': 0,
    'Airline_IndiGo': 0,
    'Airline_SpiceJet': 0,
    'Airline_Other': 0,
    'Source_Delhi': 0,
    'Source_Kolkata': 0,
    'Source_Mumbai': 0,
    'Destination_Cochin': 0,
    'Destination_Delhi': 0,
    'Destination_New Delhi': 0,
    'Additional_Info_Other': 0
}

# Activate the selected one
if f'Airline_{airline}' in all_dummies:
    all_dummies[f'Airline_{airline}'] = 1
if f'Source_{source}' in all_dummies:
    all_dummies[f'Source_{source}'] = 1
if f'Destination_{destination}' in all_dummies:
    all_dummies[f'Destination_{destination}'] = 1
if f'Additional_Info_{additional_info}' in all_dummies:
    all_dummies[f'Additional_Info_{additional_info}'] = 1

# Merge numeric and encoded inputs
input_data = {**input_dict, **all_dummies}
input_df = pd.DataFrame([input_data])

# Align input with model features
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_features]

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💸 Estimated Flight Price: ₹{int(prediction):,}")
