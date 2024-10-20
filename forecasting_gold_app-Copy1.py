#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import joblib

# try:
#     model = joblib.load('forecasting gold prices.pkl')
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"An error occurred while loading the model: {e}")


# In[2]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

# Load your trained Random Forest model
model = joblib.load('forecasting gold prices.pkl')

# Title
st.title("Gold Price Prediction with Random Forest")

# Input date
input_date = st.date_input("Select a date for prediction", datetime.today())

# Load your historical data (ensure it's preprocessed)
# For example, replace 'historical_data.csv' with your actual dataset
historical_data = pd.read_csv(r'C:\Users\Aabshaar\Downloads\Gold_data (1).csv', parse_dates=['date'])
historical_data.set_index('date', inplace=True)

# Prepare input features based on the input date
if st.button("Predict"):
    # Get the last available price before the input date
    if input_date in historical_data.index:
        last_price = historical_data.loc[input_date].price
    else:
        # Find the closest previous date if the exact date is not in the dataset
        previous_dates = historical_data[historical_data.index < input_date]
        if not previous_dates.empty:
            last_price = previous_dates.iloc[-1].price
        else:
            st.error("No available data prior to this date.")
            last_price = None

    # If we have a last price, prepare the input for the model
    if last_price is not None:
        # Create input data (using previous prices as features)
        input_data = np.array([[last_price]])  # Change based on your model input needs
        st.write(f"Using last price: {last_price}")

        # Print the shape of the input data for debugging
        st.write(f"Input data shape: {input_data.shape}")
        print(input_data.shape)

        # Make prediction
        prediction = model.predict(input_data)

        st.write(f"Predicted Gold Price for {input_date}: {prediction[0]:.2f}")

