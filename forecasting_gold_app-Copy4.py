#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import joblib

# try:
#     model = joblib.load('forecasting gold prices.pkl')
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"An error occurred while loading the model: {e}")


# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

# Load your trained Random Forest model
model = joblib.load('forecasting gold prices.pkl')

# Generate sample historical data
# Create a date range for the last 30 days
dates = pd.date_range(end=datetime.today(), periods=30).to_list()
# Generate random prices around a base price, say $1800
np.random.seed(42)  # For reproducibility
prices = np.random.uniform(1000, 6000, size=len(dates))

# Create a DataFrame
historical_data = pd.DataFrame({'date': dates, 'price': prices})
historical_data.set_index('date', inplace=True)

# Title
st.title("Gold Price Prediction")

# Show the historical data
st.subheader("Historical Prices")
st.write(historical_data)

# Input date
input_date = st.date_input("Select a date for prediction", datetime.today())

# Convert input_date to datetime
input_date = pd.to_datetime(input_date)

# Predict price based on selected date
if st.button("Predict"):
    # Check if the selected date is available in historical data
    if input_date in historical_data.index:
        last_price = historical_data.loc[input_date].price
    else:
        # Get the closest previous date's price
        previous_dates = historical_data[historical_data.index < input_date]
        if not previous_dates.empty:
            last_price = previous_dates.iloc[-1].price
        else:
            st.error("No available data prior to this date.")
            last_price = None

    # If we have a last price, prepare the input for the model
    if last_price is not None:
        # Prepare input data for prediction (change according to your model's needs)
        input_data = np.array([[last_price]])  # Assuming the model uses the last price as input
        st.write(f"Using last price: {last_price}")

        # Make prediction
        prediction = model.predict(input_data)

        # Show predicted price
        st.write(f"Predicted Gold Price for {input_date.date()}: ${prediction[0]:.2f}")

