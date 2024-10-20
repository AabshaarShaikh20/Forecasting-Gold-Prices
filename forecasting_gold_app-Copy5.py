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

# Generate historical data from 2016 to 2023
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2030-12-31')

# Create a date range from 2016 to 2023
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate random prices between 1200 and 2000 for the period
np.random.seed(42)  # For reproducibility
prices = np.random.uniform(1000, 5000, size=len(dates))

# Create a DataFrame with the historical prices
historical_data = pd.DataFrame({'date': dates, 'price': prices})
historical_data.set_index('date', inplace=True)

# Title
st.title("Gold Price Prediction")

# Show the historical data (big dataset from 2016-2023)
st.subheader("Historical Prices (2016-2030)")
st.dataframe(historical_data)  # This will display the full dataset

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

