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
from datetime import datetime

# Custom CSS for background and other styling
st.markdown(
    """
    <style>
    /* Background Image */
    .stApp {
        background-image: url('https://milli.gold/mag/wp-content/uploads/2024/07/Gold-Symbol-in-Forex-Main-780x470.webp');
        background-size: cover;
        background-position: center;
        filter: brightness(0.8); /* Darkens the background to make text more visible */
    }
    /* Dark overlay on top of the background */
    .overlay {
        background: rgba(0, 0, 0, 0.4); /* Darkens the background more */
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 0;
    }
    /* Custom Button Styling */
    .stButton button {
        background-color: #333333; /* Darker button color */
        color: white;
        font-size: 20px;
        border-radius: 10px;
    }
    /* Custom Header Text - now with black color */
    .custom-header {
        font-size: 36px;
        color: #000000; /* Changed text color to black */
        text-align: center;
        font-weight: bold;
        z-index: 1; /* Make sure it's above the overlay */
    }
    .custom-subheader {
        font-size: 28px;
        color: #FFD700; /* Gold text for subheader */
        text-align: center;
        z-index: 1;
    }
    /* Semi-transparent background for data frame and input elements */
    .block-container {
        background: rgba(255, 255, 255, 0.8); /* White background with 80% opacity */
        border-radius: 10px;
        padding: 20px;
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with custom styling
st.markdown("<div class='custom-header'>Gold Price Prediction</div>", unsafe_allow_html=True)

# Load your trained Random Forest model
model = joblib.load('forecasting gold prices.pkl')

# Generate historical data from 2016 to 2030
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2030-12-31')

# Create a date range from 2016 to 2023
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate random prices between 1000 and 6000 for the period
np.random.seed(42)
prices = np.random.uniform(1000, 6000, size=len(dates))

# Create a DataFrame with the historical prices
historical_data = pd.DataFrame({'date': dates, 'price': prices})
historical_data.set_index('date', inplace=True)

# Subheader with custom styling
st.markdown("<div class='custom-subheader'>Historical Prices (2016-2030)</div>", unsafe_allow_html=True)

# Create a semi-transparent container for the content
with st.container():
    # Show the historical data
    st.dataframe(historical_data)  

    # Input date
    input_date = st.date_input("Select a date for prediction", datetime.today())

    # Convert input_date to datetime
    input_date = pd.to_datetime(input_date)

    # Custom button for prediction
    if st.button("Predict Gold Price"):
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
            # Prepare input data for prediction
            input_data = np.array([[last_price]])
            st.write(f"Using last price: {last_price}")

            # Make prediction
            prediction = model.predict(input_data)

            # Display predicted price
            st.success(f"Predicted Gold Price for {input_date.date()}: ${prediction[0]:.2f}")

