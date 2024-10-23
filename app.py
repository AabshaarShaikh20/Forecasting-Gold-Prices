#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load your trained Random Forest model
model = joblib.load('forecasting gold prices.pkl')

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://miro.medium.com/v2/resize:fit:1400/1*ulJoL83fzF8Jsg4NRAXWlQ.jpeg');
        background-size: cover;
        background-position: center;
        filter: brightness(0.8);
    }
    .custom-header {
        font-size: 36px;
        color: #000000;
        text-align: center;
        font-weight: bold;
    }
    .custom-subheader {
        font-size: 28px;
        color: #FFD700;
        text-align: center;
    }
    .block-container {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with custom styling
st.markdown("<div class='custom-header'>Gold Price Prediction</div>", unsafe_allow_html=True)

# Generate historical data from 2016 to 2023
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2023-12-31')

# Create a date range from 2016 to 2023
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate random prices between 1200 and 2000 for the period
np.random.seed(42)
prices = np.random.uniform(1200, 2000, size=len(dates))

# Create a DataFrame with the historical prices
historical_data = pd.DataFrame({'date': dates, 'price': prices})
historical_data.set_index('date', inplace=True)

# Subheader with custom styling
st.markdown("<div class='custom-subheader'>Historical Prices (2016-2023)</div>", unsafe_allow_html=True)

# Predict prices for the entire month of October (1st to 31st)
oct_start = pd.to_datetime('2024-10-01')
oct_end = pd.to_datetime('2024-10-31')
oct_dates = pd.date_range(start=oct_start, end=oct_end, freq='D')

# Use the last price from historical data as a starting point
last_price = historical_data.iloc[-1].price

# Predict future prices based on the Random Forest model for October 2024
oct_prices = []
current_price = last_price
for _ in range(31):  # 31 days in October
    input_data = np.array([[current_price]])
    prediction = model.predict(input_data)[0]
    oct_prices.append(prediction)
    current_price = prediction

# Create a DataFrame for October predictions
oct_data = pd.DataFrame({'date': oct_dates, 'price': oct_prices})
oct_data.set_index('date', inplace=True)

# Create future data predictions from 2024 to 2030 (for 30 days)
future_start = pd.to_datetime('2024-01-01')
future_end = pd.to_datetime('2030-12-31')

future_dates = pd.date_range(start=future_start, periods=30, freq='D')
future_prices = []
current_price = last_price  # Reset starting point for future prediction
for _ in range(30):
    input_data = np.array([[current_price]])
    prediction = model.predict(input_data)[0]
    future_prices.append(prediction)
    current_price = prediction

# Create a DataFrame for future predictions
future_data = pd.DataFrame({'date': future_dates, 'price': future_prices})
future_data.set_index('date', inplace=True)

# Concatenate historical, October, and future data
combined_data = pd.concat([historical_data, oct_data, future_data])

# Plot the historical, October, and future data
st.markdown("<div class='custom-subheader'>Gold Price Predictions for October 2024</div>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(oct_data.index, oct_data['price'], label='October 2024 Prices', color='gold')
ax.set_title('Gold Price Predictions for October 2024')
ax.set_xlabel('Date')
ax.set_ylabel('Gold Price (USD)')
ax.legend()

# Show the plot
st.pyplot(fig)

# Historical Data and Future Prices
st.markdown("<div class='custom-subheader'>Historical and Future Prices (2016-2030)</div>", unsafe_allow_html=True)
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(combined_data.index, combined_data['price'], label='Price', color='gold')
ax2.axvline(x=datetime.today(), color='red', linestyle='--', label='Today')  # Mark today's date
ax2.set_title('Gold Price from 2016 to 2030 (Including Predictions)')
ax2.set_xlabel('Date')
ax2.set_ylabel('Gold Price (USD)')
ax2.legend()

# Show the plot
st.pyplot(fig2)

# Display predicted prices for the full month of October 2024
st.markdown("<div class='custom-subheader'>Predicted Prices for October 2024</div>", unsafe_allow_html=True)
st.dataframe(oct_data)

