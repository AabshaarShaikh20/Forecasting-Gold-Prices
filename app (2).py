#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# Function to predict future prices for a given number of days
def predict_future_prices(start_price, days):
    predicted_prices = []
    current_price = start_price
    for _ in range(days):
        input_data = np.array([[current_price]])
        prediction = model.predict(input_data)[0]
        predicted_prices.append(prediction)
        current_price = prediction
    return predicted_prices

# Predict prices for October (1st to 31st)
oct_prices = predict_future_prices(historical_data.iloc[-1]['price'], 31)
oct_dates = pd.date_range(start='2024-10-01', periods=31)
oct_data = pd.DataFrame({'date': oct_dates, 'price': oct_prices})
oct_data.set_index('date', inplace=True)

# Generate yearly predictions from 2016 to 2030
yearly_predictions = {}
for year in range(2016, 2031):
    start_price = historical_data.iloc[-1]['price']  # Starting from the last historical price
    future_dates = pd.date_range(start=f'{year}-01-01', periods=30)
    yearly_predictions[year] = predict_future_prices(start_price, 30)

# Create a DataFrame for yearly predictions
yearly_data = pd.DataFrame({
    'date': [date for year in yearly_predictions for date in pd.date_range(start=f'{year}-01-01', periods=30)],
    'price': [price for prices in yearly_predictions.values() for price in prices]
})
yearly_data.set_index('date', inplace=True)

# Plot the October predictions
st.markdown("<div class='custom-subheader'>Gold Price Predictions for October 2024</div>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(oct_data.index, oct_data['price'], label='October 2024 Prices', color='gold')
ax.set_title('Gold Price Predictions for October 2024')
ax.set_xlabel('Date')
ax.set_ylabel('Gold Price (USD)')
ax.legend()
st.pyplot(fig)

# Plot yearly predictions
st.markdown("<div class='custom-subheader'>Gold Price Predictions (2016-2030)</div>", unsafe_allow_html=True)
fig2, ax2 = plt.subplots(figsize=(10, 6))
for year in yearly_predictions:
    future_dates = pd.date_range(start=f'{year}-01-01', periods=30)
    ax2.plot(future_dates, yearly_predictions[year], label=f'{year} Prices')

ax2.axvline(x=datetime.today(), color='red', linestyle='--', label='Today')  # Mark today's date
ax2.set_title('Gold Price Predictions from 2016 to 2030')
ax2.set_xlabel('Date')
ax2.set_ylabel('Gold Price (USD)')
ax2.legend()
st.pyplot(fig2)

# Display predicted prices for October 2024
st.markdown("<div class='custom-subheader'>Predicted Prices for October 2024</div>", unsafe_allow_html=True)
st.dataframe(oct_data)

# Display yearly predictions
st.markdown("<div class='custom-subheader'>Predicted Prices from 2016 to 2030</div>", unsafe_allow_html=True)
st.dataframe(yearly_data)


# In[ ]:





# In[ ]:





# In[ ]:




