#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime  # Importing datetime

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
prices = np.random.uniform(1000, 5000, size=len(dates))

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

# Button to trigger prediction
if st.button("Predict"):
    # Predict prices for the next 30 days
    next_30_days = pd.date_range(start=datetime.today(), periods=30)
    last_price = historical_data.iloc[-1]['price']
    predicted_prices = predict_future_prices(last_price, 30)
    
    # Create a DataFrame for the predicted prices
    prediction_data = pd.DataFrame({'date': next_30_days, 'price': predicted_prices})
    prediction_data.set_index('date', inplace=True)

    # Plot the predicted prices
    st.markdown("<div class='custom-subheader'>Gold Price Predictions for the Next 30 Days</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prediction_data.index, prediction_data['price'], label='Predicted Prices', color='gold')
    ax.set_title('Gold Price Predictions for the Next 30 Days')
    ax.set_xlabel('Date')
    ax.set_ylabel('Gold Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # Display predicted prices
    st.markdown("<div class='custom-subheader'>Predicted Prices</div>", unsafe_allow_html=True)
    st.dataframe(prediction_data)


# In[ ]:




