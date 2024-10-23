#!/usr/bin/env python
# coding: utf-8

# In[12]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

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

# Create a placeholder for predictions
prediction_data = pd.DataFrame(columns=['date', 'price'])

# Display the initial empty table
st.markdown("<div class='custom-subheader'>Predicted Prices for the Next 30 Days</div>", unsafe_allow_html=True)
st.dataframe(prediction_data)

# Button to trigger prediction
if st.button("Predict"):
    # Predict prices for the next 30 days
    next_30_days = pd.date_range(start=datetime.today(), periods=30)
    last_price = 1500  # You can start from a specific price or use historical data
    predicted_prices = []

    # Generate predictions
    for _ in range(30):
        input_data = np.array([[last_price]])
        prediction = model.predict(input_data)[0]
        predicted_prices.append(prediction)
        last_price = prediction  # Update last price for next prediction

    # Create a DataFrame for the predicted prices
    prediction_data = pd.DataFrame({'date': next_30_days, 'price': predicted_prices})
    prediction_data.set_index('date', inplace=True)

    # Calculate total predicted price for 30 days
    total_predicted_price = prediction_data['price'].sum()

    # Display total predicted price
    st.markdown(f"<h3>Total Predicted Price for 30 Days: {total_predicted_price:.2f} USD</h3>", unsafe_allow_html=True)

    # Plot the predicted prices
    st.markdown("<div class='custom-subheader'>Gold Price Predictions for the Next 30 Days</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prediction_data.index, prediction_data['price'], label='Predicted Prices', color='gold', marker='o')
    ax.set_title('Gold Price Predictions for the Next 30 Days')
    ax.set_xlabel('Date')
    ax.set_ylabel('Gold Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # Display predicted prices
    st.markdown("<div class='custom-subheader'>Predicted Prices Table</div>", unsafe_allow_html=True)
    st.dataframe(prediction_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




