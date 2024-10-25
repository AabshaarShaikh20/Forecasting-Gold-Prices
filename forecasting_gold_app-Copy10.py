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

# Generate historical data from 2016 to 2023
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2023-12-31')

# Create a date range from 2016 to 2023
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate random prices between 1000 and 5000 for the period
np.random.seed(42)
prices = np.random.uniform(1000, 5000, size=len(dates))

# Create a DataFrame with the historical prices
historical_data = pd.DataFrame({'date': dates, 'price': prices})
historical_data.set_index('date', inplace=True)

# Display historical data first
st.markdown("<div class='custom-subheader'>Historical Prices (2016 - 2023)</div>", unsafe_allow_html=True)
st.dataframe(historical_data)

# Function to predict future prices with unique randomness per year
def predict_unique_future_prices(start_price, days, variation_factor=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set unique seed for each year

    predicted_prices = []
    current_price = start_price
    for _ in range(days):
        input_data = np.array([[current_price]])
        prediction = model.predict(input_data)[0]
        
        # Introduce random fluctuation to simulate realistic variations
        random_variation = np.random.uniform(-variation_factor, variation_factor) * prediction
        adjusted_prediction = prediction + random_variation
        
        predicted_prices.append(adjusted_prediction)
        current_price = adjusted_prediction
    return predicted_prices

# Let the user choose the year they want predictions for
selected_year = st.selectbox('Select a year for prediction', [2024, 2025, 2026, 2027, 2028, 2029, 2030])

# Button to trigger prediction
if st.button(f"Predict 30 Days for {selected_year}"):
    # Predict prices for the first 30 days of the selected year
    start_date_of_year = pd.to_datetime(f'{selected_year}-01-01')
    next_30_days = pd.date_range(start=start_date_of_year, periods=30)
    
    # Set a unique seed for each year to ensure different random variations
    unique_seed = selected_year  # Use the selected year as the seed for distinct results
    last_price = historical_data.iloc[-1]['price']  # Start from the last known price
    
    # Generate predictions with unique random variations for each year
    predicted_prices = predict_unique_future_prices(last_price, 30, seed=unique_seed)
    
    # Create a DataFrame for the predicted prices
    prediction_data = pd.DataFrame({'date': next_30_days, 'price': predicted_prices})
    prediction_data.set_index('date', inplace=True)

    # Display the total predicted price over the 30 days
    total_price = np.sum(predicted_prices)
    st.markdown(f"<div class='custom-subheader'>Total Predicted Price for 30 Days in {selected_year}: {total_price:.2f} USD</div>", unsafe_allow_html=True)

    # Plot the predicted prices
    st.markdown(f"<div class='custom-subheader'>Gold Price Predictions for 30 Days in {selected_year}</div>", unsafe_allow_html=True)
    
    # Plot the predictions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prediction_data.index, prediction_data['price'], label='Predicted Prices', color='gold')
    ax.set_title(f'Gold Price Predictions for the Next 30 Days in {selected_year}')
    
    # Rotate x-axis labels and set the ticks for better readability
    ax.set_xlabel('Date')
    ax.set_xticks(prediction_data.index[::5])  # Show every 5th date
    ax.set_xticklabels(prediction_data.index.strftime('%Y-%m-%d')[::5], rotation=45)
    
    ax.set_ylabel('Gold Price (USD)')
    ax.legend()
    
    # Display the graph
    st.pyplot(fig)
