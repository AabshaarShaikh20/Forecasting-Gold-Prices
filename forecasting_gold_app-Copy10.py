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

# Display historical data
st.markdown("<div class='custom-subheader'>Historical Prices (2016 - 2023)</div>", unsafe_allow_html=True)
st.dataframe(historical_data)

# Function to predict future prices with unique randomness
def predict_unique_future_prices(start_price, days, variation_factor=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set unique seed for distinct predictions

    predicted_prices = []
    current_price = start_price
    for _ in range(days):
        input_data = np.array([[current_price]])
        prediction = model.predict(input_data)[0]
        
        # Add random fluctuation
        random_variation = np.random.uniform(-variation_factor, variation_factor) * prediction
        adjusted_prediction = prediction + random_variation
        
        predicted_prices.append(adjusted_prediction)
        current_price = adjusted_prediction
    return predicted_prices

# Default October Prediction Display
oct_start_date = pd.to_datetime('2024-10-01')
next_30_days_oct = pd.date_range(start=oct_start_date, periods=30)
last_price_oct = historical_data.iloc[-1]['price']
predicted_prices_oct = predict_unique_future_prices(last_price_oct, 30, seed=2024)

# Display Default October Prediction Graph
st.markdown("<div class='custom-subheader'>Gold Price Predictions for October 2024 (Default)</div>", unsafe_allow_html=True)
prediction_data_oct = pd.DataFrame({'date': next_30_days_oct, 'price': predicted_prices_oct})
prediction_data_oct.set_index('date', inplace=True)

fig_oct, ax_oct = plt.subplots(figsize=(10, 6))
ax_oct.plot(prediction_data_oct.index, prediction_data_oct['price'], label='Predicted Prices (October)', color='gold')
ax_oct.set_title('Gold Price Predictions for October 2024')
ax_oct.set_xlabel('Date')
ax_oct.set_ylabel('Gold Price (USD)')
ax_oct.legend()
st.pyplot(fig_oct)

# Year Selection for Future Predictions (Starting from 2025)
selected_year = st.selectbox('Select a year for prediction (starting from 2025)', [2025, 2026, 2027, 2028, 2029, 2030])

# Button to trigger yearly prediction
if st.button(f"Predict 30 Days for {selected_year}"):
    # Predict prices for the first 30 days of the selected year
    start_date_of_year = pd.to_datetime(f'{selected_year}-01-01')
    next_30_days_year = pd.date_range(start=start_date_of_year, periods=30)
    
    # Set a unique seed for each year
    unique_seed = selected_year
    last_price_year = historical_data.iloc[-1]['price']
    
    # Generate predictions with unique randomness
    predicted_prices_year = predict_unique_future_prices(last_price_year, 30, seed=unique_seed)
    
    # Create DataFrame for yearly prediction
    prediction_data_year = pd.DataFrame({'date': next_30_days_year, 'price': predicted_prices_year})
    prediction_data_year.set_index('date', inplace=True)

    # Display total predicted price for selected year
    total_price_year = np.sum(predicted_prices_year)
    st.markdown(f"<div class='custom-subheader'>Total Predicted Price for 30 Days in {selected_year}: {total_price_year:.2f} USD</div>", unsafe_allow_html=True)

    # Plot yearly predictions
    st.markdown(f"<div class='custom-subheader'>Gold Price Predictions for 30 Days in {selected_year}</div>", unsafe_allow_html=True)
    
    fig_year, ax_year = plt.subplots(figsize=(10, 6))
    ax_year.plot(prediction_data_year.index, prediction_data_year['price'], label=f'Predicted Prices ({selected_year})', color='gold')
    ax_year.set_title(f'Gold Price Predictions for 30 Days in {selected_year}')
    ax_year.set_xlabel('Date')
    ax_year.set_ylabel('Gold Price (USD)')
    ax_year.legend()
    st.pyplot(fig_year)
