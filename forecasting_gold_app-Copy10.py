import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Load the trained model
model = joblib.load('forecasting gold prices.pkl')

# Custom CSS styling
st.markdown(
    """
    <style>
    .stApp { background-image: url('https://miro.medium.com/v2/resize:fit:1400/1*ulJoL83fzF8Jsg4NRAXWlQ.jpeg');
             background-size: cover; background-position: center; filter: brightness(0.8); }
    .custom-header { font-size: 36px; color: #000; text-align: center; font-weight: bold; }
    .custom-subheader { font-size: 28px; color: #FFD700; text-align: center; }
    .block-container { background: rgba(255, 255, 255, 0.8); border-radius: 10px; padding: 20px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with custom styling
st.markdown("<div class='custom-header'>Gold Price Prediction</div>", unsafe_allow_html=True)

# Set up year selection and custom parameter sliders
selected_year = st.selectbox('Select a year for prediction', [2024, 2025, 2026, 2027, 2028, 2029, 2030])
variation_factor = st.slider("Variation Factor (for market fluctuations)", 0.01, 0.1, 0.05)

# Generate historical data (for display and initial calculations)
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2023-12-31')
dates = pd.date_range(start=start_date, end=end_date, freq='D')
prices = np.random.uniform(1000, 5000, size=len(dates))
historical_data = pd.DataFrame({'date': dates, 'price': prices})
historical_data.set_index('date', inplace=True)

st.markdown("<div class='custom-subheader'>Historical Prices (2016 - 2023)</div>", unsafe_allow_html=True)
st.dataframe(historical_data)

# Custom starting price for each year to simulate yearly variations
starting_price_dict = {
    year: historical_data.iloc[-1]['price'] * np.random.uniform(1 - (0.1 * (year - 2023)), 1 + (0.1 * (year - 2023)))
    for year in range(2024, 2031)
}

# Prediction function with confidence intervals and moving average
def predict_future_prices(start_price, days, variation_factor=0.05):
    predicted_prices = []
    lower_bound, upper_bound = [], []
    current_price = start_price
    for _ in range(days):
        input_data = np.array([[current_price]])
        prediction = model.predict(input_data)[0]
        
        random_variation = np.random.uniform(-variation_factor, variation_factor) * prediction
        adjusted_prediction = prediction + random_variation
        confidence_interval = prediction * variation_factor  # simple confidence interval
        
        predicted_prices.append(adjusted_prediction)
        lower_bound.append(adjusted_prediction - confidence_interval)
        upper_bound.append(adjusted_prediction + confidence_interval)
        
        current_price = adjusted_prediction
    
    return predicted_prices, lower_bound, upper_bound

# Prediction and plotting based on selected year and parameters
if st.button(f"Predict 30 Days for {selected_year}"):
    start_price = starting_price_dict[selected_year]
    predicted_prices, lower_bound, upper_bound = predict_future_prices(start_price, 30, variation_factor)
    next_30_days = pd.date_range(start=f"{selected_year}-01-01", periods=30)
    
    prediction_data = pd.DataFrame({
        'date': next_30_days,
        'price': predicted_prices,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })
    prediction_data.set_index('date', inplace=True)

    # Moving Average Calculation for smooth trend display
    prediction_data['7-day MA'] = prediction_data['price'].rolling(window=7).mean()

    # Plotting
    st.markdown(f"<div class='custom-subheader'>Gold Price Predictions for 30 Days in {selected_year}</div>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prediction_data.index, prediction_data['price'], label='Predicted Price', color='gold')
    ax.fill_between(prediction_data.index, prediction_data['lower_bound'], prediction_data['upper_bound'], color='lightgoldenrodyellow', alpha=0.4, label='Confidence Interval')
    ax.plot(prediction_data.index, prediction_data['7-day MA'], label='7-Day Moving Average', linestyle='--', color='orange')
    ax.set_title(f'Gold Price Predictions with Confidence Interval and Moving Average for {selected_year}')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Gold Price (USD)')
    ax.set_xticks(prediction_data.index[::5])  # Show every 5th date
    ax.set_xticklabels(prediction_data.index.strftime('%Y-%m-%d')[::5], rotation=45)
    ax.legend()

    st.pyplot(fig)
