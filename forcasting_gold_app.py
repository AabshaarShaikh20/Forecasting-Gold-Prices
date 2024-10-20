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
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib  # for loading the model

# Load your trained Random Forest model
model = joblib.load('forecasting gold prices.pkl')  # Ensure the model is saved before this

# Title
st.title("Gold Price Prediction with Random Forest")

# Input features (customize based on your model's requirements)
st.sidebar.header("Input Features")
feature1 = st.sidebar.number_input("Feature 1")
feature2 = st.sidebar.number_input("Feature 2")
# Add more input fields as necessary...

# Prediction
if st.sidebar.button("Predict"):
    # Prepare the input data
    input_data = np.array([[feature1, feature2]])  # Adjust based on your features
    prediction = model.predict(input_data)
    
    st.write(f"Predicted Gold Price: {prediction[0]:.2f}")

# Optional: Add more features to visualize or explain predictions

