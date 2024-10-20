#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import joblib

# try:
#     model = joblib.load('forecasting gold prices.pkl')
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"An error occurred while loading the model: {e}")


# In[ ]:


import streamlit as st
import numpy as np
import joblib

model = joblib.load('forecasting gold prices.pkl')

# Title
st.title("Gold Price Prediction with Random Forest")

# Input feature (adjust based on your model's requirements)
st.sidebar.header("Input Feature")
feature = st.sidebar.number_input("Enter the Feature Value")  # Adjust feature input

# Prediction
if st.sidebar.button("Predict"):
    # Prepare the input data (as a 2D array)
    input_data = np.array([[feature]])  # Use only the one feature
    prediction = model.predict(input_data)
    
    st.write(f"Predicted Gold Price: {prediction[0]:.2f}")

