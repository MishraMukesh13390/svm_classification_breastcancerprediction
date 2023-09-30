# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:34:18 2023

@author: ADMIN
"""
!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the SVM model
with open('G:/360digitmg/PROJECT WORK/Mukesh Mishra/breast cancer model/svm_model.pickle', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Streamlit app header
st.title('Breast Cancer Classification')

# Sidebar with user input
st.sidebar.header('User Input')

# Create input fields for features
feature1 = st.sidebar.slider('Feature 1', min_value=0.0, max_value=10.0, value=5.0)
feature2 = st.sidebar.slider('Feature 2', min_value=0.0, max_value=10.0, value=5.0)
# Add more feature sliders as needed

# Create a button to make predictions
if st.sidebar.button('Predict'):
    # Preprocess the user input (standardization)
    user_input = np.array([[feature1, feature2]])  # Add more features as needed
    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform(user_input)

    # Make predictions using the SVM model
    prediction = svm_model.predict(user_input_scaled)

    # Display the prediction result
    if prediction[0] == 'B':
        st.sidebar.success('Prediction: Benign (B)')
    else:
        st.sidebar.error('Prediction: Malignant (M)')

# Add more content to your Streamlit app as needed
