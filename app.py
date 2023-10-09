import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the SVM model
with open('svm_model.pickle', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Streamlit app header
st.title('Breast Cancer Classification')

# Sidebar with user input
st.sidebar.header('User Input')

# Create input fields for all the features used in the model
features = []
for i in range(31):
    feature = st.sidebar.slider(
        f'Feature {i + 1}', min_value=0.0, max_value=10.0, value=5.0)
    features.append(feature)

# Create a button to make predictions
if st.sidebar.button('Predict'):
    # Preprocess the user input (standardization)
    user_input = np.array([features])
    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform(user_input)

    # Make predictions using the SVM model
    prediction = svm_model.predict(user_input_scaled)

    # Display the prediction result
    if prediction[0] == 'B':
        st.sidebar.success('Prediction: Benign (B)')
    else:
        st.sidebar.error('Prediction: Malignant (M)')