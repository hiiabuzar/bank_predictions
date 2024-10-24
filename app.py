import streamlit as st
import pickle
import numpy as np

# Load the saved model, scaler, and label encoders
with open('bank_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Define the input form in Streamlit
st.title("Bank Term Deposit Subscription Prediction")

# Define input fields for user to fill (excluding 'contact', 'duration', and 'poutcome')
job = st.selectbox('Job', list(label_encoders['job'].classes_))
marital = st.selectbox('Marital Status', list(label_encoders['marital'].classes_))
education = st.selectbox('Education', list(label_encoders['education'].classes_))
default = st.selectbox('Default Credit', list(label_encoders['default'].classes_))
housing = st.selectbox('Housing Loan', list(label_encoders['housing'].classes_))
loan = st.selectbox('Personal Loan', list(label_encoders['loan'].classes_))
age = st.slider('Age', min_value=18, max_value=100)
balance = st.number_input('Balance ($)')

# Predict button
if st.button('Predict'):
    # Prepare input data for prediction (excluding 'contact', 'duration', and 'poutcome')
    input_data = np.array([[age, balance, 
                            label_encoders['job'].transform([job])[0],
                            label_encoders['marital'].transform([marital])[0],
                            label_encoders['education'].transform([education])[0],
                            label_encoders['default'].transform([default])[0],
                            label_encoders['housing'].transform([housing])[0],
                            label_encoders['loan'].transform([loan])[0]]])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict using the loaded model
    prediction = model.predict(input_data_scaled)

    # Display the result
    result = "Yes" if prediction[0] == 1 else "No"
    st.write(f"The client will subscribe to a term deposit: {result}")
