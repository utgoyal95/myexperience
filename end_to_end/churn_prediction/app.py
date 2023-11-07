import pandas as pd
import streamlit as st
import numpy as np
import pickle

# Load the saved components:
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


st.title("Telecom Churn Prediction App")

st.caption("This app predicts if a customer will churn on the inputs with 80% accuracy.")

# Create the input fields
input_data = {}


input_data['gender'] = st.selectbox("Gender", ['Male', 'Female'])
input_data['Partner'] = st.selectbox("Partner", ['Yes', 'No'])

input_data['tenure'] =  st.number_input("Number of Years on service", step=1)

input_data['InternetService'] = st.selectbox("Do they use internet service?", ['DSL','Fiber optic','No'])
input_data['Contract'] = st.selectbox("Contract Type", ['Month-to-month','One year','Two year'])
input_data['PaperlessBilling'] = st.selectbox("Do they have paperless billing?", ['Yes', 'No'])
input_data['PaymentMethod'] = st.selectbox("Partner", ['Electronic check','Mailed check',
'Bank transfer (automatic)','Credit card (automatic)'])

input_data['MonthlyCharges'] =  st.number_input("Monthly Charges", step=1)
input_data['TotalCharges'] =  st.number_input("Total Charges", step=1)

out_data = {1:'They are likely to churn.', 0:'They are not likely to churn.'}
  # Create a button to make a prediction
if st.button("Predict"):
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Make predictions using the trained model
    predictions = model.predict(input_df)[0]

    # Display the predicted sales value to the user:
    st.write(out_data[predictions])