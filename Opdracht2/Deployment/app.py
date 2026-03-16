import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Titel
st.title("Real Estate Price Predictor")
st.write("Voorspel de woningprijs per unit area met een getraind regressiemodel.")

# Model en scaler laden
model = joblib.load("regression_model.pkl")
scaler = joblib.load("regression_scaler.pkl")

st.header("Voer woninggegevens in")

# Inputs
transaction_date = st.number_input("Transaction Date", value=2013.5)
house_age = st.number_input("House Age (years)", value=10.0)
distance_mrt = st.number_input("Distance to MRT station (meters)", value=300.0)
stores = st.number_input("Number of convenience stores", value=5)
latitude = st.number_input("Latitude", value=24.97)
longitude = st.number_input("Longitude", value=121.54)

# Dataframe maken
input_data = pd.DataFrame({
    "X1 transaction date": [transaction_date],
    "X2 house age": [house_age],
    "X3 distance to the nearest MRT station": [distance_mrt],
    "X4 number of convenience stores": [stores],
    "X5 latitude": [latitude],
    "X6 longitude": [longitude]
})

# Scaling
input_scaled = scaler.transform(input_data.values)

# Predictie
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)

    st.subheader("Voorspelde woningprijs per unit area:")
    st.write(f"{prediction[0]:.2f}")