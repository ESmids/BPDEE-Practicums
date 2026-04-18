import streamlit as st
import numpy as np
from datetime import timedelta
import joblib
import sqlite3
from datetime import datetime
import pandas as pd

# Database initialiseren
def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            transaction_date REAL,
            house_age REAL,
            distance_mrt REAL,
            stores REAL,
            latitude REAL,
            longitude REAL,
            prediction REAL
        )
    """)

    conn.commit()
    conn.close()


# Opslaan
def save_prediction(data, pred, timestamp=None):
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()

    if timestamp is None:
        timestamp = datetime.now()

    c.execute("""
        INSERT INTO predictions (
            timestamp, transaction_date, house_age,
            distance_mrt, stores, latitude, longitude, prediction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        data["X1 transaction date"].iloc[0],
        data["X2 house age"].iloc[0],
        data["X3 distance to the nearest MRT station"].iloc[0],
        data["X4 number of convenience stores"].iloc[0],
        data["X5 latitude"].iloc[0],
        data["X6 longitude"].iloc[0],
        pred
    ))

    conn.commit()
    conn.close()

# Data ophalen
def load_data():
    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()
    return df

init_db()

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

    # Opslaan in database
    save_prediction(input_data, float(prediction[0]))

st.header("History")

df_history = load_data()

if not df_history.empty:
    df_history["timestamp"] = pd.to_datetime(df_history["timestamp"])

    df_history = df_history.sort_values("timestamp")

    st.line_chart(
        df_history.set_index("timestamp")["prediction"]
    )
else:
    st.write("Nog geen data beschikbaar")

if st.button("Simulate Predictions"):

    synthetic_data = pd.read_csv("synthetic_data.csv")

    synthetic_data = synthetic_data[[
        "X1 transaction date",
        "X2 house age",
        "X3 distance to the nearest MRT station",
        "X4 number of convenience stores",
        "X5 latitude",
        "X6 longitude"
    ]]

    base_time = datetime.now()

    for i, row in synthetic_data.iterrows():
        input_df = pd.DataFrame([row])
        scaled = scaler.transform(input_df.values)
        pred = model.predict(scaled)[0]

        timestamp = base_time + timedelta(minutes=i)

        save_prediction(input_df, float(pred), timestamp)

    st.success("Synthetic dataset verwerkt")