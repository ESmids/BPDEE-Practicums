import streamlit as st
import numpy as np
from datetime import timedelta
import joblib
import sqlite3
from datetime import datetime
import pandas as pd

def generate_synthetic_data(n=50):
    np.random.seed(42)

    data = pd.DataFrame({
        "X1 transaction date": np.random.normal(2013, 0.3, n),
        "X2 house age": np.random.normal(20, 10, n).clip(0, 50),
        "X3 distance to the nearest MRT station": np.random.normal(1000, 500, n).clip(0, 5000),
        "X4 number of convenience stores": np.random.randint(0, 10, n),
        "X5 latitude": np.random.normal(24.97, 0.01, n),
        "X6 longitude": np.random.normal(121.54, 0.01, n)
    })

    return data

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

    synthetic_data = generate_synthetic_data(50)

    base_time = datetime.now()

    for i, row in synthetic_data.iterrows():
        input_df = pd.DataFrame([row])
        scaled = scaler.transform(input_df.values)
        pred = model.predict(scaled)[0]

        # Elke datapunt iets later in de tijd
        timestamp = base_time + timedelta(minutes=i)

        save_prediction(input_df, float(pred), timestamp)

    st.success("50 synthetic predictions toegevoegd")