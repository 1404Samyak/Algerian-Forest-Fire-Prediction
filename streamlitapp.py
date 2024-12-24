import streamlit as st
import pickle
import numpy as np

# Load the model and scaler from pickle files
with open('algerian_forest_prediction.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Algerian Forest Fire Prediction")
st.write("Enter the following details to predict the fire occurrence:")

# Input fields
temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
relative_humidity = st.number_input("Relative Humidity (%)", min_value=0.0, step=0.1)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
ffmc = st.number_input("FFMC (Fine Fuel Moisture Code)", min_value=0.0, step=0.1)
dmc = st.number_input("DMC (Duff Moisture Code)", min_value=0.0, step=0.1)
dc = st.number_input("DC (Drought Code)", min_value=0.0, step=0.1)
isi = st.number_input("ISI (Initial Spread Index)", min_value=0.0, step=0.1)
bui = st.number_input("BUI (Build Up Index)", min_value=0.0, step=0.1)
fire_class = st.selectbox("Fire Class", ["fire", "no_fire"])
region = st.selectbox("Region", ["Sidi-Bel Abbes Region", "Belagiah region"])
fire_class_value = 1 if fire_class == "fire" else 0
region_value = 1 if region == "Sidi-Bel Abbes Region" else 0

# Button to predict
if st.button("Predict"):
    # Prepare input data
    arr = [[
        temperature, relative_humidity, wind_speed, rainfall, ffmc, dmc, dc, isi, fire_class_value, region_value
    ]]
    X_test = np.array(arr)
    X_test = scaler.transform(X_test)  # Apply scaling

    # Predict
    prediction = model.predict(X_test)[0]

    # Display prediction
    st.write("The predicted fire wall index is",prediction)
