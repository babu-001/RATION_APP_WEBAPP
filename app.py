import os
import joblib
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "rationshop_model.pkl"))
le_district = joblib.load(os.path.join(BASE_DIR, "le_district.pkl"))

st.set_page_config(page_title="Ration Shop Predictor", layout="centered")

st.title("🛒 Ration Shop Demand Predictor")

st.markdown("Enter details below to predict customer demand")

# INPUTS
district = st.number_input("District Code", step=1)
taluks = st.number_input("Taluks", step=1)
shops = st.number_input("Shops", step=1)
cards = st.number_input("Cards", step=1)

# Prediction button
if st.button("Predict"):

    try:
        features = np.array([[district, taluks, shops, cards]])
        prediction = model.predict(features)[0]

        district_name = le_district.inverse_transform([int(district)])[0]

        st.success(f"District: {district_name}")
        st.info(f"Predicted Customers: {round(float(prediction), 2)}")

    except Exception as e:
        st.error(f"Error: {str(e)}")

