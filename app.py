import streamlit as st
import numpy as np
import pandas as pd

# Simulated AI model for CO₂ absorption
def predict_absorption(temp, humidity, co2):
    return round(100 - (co2 / 8) + (temp / 10) - (humidity / 20), 2)

st.title("🌿 AI-Powered CO₂ Absorption Monitoring")

st.write("Use the sliders to simulate environmental conditions and predict CO₂ absorption efficiency.")

temp = st.slider("🌡️ Temperature (°C)", 10, 50, 25)
humidity = st.slider("💧 Humidity (%)", 10, 100, 50)
co2 = st.slider("🛑 CO₂ Level (ppm)", 300, 1000, 500)

efficiency = predict_absorption(temp, humidity, co2)

st.subheader(f"✅ **Predicted CO₂ Absorption Efficiency:** {efficiency}%")

st.info("This AI model predicts the efficiency of CO₂ absorption based on temperature, humidity, and CO₂ levels.")
