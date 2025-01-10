import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load model and scaler
with open("rf.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

data = pd.read_csv('cuaca.csv')


# Header
st.markdown("<h1 style='color:#6A80B9; text-align:center;'>Prediksi Cuaca</h1>", unsafe_allow_html=True)

st.image('cuaca2.webp', width=1000)
st.write("This dashboard created by: [@kheylina](https://www.linkedin.com/in/kheylina-lidya-situmorang-474a0728a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")
st.write("Link Github: [@Github](https://github.com/kheylina)")
st.write("Gmail : kheylina172000@gmail")
st.write("Contact : (+62)821 8578 9552")

# Abstrak
st.write("<h2 style='color:#6A80B9;'>Abstrak</h2>", unsafe_allow_html=True)
st.write("""
Prediksi jenis cuaca merupakan salah satu tantangan penting dalam bidang meteorologi
untuk mendukung perencanaan dan mitigasi risiko. Penelitian ini membandingkan performa
berbagai algoritma machine learning, seperti K-Nearest Neighbors (KNN), Naïve Bayes (NB),
Support Vector Machine (SVM), Logistic Regression, Random Forest (RF), Multilayer Perceptron (MLP),
dan Decision Tree. Selain itu, dilakukan optimasi hyperparameter untuk meningkatkan akurasi.
""")

st.write("<h2 style='color:#6A80B9;'>Dataset</h2>", unsafe_allow_html=True)
st.dataframe(data.head())

# Input Data Pengguna
st.write("<h3 style='color:#F6C794;'>Silakan Input Data</h3>", unsafe_allow_html=True)

Temperature = st.number_input('Temperature (°C)', min_value=-25.0, max_value=100.0, value=25.0, step=0.1)
Humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
Wind_Speed = st.number_input('Wind Speed (km/h)', min_value=0.0, max_value=200.0, value=10.0, step=0.1)
Precipitation = st.number_input('Precipitation (mm)', min_value=0.0, max_value=500.0, value=0.0, step=0.1)
Cloud_Cover = st.selectbox("Cloud Cover", options=[0, 1, 2, 3], 
                           format_func=lambda x: ["Partly Cloudy", "Clear", "Overcast", "Cloudy"][x])
Atmospheric_Pressure = st.number_input('Atmospheric Pressure (hPa)', min_value=800.0, max_value=1100.0, value=1013.0, step=0.1)
UV_Index = st.number_input('UV Index', min_value=0.0, max_value=15.0, value=5.0, step=0.1)
Season = st.selectbox("Season", options=[0, 1, 2, 3],
                      format_func=lambda x: ["Summer", "Autumn", "Winter", "Spring"][x])
Visibility = st.number_input('Visibility (km)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
Location = st.selectbox("Location", options=[0, 1, 2],
                        format_func=lambda x: ["Inland", "Mountain", "Coastal"][x])

# Prediction Button
if st.button('Prediksi Cuaca'):
    try:
        # Preparing input data
        input_data = pd.DataFrame({
            'Temperature': [Temperature],
            'Humidity': [Humidity],
            'Wind Speed': [Wind_Speed],
            'Precipitation (%)': [Precipitation],
            'Cloud Cover': [Cloud_Cover],
            'Atmospheric Pressure': [Atmospheric_Pressure],
            'UV Index': [UV_Index],
            'Season': [Season],
            'Visibility (km)': [Visibility],
            'Location': [Location]
        })

        # Scaling the input data
        scaled_data = scaler.transform(input_data)

        # Predicting the weather type
        prediction_proba = model.predict_proba(scaled_data)[0]
        weather_types = ['Rainy', 'Cloudy', 'Sunny', 'Snowy']
        predicted_weather = weather_types[np.argmax(prediction_proba)]

        st.success(f"Jenis cuaca yang diprediksi adalah: {predicted_weather}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
