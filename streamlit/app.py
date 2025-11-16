import streamlit as st
import requests
import plotly.graph_objects as go
import datetime
import math

API_URL = "https://air-quality-prediction-28ic.onrender.com/predict"

SAFE_LIMIT = 25
MODERATE_LIMIT = 10

st.set_page_config(page_title="Air Quality Predictor", layout="centered")
st.title("Air Quality Predictor (Benzene C6H6)")

st.write("""
Enter the environmental sensor data and choose date/time.  
""")

st.subheader("Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    CO_GT = st.slider("CO(GT)", 0.0, 20.0, 2.0)
    PT08_S1_CO = st.slider("PT08.S1(CO)", 500, 2000, 1000)
    PT08_S2_NMHC = st.slider("PT08.S2(NMHC)", 300, 2000, 900)
    NOx_GT = st.slider("NOx(GT)", 0, 500, 150)
    PT08_S3_NOx = st.slider("PT08.S3(NOx)", 300, 2000, 800)

with col2:
    NO2_GT = st.slider("NO2(GT)", 0, 500, 120)
    PT08_S4_NO2 = st.slider("PT08.S4(NO2)", 300, 2000, 850)
    PT08_S5_O3 = st.slider("PT08.S5(O3)", 300, 2000, 900)
    T = st.slider("Temperature (°C)", -10.0, 50.0, 20.0)
    RH = st.slider("Relative Humidity (%)", 0.0, 100.0, 50.0)
    AH = st.slider("Absolute Humidity", 0.0, 2.0, 0.7)


st.subheader("Date & Time")

date_input = st.date_input("Date", datetime.date(2004, 5, 1))
hour_input = st.slider("Hour of Day", 0, 23, 12)

# Convert to needed features
day_of_year = date_input.timetuple().tm_yday
weekday = date_input.weekday()
month = date_input.month

# Cyclical encoding
Hour_sin = math.sin(2 * math.pi * hour_input / 24)
Hour_cos = math.cos(2 * math.pi * hour_input / 24)

Month_sin = math.sin(2 * math.pi * month / 12)
Month_cos = math.cos(2 * math.pi * month / 12)

Weekday_sin = math.sin(2 * math.pi * weekday / 7)
Weekday_cos = math.cos(2 * math.pi * weekday / 7)


payload = {
    "CO_GT": CO_GT,
    "PT08_S1_CO": PT08_S1_CO,
    "PT08_S2_NMHC": PT08_S2_NMHC,
    "NOx_GT": NOx_GT,
    "PT08_S3_NOx": PT08_S3_NOx,
    "NO2_GT": NO2_GT,
    "PT08_S4_NO2": PT08_S4_NO2,
    "PT08_S5_O3": PT08_S5_O3,
    "T": T,
    "RH": RH,
    "AH": AH,
    "Year": date_input.year,
    "Day": day_of_year,
    "Hour_sin": Hour_sin,
    "Hour_cos": Hour_cos,
    "Month_sin": Month_sin,
    "Month_cos": Month_cos,
    "Weekday_sin": Weekday_sin,
    "Weekday_cos": Weekday_cos
}


if st.button("🔍 Predict Air Quality"):

    with st.spinner("Predicting..."):
        response = requests.post(API_URL, json=payload)
        prediction = response.json()["predicted_C6H6(GT)"]

    st.subheader("Benzene Level Gauge")


    max_value = 65  
    value = prediction  

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': "Predicted Benzene Concentration (µg/m³)"},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 15], 'color': "green"},   
                {'range': [15, 30], 'color': "yellow"}, 
                {'range': [30, 45], 'color': "orange"}, 
                {'range': [45, max_value], 'color': "red"} 
            ],
        }
    ))

    st.plotly_chart(fig)


    

    # Status Message
    if prediction <= SAFE_LIMIT:
        st.success("Air Quality: **SAFE**")
    elif prediction <= MODERATE_LIMIT:
        st.warning(" Air Quality: **MODERATE** — sensitive individuals beware.")
    else:
        st.error("Air Quality: **HAZARDOUS** — avoid outdoor activity!")

    st.write(f"**Predicted C6H6(GT): {prediction:.2f} µg/m³**")

