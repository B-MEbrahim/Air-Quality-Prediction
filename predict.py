from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('rf_model.pkl')


FEATURES = [
    'CO(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'NOx(GT)',
    'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
    'T', 'RH', 'AH', 'Year', 'Day',
    'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos',
    'Weekday_sin', 'Weekday_cos'
]


app = FastAPI(title='Air Quality Prediciton API')

class AirInput(BaseModel):
    CO_GT: float
    PT08_S1_CO: float
    PT08_S2_NMHC: float
    NOx_GT: float
    PT08_S3_NOx: float
    NO2_GT: float
    PT08_S4_NO2: float
    PT08_S5_O3: float
    T: float
    RH: float
    AH: float
    Year: int
    Day: int
    Hour_sin: float
    Hour_cos: float
    Month_sin: float
    Month_cos: float
    Weekday_sin: float
    Weekday_cos: float


def prepare_input(data: AirInput):
    row = {
        'CO(GT)': data.CO_GT,
        'PT08.S1(CO)': data.PT08_S1_CO,
        'PT08.S2(NMHC)': data.PT08_S2_NMHC,
        'NOx(GT)': data.NOx_GT,
        'PT08.S3(NOx)': data.PT08_S3_NOx,
        'NO2(GT)': data.NO2_GT,
        'PT08.S4(NO2)': data.PT08_S4_NO2,
        'PT08.S5(O3)': data.PT08_S5_O3,
        'T': data.T,
        'RH': data.RH,
        'AH': data.AH,
        'Year': data.Year,
        'Day': data.Day,
        'Hour_sin': data.Hour_sin,
        'Hour_cos': data.Hour_cos,
        'Month_sin': data.Month_sin,
        'Month_cos': data.Month_cos,
        'Weekday_sin': data.Weekday_sin,
        'Weekday_cos': data.Weekday_cos,
    }

    df = pd.DataFrame([row])
    df = df[FEATURES]
    return df 

@app.get("/")
def home():
    return {"message": "Welcome to the Air Quality Prediction API"}

@app.post("/predict")
def predict_air_quality(data:AirInput):
    df = prepare_input(data)
    pred = model.predict(df)[0]
    return {"predicted_C6H6(GT)": float(pred)}

