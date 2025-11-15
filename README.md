# Air Quality Prediction ML Project

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Problem Statement](#problem-statement)
4. [Solution Overview](#solution-overview)
5. [Data Preparation](#data-preparation)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Model Selection and Tuning](#model-selection-and-tuning)
8. [Project Structure](#project-structure)
9. [Installation](#installation)
10. [Usage](#usage)
11. [API Documentation](#api-documentation)
12. [Deployment](#deployment)
13. [Future Work](#future-work)
14. [References](#references)

---

## Project Description

This project aims to **predict air pollution levels**, specifically **benzene concentration (C6H6(GT))**, using environmental and sensor data.  
A machine learning model is trained to provide fast predictions that can assist in monitoring and managing air quality.  
The project includes data preprocessing, exploratory analysis, model training, hyperparameter tuning, and deployment as a REST API.

---

## Dataset

- **Source:** [UCI Machine Learning Repository – Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
- **Description:** Contains hourly averaged responses from an array of chemical sensors embedded in an Air Quality Chemical Multisensor Device, along with meteorological data.
- **Features:**
  - Gas sensor measurements:  
    - `CO(GT)`: True hourly averaged concentration CO in mg/m^3
    - `PT08.S1(CO)`: Tin Oxide sensor response (CO)
    - `NMHC(GT)`: Non-methane hydrocarbons in microg/m^3
    - `C6H6(GT)`: Benzene concentration in microg/m^3 (target)
    - `PT08.S2(NMHC)`: Indium Oxide sensor response (NMHC)
    - `NOx(GT)`: Nitrogen oxides concentration in ppb
    - `PT08.S3(NOx)`: Tungsten Oxide sensor response (NOx)
    - `NO2(GT)`: Nitrogen dioxide concentration in microg/m^3
    - `PT08.S4(NO2)`: Tungsten Oxide sensor response (NO2)
    - `PT08.S5(O3)`: Indium Oxide sensor response (O3)
  - Meteorological data:
    - `T`: Temperature in °C
    - `RH`: Relative Humidity (%)
    - `AH`: Absolute Humidity
  - Time-related features (engineered):
    - `Year`, `Month`, `Day`, `Hour`
    - `Hour_sin`, `Hour_cos`, `Month_sin`, `Month_cos`, `Weekday_sin`, `Weekday_cos`
- **Target:** `C6H6(GT)` – benzene concentration

---

## Problem Statement

Air pollution is a critical public health issue.  
Predicting **benzene concentration** helps:
- Alert communities about high pollution periods
- Enable authorities to take preventive measures
- Improve environmental monitoring

The goal is to create a **predictive model** that can be accessed via a **REST API** for real-time inference.

---

## Solution Overview

- **Model:** RandomForestRegressor (with tuned hyperparameters)
- **Pipeline:**
  1. Data cleaning (handle missing values, duplicates, outliers)
  2. Feature engineering (sine/cosine transformations for cyclical features)
  3. Train/validation/test split
  4. Model training and hyperparameter tuning
  5. Model evaluation (RMSE, MAE, R²)
  6. Deployment as a **FastAPI** web service

- **Input:** Environmental sensor readings + time features
- **Output:** Predicted benzene concentration (`C6H6(GT)`)

---

## Data Preparation

- **Missing values:** Imputed using median or mean as appropriate
- **Duplicates:** Removed to ensure data quality
- **Outliers:** Treated using the IQR method for robust modeling
- **Feature engineering:** Added cyclical encodings for hour, month, weekday to capture periodicity

---

## Exploratory Data Analysis (EDA)

- **Correlation heatmaps:** To identify relationships between features and target
- **Time-series plots:** Visualization of `C6H6(GT)` trends over time
- **Distribution plots:** For sensor readings and meteorological variables
- **Feature importance:** Extracted from RandomForest to interpret model decisions

---

## Model Selection and Tuning

- **Models evaluated:**
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor (best performance)
  - GradientBoostingRegressor
  - XGBRegressor
- **Hyperparameter tuning:** Performed using RandomizedSearchCV
- **Evaluation metrics:** RMSE, MAE, R² on validation and test sets
- **Final model:** Trained on combined train + validation data, saved as a serialized artifact (`rf_model.pkl`)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/B-MEbrahim/Air-Quality-Prediction.git
   cd Air-Quality-Prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 3. Run the API

Start the FastAPI server:
```bash
uvicorn predict:app --reload
```

### 4. Make Predictions

Send a POST request to the API endpoint with sensor and time features:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"CO_GT": 2.5, "PT08_S1_CO": 1200, "NMHC_GT": 150, "PT08_S2_NMHC": 900, "NOx_GT": 50, "PT08_S3_NOx": 600, "NO2_GT": 30, "PT08_S4_NO2": 800, "PT08_S5_O3": 1000, "T": 18.5, "RH": 45, "AH": 0.8, "Hour": 14, "Month": 5, "Weekday": 2}'
```

---

## API Documentation

### Endpoint

- **POST** `/predict`

### Request Body

```json
{
  "CO_GT": 2.3,
  "PT08_S1_CO": 1200,
  "PT08_S2_NMHC": 950,
  "NOx_GT": 140,
  "PT08_S3_NOx": 800,
  "NO2_GT": 110,
  "PT08_S4_NO2": 850,
  "PT08_S5_O3": 900,
  "T": 18.5,
  "RH": 45,
  "AH": 0.7,
  "Year": 2004,
  "Day": 123,
  "Hour_sin": 0.5,
  "Hour_cos": 0.8,
  "Month_sin": 0.1,
  "Month_cos": 0.99,
  "Weekday_sin": -0.3,
  "Weekday_cos": 0.95
}
```

### Response

```json
{
  "predicted_C6H6_GT": float
}
```

### Example

```json
{
  "predicted_C6H6_GT": 12.34
}
```

---

## Deployment

- Deployed on Render: https://air-quality-prediction-28ic.onrender.com/

---

## Future Work

- Integrate real-time data ingestion from IoT sensors
- Add support for additional pollutants (e.g., NO2, CO)
- Implement model retraining pipeline for continuous improvement
- Enhance API with authentication and logging
- Build a dashboard for visualization and monitoring

---

## References

- [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)

