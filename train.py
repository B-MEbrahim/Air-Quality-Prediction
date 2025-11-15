import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib 


def load_data(path='data\AirQualityUCI.csv'):
    df = pd.read_csv(path, sep=';', decimal=',')
    df = df.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1)
    return df


def preprocess_data(df):
    df = df.drop_duplicates(keep='first')

    df = df.replace(-200, np.nan)

    df.drop(columns=['NMHC(GT)'],inplace=True)

    # cols with missing values
    cols = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)','PT08.S2(NMHC)', 'NOx(GT)', 
        'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)','PT08.S5(O3)', 'T', 
        'RH', 'AH']
    for col in cols:
        df[col] = df[col].fillna(df[col].mean())

    df = df.dropna()


    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    df = df.drop(['Date'], axis=1)

    df['Hour'] = df['Time'].str.split('.').str[0].astype(int)
    df = df.drop(['Time'], axis=1)


    df = encode_cyclical(df, 'Hour', 23)
    df = encode_cyclical(df, 'Month', 12)
    df = encode_cyclical(df, 'Weekday', 6)

    df = df.drop(['Hour', 'Month', 'Weekday'], axis=1)

    return df


def encode_cyclical(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def split_data(df, target="C6H6(GT)"):
    X = df.drop(columns=[target])
    y = df[target]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42)  

    # outlier caping
    q1 = X_train.quantile(0.25)
    q3 = X_train.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr


    for col in X_train.columns:
        if col in lower_bound.index: # Ensure column is numeric and in iqr calc
            train_med = X_train[col].median()
            
            # Cap X_train
            X_train.loc[X_train[col] < lower_bound[col], col] = train_med
            X_train.loc[X_train[col] > upper_bound[col], col] = train_med
            
            # Cap X_val
            X_val.loc[X_val[col] < lower_bound[col], col] = train_med
            X_val.loc[X_val[col] > upper_bound[col], col] = train_med
            
            # Cap X_test
            X_test.loc[X_test[col] < lower_bound[col], col] = train_med
            X_test.loc[X_test[col] > upper_bound[col], col] = train_med

    return X_train, X_val, X_test, y_train, y_val, y_test



def train_final_model(X_train, X_val, y_train, y_val):
    X_final = np.vstack([X_train, X_val])
    y_final = pd.concat([y_train, y_val])

    model = RandomForestRegressor(
        n_estimators= 500,
        min_samples_split=10,
        min_samples_leaf=2,
        max_depth=30,
        random_state=42,
        n_jobs=-1,
        )
    
    model.fit(X_final, y_final)

    return model


def save_artifacts(model):
    joblib.dump(model, "rf_model.pkl")
    


def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    df = preprocess_data(df)

    print("Splitting...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("Training the final model....")
    model = train_final_model(X_train, X_val, y_train, y_val)

    print("Saving the model...")
    save_artifacts(model)

    print("Training Complete.")


if __name__ == '__main__':
    main()