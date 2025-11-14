import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    return {'mse': mse, 'preds': preds}

# ARIMA using pmdarima (auto_arima)
def train_arima(series, seasonal=False, m=1):
    import pmdarima as pm
    model = pm.auto_arima(series, seasonal=seasonal, m=m, suppress_warnings=True)
    return model

# Prophet (now part of prophet package)
def train_prophet(df):
    from prophet import Prophet
    m = Prophet()
    m.fit(df[['ds','y']])
    return m

# LSTM model (Keras) - simple example
def build_lstm(input_shape, units=50):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
