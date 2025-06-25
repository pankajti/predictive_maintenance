import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
# Skeleton for CMAPSS-based TimesFM RUL Prediction Pipeline

columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]

# Step 1: Load and select features from CMAPSS dataset
def load_cmapss_data(file_path):
    df = pd.read_csv(file_path, sep=' ', header=None)
    return df

# Step 2: Apply StandardScaler
def scale_features(df, feature_cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    return scaled, scaler

# Step 3: Apply PCA
def apply_pca(scaled_data, n_components=5):
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(scaled_data)
    return pcs, pca

# Step 4: Dummy TimesFM Forecast (replace with actual model)
def forecast_timesfm(pc_series, context_len=30, horizon=10):
    # Dummy placeholder: shift values forward as "forecast"
    forecasted = pc_series[-context_len:].mean(axis=0).reshape(1, -1).repeat(horizon, axis=0)
    return forecasted

# Step 5: Train regression model on PC + RUL
def train_regression_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Step 6: Predict RUL using TimesFM forecasts
def predict_rul(model, forecasted_pcs):
    return model.predict(forecasted_pcs)

# Example driver function
def run_pipeline():
    DATA_PATH = r'/Users/pankajti/dev/data/kaggle/nasa/CMaps'

    file_path = os.path.join(DATA_PATH, 'train_FD001.txt')
    df = load_cmapss_data(file_path)
    df.drop(columns=[26, 27], inplace=True)
    df.columns = columns
    df.drop(columns=['Nf_dmd', 'PCNfR_dmd', 'P2', 'T2', 'TRA', 'farB', 'epr'], inplace=True)
    # Define columns
    feature_cols = list(df.columns[5:])  # sensor readings
    unit_col = 0
    cycle_col = 1

    # Calculate RUL
    df['max_cycle'] = df.groupby(unit_col)[cycle_col].transform('max')
    df['RUL'] = df['max_cycle'] - df[cycle_col]

    # Step 2 & 3
    scaled_data, scaler = scale_features(df, feature_cols)
    pcs, pca = apply_pca(scaled_data, n_components=5)

    # Step 4: Forecast PCs
    forecasted_pcs = forecast_timesfm(pcs)

    # Step 5: Train regression on PCs to RUL
    X_train = pcs
    y_train = df['RUL'].values
    reg_model = train_regression_model(X_train, y_train)

    # Step 6: Predict RUL
    predicted_rul = predict_rul(reg_model, forecasted_pcs)

    # Plot
    plt.plot(predicted_rul, label='Predicted RUL')
    plt.title("Forecasted RUL using TimesFM + PCA + Regression")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("RUL")
    plt.legend()
    plt.show()

    return predicted_rul

predicted_rul_result = run_pipeline()
