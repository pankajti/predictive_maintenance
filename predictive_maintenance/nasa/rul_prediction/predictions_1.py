#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import timesfm
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
from sklearn.metrics import mean_squared_error
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
FEATURES_NAMES = ['mean', 'std', 'kurtosis']
CHANNEL_MAP = {1: (0, 1), 2: (2, 3), 3: (4, 5), 4: (6, 7)}
DATA_PATH = r'/Users/pankajti/dev/data/kaggle/nasa/archive (1)/1st_test/1st_test'
SAMPLING_RATE = 20000
WINDOW_SIZE = 400
STEP_SIZE = WINDOW_SIZE // 2  # 50% overlap
SECONDS_PER_FILE = 1
CONTEXT_LENGTH = 256
HORIZON_LENGTH = 32
HEALTH_THRESHOLD = 0.9

def extract_timestamp(filename):
    try:
        base_name = os.path.basename(filename)
        timestamp_str = base_name.replace('.', '-')
        return datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')
    except ValueError as e:
        logging.warning(f"Invalid filename: {filename}, error: {e}")
        return None

def extract_features_with_timestamps(bearing_number=3):
    files = [f for f in sorted(os.listdir(DATA_PATH)) ]
    all_features = []
    all_timestamps = []

    for file in tqdm(files, desc="Processing files"):
        file_path = os.path.join(DATA_PATH, file)
        timestamp = extract_timestamp(file)
        if timestamp is None:
            continue
        try:
            data = pd.read_table(file_path, header=None)
            ch1, ch2 = CHANNEL_MAP[bearing_number]
            signal1 = data.iloc[:, ch1].values
            signal2 = data.iloc[:, ch2].values

            for i in range(0, len(signal1) - WINDOW_SIZE + 1, STEP_SIZE):
                window1 = signal1[i:i + WINDOW_SIZE]
                window2 = signal2[i:i + WINDOW_SIZE]
                if len(window1) == WINDOW_SIZE and len(window2) == WINDOW_SIZE:
                    feat = [
                        np.mean(window1), np.std(window1), kurtosis(window1),
                        np.mean(window2), np.std(window2), kurtosis(window2)
                    ]
                    delta_seconds = i / SAMPLING_RATE
                    window_timestamp = timestamp + timedelta(seconds=delta_seconds)
                    all_features.append(feat)
                    all_timestamps.append(window_timestamp)
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")
            continue

    columns = ['mean1', 'std1', 'kurtosis1', 'mean2', 'std2', 'kurtosis2']
    df = pd.DataFrame(all_features, columns=columns)
    df['timestamp'] = all_timestamps
    return df

def compute_health_index(features):
    scaler = MinMaxScaler()
    kurtosis_scaled = scaler.fit_transform(features[['kurtosis1', 'kurtosis2']].mean(axis=1).values.reshape(-1, 1))
    std_scaled = scaler.fit_transform(features[['std1', 'std2']].mean(axis=1).values.reshape(-1, 1))
    return 0.7 * kurtosis_scaled + 0.3 * std_scaled

def calculate_rul(health_index, total_samples):
    failure_idx = np.where(health_index >= HEALTH_THRESHOLD)[0]
    failure_point = failure_idx[0] if len(failure_idx) > 0 else total_samples - 1
    rul = np.zeros(total_samples)
    for i in range(total_samples):
        if i <= failure_point:
            rul[i] = (failure_point - i) * np.exp(-health_index[i] * 2)
        else:
            rul[i] = 0
    return rul.reshape(-1, 1)

def get_model(context_length, horizon_length):
    try:
        hparams = TimesFmHparams(
            backend="torch",
            context_len=context_length,
            horizon_len=horizon_length,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
        )
        checkpoint = TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
        tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
        logging.info(f"TimesFM initialized with context_len={context_length}, horizon_len={horizon_length}")
        return tfm
    except Exception as e:
        logging.error(f"Failed to initialize TimesFM: {e}")
        raise

def get_processed_data():
    features = extract_features_with_timestamps(bearing_number=3)
    if features.empty:
        logging.error("No features extracted.")
        return None
    features = features.set_index('timestamp')
    health_index = compute_health_index(features)
    rul_labels = calculate_rul(health_index, len(features))

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    features_scaled = scaler_X.fit_transform(features)
    health_index_scaled = scaler_y.fit_transform(health_index)
    rul_scaled = scaler_y.fit_transform(rul_labels)

    columns = ['mean1', 'std1', 'kurtosis1', 'mean2', 'std2', 'kurtosis2']
    processed_data = pd.DataFrame(features_scaled, columns=columns, index=features.index)
    processed_data['health_index'] = health_index_scaled
    processed_data['RUL'] = rul_scaled

    baseline_window_size = int(0.05 * len(processed_data))
    baseline_data = processed_data.iloc[:baseline_window_size]
    baseline_params = {"mean": baseline_data.mean().to_dict()}

    return processed_data, baseline_params

def get_data_for_model(context_length, horizon_length, processed_data):
    data = processed_data[['health_index','RUL']]
    total_data_needed = context_length + horizon_length
    if len(data) < total_data_needed:
        logging.error(f"Not enough data. Need {total_data_needed}, found {len(data)}.")
        return None, None
    data_for_analysis = data[-total_data_needed:]
    historical_data = data_for_analysis[:context_length]
    actual_future_data = data_for_analysis[context_length:]
    forecast_input = [historical_data['health_index'].values]
    return actual_future_data, forecast_input

def main():
    processed_data, baseline_params = get_processed_data()
    if processed_data is None:
        return

    tfm = get_model(CONTEXT_LENGTH, HORIZON_LENGTH)
    actual_future_data, forecast_input = get_data_for_model(CONTEXT_LENGTH, HORIZON_LENGTH, processed_data)
    if actual_future_data is None:
        return

    frequency_input = [1]  # Assume periodicity
    try:
        point_forecast, _ = tfm.forecast(forecast_input, freq=frequency_input)
        predicted_health = point_forecast[0]
    except Exception as e:
        logging.error(f"Forecasting failed: {e}")
        return

    predicted_rul = np.zeros(HORIZON_LENGTH)
    for i in range(HORIZON_LENGTH):
        if predicted_health[i] >= HEALTH_THRESHOLD:
            predicted_rul[i:] = 0
            break
        predicted_rul[i] = (HORIZON_LENGTH - i) * np.exp(-predicted_health[i] * 2)

    futures_data = actual_future_data.copy()
    futures_data['predicted_health_index'] = predicted_health
    futures_data['predicted_rul'] = predicted_rul

    plt.figure(figsize=(10, 6))
    futures_data.plot(title='Actual vs Predicted Health Index and RUL')
    plt.xlabel('Timestamp')
    plt.ylabel('Normalized Value')
    plt.savefig('forecast.png')
    plt.show()

    mse_health = mean_squared_error(futures_data['health_index'], futures_data['predicted_health_index'])
    mse_rul = mean_squared_error(futures_data['RUL'], futures_data['predicted_rul'])
    logging.info(f"MSE for Health Index: {mse_health:.4f}")
    logging.info(f"MSE for RUL: {mse_rul:.4f}")

if __name__ == '__main__':
    main()