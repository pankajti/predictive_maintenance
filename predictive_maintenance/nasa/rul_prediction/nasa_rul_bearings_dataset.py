import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import timesfm
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error # Added for MSE calculation

features_names = ['mean', 'std', 'kurtosis']
baseline_params ={}

# Define constants
CHANNEL_MAP = {
    1: (0, 1),
    2: (2, 3),
    3: (4, 5),
    4: (6, 7)
}
DATA_PATH = r'/Users/pankajti/dev/data/kaggle/nasa/archive (1)/1st_test/1st_test'
SAMPLING_RATE = 20000  # 20 kHz
WINDOW_SIZE = 400
SECONDS_PER_FILE = 1  # Each file is a 1-second snapshot

from tqdm import tqdm
def extract_timestamp(filename):
    base_name = os.path.basename(filename)
    timestamp_str = base_name.replace('.', '-')  # e.g., '2003.10.22.12.06.24' to '2003-10-22-12-06-24'
    return datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')

# Function to extract rolling features with timestamp per sliding window
def extract_features_with_timestamps(bearing_number=3):
    files = sorted(os.listdir(DATA_PATH))
    all_features = []
    all_timestamps = []

    for file in tqdm(files):
        file_path = os.path.join(DATA_PATH, file)
        #print(f"reading file {file_path}")

        if not os.path.exists(file_path):
            continue
        try:
            data = pd.read_table(file_path, header=None)
            ch1, ch2 = CHANNEL_MAP[bearing_number]
            signal = data.iloc[:, ch2].values
            base_timestamp = extract_timestamp(file)

            # Generate sliding windows
            for i in range(0, len(signal) - WINDOW_SIZE + 1, WINDOW_SIZE):
                window = signal[i:i + WINDOW_SIZE]
                if len(window) == WINDOW_SIZE:
                    feat = [
                        np.mean(window),
                        np.std(window),
                        kurtosis(window)
                    ]
                    # Each window spans window_size/SAMPLING_RATE seconds
                    delta_seconds = i / SAMPLING_RATE
                    timestamp = base_timestamp + timedelta(seconds=delta_seconds)
                    all_features.append(feat)
                    all_timestamps.append(timestamp)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            raise e

    df = pd.DataFrame(all_features, columns=['mean', 'std', 'kurtosis'])
    df['timestamp'] = all_timestamps
    return df

def generate_rul_labels(num_samples):
    return np.arange(num_samples - 1, -1, -1).reshape(-1, 1)

def get_model(context_length, horizon_length):
    hparams = TimesFmHparams(
        backend="torch",
        context_len=context_length,
        horizon_len=horizon_length,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
    )
    checkpoint = TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    )
    print(f"Initializing TimesFM with context_len={hparams.context_len}, horizon_len={hparams.horizon_len}")
    tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
    return tfm


def main():
    processed_data = get_processed_data()
    context_length = 256
    horizon_length = 32
    tfm = get_model(context_length, horizon_length)

    print("TimesFM model loaded successfully!")
    actual_future_data, forecast_input = get_data_for_model(context_length, horizon_length, processed_data)

    frequency_input = [0, 0, 0]  # Assuming unknown frequency as before

    print("\nPerforming forecast...")
    point_forecast, point_forecast_2 = tfm.forecast(
        forecast_input,
        freq=frequency_input,
    )
    predicted_future_data = pd.DataFrame(index=actual_future_data.index)
    for idx, feature in enumerate(features_names):
        predicted_future_data[feature] = point_forecast[idx]
        print(
            f"mse for feature {feature} is {mean_squared_error(predicted_future_data[feature], actual_future_data[feature])}")

def get_data_for_model(context_length, horizon_length, processed_data):
    data = processed_data[features_names]
    total_data_needed = context_length + horizon_length
    if len(data) < total_data_needed:
        print(f"Not enough data downloaded. Need at least {total_data_needed} data points.")
        print(f"Only found {len(data)}. Please adjust context_length or horizon_length, or get more data.")
    data_for_analysis = data[-total_data_needed:]
    historical_data_for_input = data_for_analysis[:context_length]
    # The actual future data that we want to compare our forecast against
    actual_future_data = data_for_analysis[context_length:]
    forecast_input = [ historical_data_for_input[feature].values for feature in features_names]
    return actual_future_data, forecast_input


def get_processed_data():
    # Main processing
    features = extract_features_with_timestamps(bearing_number=3)
    features = features.set_index('timestamp')
    rul_labels = generate_rul_labels(len(features))
    # Normalize features and labels
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    features_scaled = scaler_X.fit_transform(features)
    rul_scaled = scaler_y.fit_transform(rul_labels)
    processed_data = pd.DataFrame(features_scaled, columns=['mean', 'std', 'kurtosis'], index=features.index)
    processed_data['RUL'] = rul_scaled

    baseline_window_size = int(0.05 * len(processed_data))
    baseline_data = processed_data.iloc[:baseline_window_size]

    # Calculate baseline mean and std
    baseline_mean = baseline_data.mean()

    # Store as dictionary for inspection or later use
    baseline_params["mean"]= baseline_mean.to_dict()
 

    return processed_data


if __name__ == '__main__':
    main()

