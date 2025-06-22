import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
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

# Extract timestamp from filename
def extract_timestamp(filename):
    base_name = os.path.basename(filename)
    timestamp_str = base_name.replace('.', '-')  # e.g., '2003.10.22.12.06.24' to '2003-10-22-12-06-24'
    return datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')

# Function to extract rolling features with timestamp per sliding window
def extract_features_with_timestamps(bearing_number=3):
    files = sorted(os.listdir(DATA_PATH))
    all_features = []
    all_timestamps = []

    for file in files:
        file_path = os.path.join(DATA_PATH, file)
        print(f"reading file {file_path}")

        if not os.path.exists(file_path):
            continue

        try:
            data = pd.read_table(file_path, header=None)
            ch1, _ = CHANNEL_MAP[bearing_number]
            signal = data.iloc[:, ch1].values
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


# Generate RUL labels
def generate_rul_labels(num_samples):
    return np.arange(num_samples - 1, -1, -1).reshape(-1, 1)

# Main processing
features = extract_features_with_timestamps(bearing_number=3)
features = features.set_index('timestamp')
rul_labels = generate_rul_labels(len(features))

# Normalize features and labels
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

features_scaled = scaler_X.fit_transform(features)
rul_scaled = scaler_y.fit_transform(rul_labels)

# Combine for display
processed_data = pd.DataFrame(features_scaled, columns=['mean', 'std', 'kurtosis'],index=features.index)
processed_data['RUL'] = rul_scaled
processed_data.mean().plot()
plt.show()
print(processed_data.shape)

print(processed_data.head())

