# Simulate time-degrading vibration signal leading to failure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def simulate_degrading_signal(sampling_rate=20000, duration=10, window_size=1):
    """
    Simulates vibration data over time with increasing defect characteristics
    leading to failure at the end.
    """
    total_windows = int(duration / window_size)
    samples_per_window = int(sampling_rate * window_size)
    full_signal = []
    timestamps = []

    for i in range(total_windows):
        t = np.linspace(0, window_size, samples_per_window)

        # Healthy base signal
        base = 0.1 * np.sin(2 * np.pi * 50 * t)

        # Gradually increasing defect (higher frequency)
        defect_strength = 0.01 + (i / total_windows) * 0.1
        defect = defect_strength * np.sin(2 * np.pi * 1200 * t)

        # Gradually increasing impulse noise
        impulses = np.zeros_like(t)
        num_impulses = 2 + int((i / total_windows) * 10)
        impulse_indices = np.random.choice(samples_per_window, size=num_impulses, replace=False)
        impulses[impulse_indices] = np.random.uniform(-1.0, 1.0, size=num_impulses)

        signal = base + defect + impulses
        full_signal.append(signal)
        timestamps.extend([i * window_size + x for x in t])

        full_signal = np.concatenate(full_signal)
        timestamps = np.array(timestamps)

        # Calculate RUL
        # RUL = Total Duration - Current Time
        rul = duration - timestamps

        # Show sample data as DataFrame
        degrading_df = pd.DataFrame({'time_sec': timestamps, 'amplitude': full_signal, 'RUL_sec': rul})

    return degrading_df


import numpy as np
import timesfm
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd # Added for Timedelta and date_range
from sklearn.metrics import mean_squared_error # Added for MSE calculation

from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint


def main():
    context_length = 256
    horizon_length = 64 # This will be the length of our "future" period for comparison
    ticker = 'MSFT'

    tfm = get_model(context_length, horizon_length)

    print("TimesFM model loaded successfully!")

    data = simulate_degrading_signal(2000)
    # Ensure we have enough data for context + horizon for comparison
    # We'll take the last (context_length + horizon_length) data points
    # to allow for a direct comparison of actual vs. predicted.
    total_data_needed = context_length + horizon_length
    if len(data) < total_data_needed:
        print(f"Not enough data downloaded. Need at least {total_data_needed} data points.")
        print(f"Only found {len(data.Close)}. Please adjust context_length or horizon_length, or get more data.")
        return

    # Take the required portion of data from the end
    data = data.set_index('time_sec')[['amplitude']]
    data_for_analysis = data[-total_data_needed:]


    historical_data_for_input = data_for_analysis.iloc[:context_length]['amplitude']
    # The actual future data that we want to compare our forecast against
    actual_future_data = data_for_analysis.iloc[context_length:context_length+horizon_length]['amplitude']

    # Prepare input for forecasting
    # Ensure it's a list of numpy arrays or lists
    forecast_input = [historical_data_for_input.values]
    frequency_input = [0] # Assuming unknown frequency as before

    print("\nPerforming forecast...")
    point_forecast, _ = tfm.forecast(
        forecast_input,
        freq=frequency_input,
    )

    predicted_future = point_forecast[0]

    print("Forecast complete!")
    print(f"Shape of predicted future: {predicted_future.shape}")
    print(f"Shape of actual future: {actual_future_data.shape}")

    # Calculate and print MSE
    # Ensure both arrays have the same length for MSE calculation
    min_len = min(len(actual_future_data), len(predicted_future))
    mse = calculate_mse(actual_future_data[:min_len], predicted_future[:min_len])
    print(f"\nMean Squared Error (MSE): {mse:.4f}")

    # Pass the actual_future_data to the plot function as well
    plot_forecast_float_index(historical_data_for_input, actual_future_data, predicted_future)


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


def calculate_mse(actual, predicted):
    """
    Calculates the Mean Squared Error (MSE) between actual and predicted values.
    """
    return mean_squared_error(actual, predicted)


import matplotlib.pyplot as plt
import numpy as np # Import numpy for arange, if not already imported

def plot_forecast_float_index(historical_data, actual_future_data, predicted_future):
    """
    Plots historical data, actual future data, and predicted future data
    when the DataFrame index is a float.

    Args:
        historical_data (pd.Series or np.ndarray): Historical data used for context.
                                                   Assumed to have a numerical (float) index.
        actual_future_data (pd.Series or np.ndarray): Actual data for the future period.
                                                      Assumed to have a numerical (float) index.
        predicted_future (np.ndarray): Predicted values for the future period.
    """
    plt.figure(figsize=(14, 7))

    # --- Plot historical data ---
    # If historical_data is a Series, use its index. If it's a plain array, create a numerical range.
    if hasattr(historical_data, 'index'):
        hist_index = historical_data.index
    else:
        hist_index = np.arange(len(historical_data))
    plt.plot(hist_index, historical_data, label='Historical Data', color='blue')

    future_start_val = hist_index[-1] + 1 # Start next numerical value after last historical
    plot_future_len = min(len(actual_future_data), len(predicted_future))

    # Generate a numerical range for the future period
    future_index_values = np.arange(future_start_val, future_start_val + plot_future_len)

    plt.plot(future_index_values, actual_future_data[:plot_future_len],
             label='Actual Future Data', color='green', linestyle='-')

    # --- Plot TimesFM forecast ---
    # Align predicted_future with the numerical future index
    plt.plot(future_index_values, predicted_future[:plot_future_len],
             label='TimesFM Forecast', color='red', linestyle='--')

    plt.title('TimesFM Forecasting: Actual vs. Predicted (Float Index)')
    plt.xlabel('Time Step (Float Index)')
    plt.ylabel('Value') # Generic ylabel, adjust based on your data's meaning
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()


if __name__ == '__main__':
    main()