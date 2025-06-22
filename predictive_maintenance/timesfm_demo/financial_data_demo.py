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
    ticker = 'AMZN'

    tfm = get_model(context_length, horizon_length)

    print("TimesFM model loaded successfully!")

    data = yf.download(ticker)
    # Ensure we have enough data for context + horizon for comparison
    # We'll take the last (context_length + horizon_length) data points
    # to allow for a direct comparison of actual vs. predicted.
    total_data_needed = context_length + horizon_length
    if len(data.Close) < total_data_needed:
        print(f"Not enough data downloaded. Need at least {total_data_needed} data points.")
        print(f"Only found {len(data.Close)}. Please adjust context_length or horizon_length, or get more data.")
        return

    # Take the required portion of data from the end
    data_for_analysis = data.Close[-total_data_needed:]


    historical_data_for_input = data_for_analysis[:context_length][ticker]
    # The actual future data that we want to compare our forecast against
    actual_future_data = data_for_analysis[context_length:]

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
    plot_forecast(historical_data_for_input, actual_future_data, predicted_future)


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


def plot_forecast(historical_data, actual_future_data, predicted_future):
    plt.figure(figsize=(14, 7))

    # Plot historical data
    plt.plot(historical_data.index, historical_data, label='Historical Data', color='blue')

    # Create an index for the future period for both actual and predicted values
    # Start the future index from the day after the last historical data point
    future_start_index = historical_data.index[-1] + pd.Timedelta(days=1)
    # Generate a date range for the future period, assuming daily data
    # Ensure the length matches the minimum of actual_future_data and predicted_future
    plot_future_len = min(len(actual_future_data), len(predicted_future))
    future_index = pd.date_range(start=future_start_index, periods=plot_future_len, freq='D')

    # Plot actual future data (only up to the length of predicted_future if shorter)
    plt.plot(actual_future_data.index[:plot_future_len], actual_future_data[:plot_future_len],
             label='Actual Future Data', color='green', linestyle='dotted')

    # Plot TimesFM forecast (only up to the length of actual_future_data if shorter)
    plt.plot(future_index, predicted_future[:plot_future_len], label='TimesFM Forecast', color='red', linestyle='--')

    plt.title('TimesFM Forecasting: Actual vs. Predicted')
    plt.xlabel('Date')
    plt.ylabel('Value (MSFT Close Price)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()


if __name__ == '__main__':
    main()