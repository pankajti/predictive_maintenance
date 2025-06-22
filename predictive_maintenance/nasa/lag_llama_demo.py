import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from gluonts.dataset.common import ListDataset
from lag_llama.gluon.estimator import LagLlamaEstimator
from predictive_maintenance.nasa.data_reader import load_first_test_data

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.predictor import PyTorchPredictor
from huggingface_hub import hf_hub_download
from pathlib import Path
import pickle


def main():
    # Load NASA bearing dataset
    data = load_first_test_data(bearing_number=3).reset_index()
    context_length = 256
    horizon_length = 32

    # Get the Lag-Llama model
    predictor = get_model(context_length, horizon_length)

    print("Lag-Llama model loaded successfully!")

    # Ensure enough data for context + horizon
    total_data_needed = context_length + horizon_length
    if len(data) < total_data_needed:
        print(f"Not enough data downloaded. Need at least {total_data_needed} data points.")
        print(f"Only found {len(data)}. Please adjust context_length or horizon_length, or get more data.")
        return

    # Take the required portion of data from the end
    data_for_analysis = data[-total_data_needed:]
    data_for_analysis.index = pd.date_range(start="2023-01-01", periods=len(data_for_analysis), freq="1S")

    historical_data_for_input = data_for_analysis[:context_length]
    actual_future_data = data_for_analysis[context_length:]

    # Prepare input for forecasting in gluonts format
    forecast_input = ListDataset(
        [{"start": historical_data_for_input.index[0], "target": historical_data_for_input.iloc[:,1].values}],use_timestamp=False,
        freq="1ms"  # Assuming 1-second frequency for vibration data
    )

    print("\nPerforming forecast...")
    # Forecast using Lag-Llama
    forecast_it = predictor.predict(forecast_input, num_samples=100)
    forecast = list(forecast_it)[0]  # Get the first (and only) forecast
    predicted_future = forecast.mean  # Use mean of samples for point forecast

    print("Forecast complete!")
    print(f"Shape of predicted future: {predicted_future.shape}")
    print(f"Shape of actual future: {actual_future_data.shape}")

    # Calculate and print MSE
    min_len = min(len(actual_future_data), len(predicted_future))
    mse = calculate_mse(actual_future_data[:min_len], predicted_future[:min_len])
    print(f"\nMean Squared Error (MSE): {mse:.4f}")

    # Plot forecast with uncertainty bands
    plot_forecast(historical_data_for_input, actual_future_data, predicted_future, forecast)

def get_model(context_length, horizon_length):
    # Download the pre-trained Lag-Llama checkpoint
    ckpt_path = hf_hub_download(
        repo_id="time-series-foundation-models/Lag-Llama",
        filename="lag-llama.ckpt",
        local_dir="checkpoints"
    )

    # Load the model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)
    model = checkpoint["model"]
    # Move model to device
    model.to(device)
    model.eval()

    # Create predictor
    predictor = PyTorchPredictor(
        prediction_length=horizon_length,
        context_length=context_length,
        model=model,
        device=device
    )
    print(f"Initializing Lag-Llama with context_length={context_length}, prediction_length={horizon_length}")
    return predictor

def calculate_mse(actual, predicted):
    """
    Calculates the Mean Squared Error (MSE) between actual and predicted values.
    """
    return mean_squared_error(actual, predicted)

def plot_forecast(historical_data, actual_future_data, predicted_future, forecast):
    plt.figure(figsize=(14, 7))

    # Plot historical data
    plt.plot(historical_data.index, historical_data, label='Historical Data', color='blue')

    # Create index for future period
    future_start_index = historical_data.index[-1] + 1
    plot_future_len = min(len(actual_future_data), len(predicted_future))
    future_index = list(range(future_start_index, future_start_index + plot_future_len))

    # Plot actual future data
    plt.plot(actual_future_data.index[:plot_future_len], actual_future_data[:plot_future_len],
             label='Actual Future Data', color='green', linestyle='dotted')

    # Plot Lag-Llama forecast
    plt.plot(future_index, predicted_future[:plot_future_len], label='Lag-Llama Forecast', color='red', linestyle='--')

    # Plot uncertainty bands (80% confidence interval)
    plt.fill_between(future_index,
                     forecast.quantile(0.1)[:plot_future_len],
                     forecast.quantile(0.9)[:plot_future_len],
                     color='red', alpha=0.2, label='80% Confidence Interval')

    plt.title('Lag-Llama Forecasting: Actual vs. Predicted with Uncertainty')
    plt.xlabel('Time')
    plt.ylabel('Vibration Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()