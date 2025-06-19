import numpy as np # Added for data generation
import timesfm
import matplotlib.pyplot as plt # Added for plotting

# --- 1. Import necessary classes explicitly ---
# It's good practice to list all classes you use from the module.
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint


def main():
    context_length = 256
    horizon_length = 64

    tfm = get_model(context_length, horizon_length)

    print("TimesFM model loaded successfully!")

    total_points = context_length + horizon_length * 2
    time = np.arange(total_points)
    data = 5 * np.sin(time / 10) + 2 * np.cos(time / 5) + np.random.normal(0, 0.5, total_points)

    historical_data = data[:context_length]
    forecast_input = [historical_data]
    frequency_input = [0]

    print("\nPerforming forecast...")
    point_forecast, _ = tfm.forecast(
        forecast_input,
        freq=frequency_input,
    )

    tfm.forecast_with_covariates()

    tfm.forecast_on_df()

    predicted_future = point_forecast[0]

    print("Forecast complete!")
    print(f"Shape of predicted future: {predicted_future.shape}")

    #plot_forecast(historical_data, predicted_future)


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


def plot_forecast(historical_data, predicted_future):
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(historical_data)), historical_data, label='Historical Data', color='blue')
    plt.plot(np.arange(len(historical_data), len(historical_data) + len(predicted_future)), predicted_future,
             label='TimesFM Forecast', color='red', linestyle='--')
    plt.title('TimesFM Forecasting Example')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    main()