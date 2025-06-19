import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timesfm
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint

# Load simulated vibration dataset
df = pd.read_csv("simulated_rul_data.csv")  # Replace with full path if needed
vibration_data = df.drop(columns=["RUL"]).values

# Select one example
sample_index = 100
sample_series = vibration_data[sample_index]

# TimesFM setup
context_length = 256
horizon_length = 64

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

tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

# Forecast
forecast_input = [sample_series[:context_length]]
frequency_input = [0]  # Sensor frequency

point_forecast, _ = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)

predicted_future = point_forecast[0]

# Plot
plt.figure(figsize=(12, 6))
context_length=len(forecast_input[0])
plt.plot(np.arange(context_length), forecast_input[0], label="Historical Vibration")
plt.plot(np.arange(context_length, context_length + len(predicted_future)),
         predicted_future, label="Predicted Future", linestyle="--", color="red")
plt.title("TimesFM Forecast on Simulated Vibration Data")
plt.xlabel("Time Step")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
