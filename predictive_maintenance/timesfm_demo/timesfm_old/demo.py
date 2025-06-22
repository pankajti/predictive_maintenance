import numpy as np # Added for data generation
import timesfm
import matplotlib.pyplot as plt # Added for plotting

# --- 1. Import necessary classes explicitly ---
# It's good practice to list all classes you use from the module.
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint

# --- 2. Define Hyperparameters and Checkpoint Correctly ---
# You need to specify a valid backend and context/horizon lengths.
# Also, the checkpoint requires the Hugging Face repository ID.

# These parameters should match the model you intend to load.
# For 'google/timesfm_old-1.0-200m-pytorch':
#  - context_len: typically max 512, multiple of 32 (input_patch_len)
#  - horizon_len: can be anything, but often <= context_len for general use cases.
#  - input_patch_len: 32 (for v1.0)
#  - output_patch_len: 128 (for v1.0)
#  - num_layers, model_dims: fixed by the loaded checkpoint
context_length = 256 # A common, valid context length for TimesFM 1.0
horizon_length = 64  # How many future steps to predict

hparams = TimesFmHparams(
    backend="torch",  # Specify backend: "cpu", "gpu", or "torch" if using PyTorch only
                      # "torch" is often the safest when using PyTorch checkpoints
    context_len=context_length,
    horizon_len=horizon_length,
    input_patch_len=32, # Default for 1.0 models
    output_patch_len=128, # Default for 1.0 models
    num_layers=20, # Default for 200M model
    model_dims=1280, # Default for 200M model
)

checkpoint = TimesFmCheckpoint(
    huggingface_repo_id="google/timesfm-1.0-200m-pytorch" # MANDATORY: specify the model to load
)

print(f"Initializing TimesFM with context_len={hparams.context_len}, horizon_len={hparams.horizon_len}")

# --- 3. Initialize the TimesFM Model ---
# Ensure your environment has the 'timesfm_old[torch]' or 'timesfm_old[jax]' dependency installed.
tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

print("TimesFM model loaded successfully!")

# --- 4. Example usage (Forecasting a simple sine wave) ---
# This part is crucial to actually use the model after initialization.
# For RUL, this would be your pre-processed Health Indicator data.

total_points = context_length + horizon_length * 2
time = np.arange(total_points)
data = 5 * np.sin(time / 10) + 2 * np.cos(time / 5) + np.random.normal(0, 0.5, total_points)

historical_data = data[:context_length]
# TimesFM expects a list of arrays for forecast_input.
forecast_input = [historical_data]
frequency_input = [0] # 0 for high frequency (e.g., sensor data)

print("\nPerforming forecast...")
point_forecast, _ = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)

predicted_future = point_forecast[0]

print("Forecast complete!")
print(f"Shape of predicted future: {predicted_future.shape}")

# --- 5. Visualize (Optional) ---
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(historical_data)), historical_data, label='Historical Data', color='blue')
plt.plot(np.arange(len(historical_data), len(historical_data) + len(predicted_future)), predicted_future, label='TimesFM Forecast', color='red', linestyle='--')
plt.title('TimesFM Forecasting Example')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()