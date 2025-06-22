import numpy as np # Added for data generation
import timesfm
import matplotlib.pyplot as plt # Added for plotting
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

model.train()

# Example: dummy data
X = torch.randn(100, 32, 1)  # (batch, context_len, features)
y = torch.randn(100, 8, 1)   # (batch, horizon_len, features)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_x, prediction_length=batch_y.shape[1])  # Call with prediction_length
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")