import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os  # For saving/loading models
from predictive_maintenance.simulated.lstm.training import get_lstm_model,MODEL_SAVE_PATH,get_bi_lstm_model
from predictive_maintenance.simulated.lstm.data_prep import get_data_loaders
from predictive_maintenance.simulated.lstm.utils import get_device


model = get_bi_lstm_model()
train_dataloader, val_dataloader, test_datasets, test_sim_ids = get_data_loaders()
device = get_device()
device = torch.device("cpu")
# --- 6. Load Best Model and Evaluate on Test Simulations ---
print(f"\nLoading best model from {MODEL_SAVE_PATH} for final evaluation...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()
BATCH_SIZE=64
print("\nStarting Evaluation on Test Simulations...")

all_test_predictions = []
all_test_true_ruls = []
all_test_window_end_times = []

with torch.no_grad():
    for sim_idx_in_list, test_dataset in enumerate(test_datasets):
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"  Evaluating Simulation ID: {test_sim_ids[sim_idx_in_list]} with {len(test_dataset)} windows...")

        for inputs, targets, window_end_times in test_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            all_test_predictions.extend(outputs.squeeze().cpu().numpy())
            all_test_true_ruls.extend(targets.cpu().numpy())
            all_test_window_end_times.extend(window_end_times.cpu().numpy())

all_test_predictions = np.array(all_test_predictions)
all_test_true_ruls = np.array(all_test_true_ruls)
all_test_window_end_times = np.array(all_test_window_end_times)

# Plotting predictions vs true RUL for the test set
plt.figure(figsize=(12, 6))
plt.plot(all_test_window_end_times, all_test_true_ruls, label='True RUL', color='blue', alpha=0.7)
plt.plot(all_test_window_end_times, all_test_predictions, label='Predicted RUL', color='red', linestyle='--')
plt.title("True vs. Predicted RUL (Deeper LSTM) on Unseen Test Simulations")
plt.xlabel("Time (seconds)")
plt.ylabel("RUL (seconds)")
plt.legend()
plt.grid(True)
plt.show()

# Calculate metrics for the entire test set
mae_test = mean_absolute_error(all_test_true_ruls, all_test_predictions)
rmse_test = np.sqrt(mean_squared_error(all_test_true_ruls, all_test_predictions))
r2_test = r2_score(all_test_true_ruls, all_test_predictions)

print(f"\nOverall Test Set Metrics (on {len(test_sim_ids)} unseen simulations):")
print(f"  MAE: {mae_test:.4f} seconds")
print(f"  RMSE: {rmse_test:.4f} seconds")
print(f"  R-squared: {r2_test:.4f}")