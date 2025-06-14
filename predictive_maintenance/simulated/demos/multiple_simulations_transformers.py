import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import math  # For positional encoding


# --- 1. Simulate Data (Modified for Varied Simulations) ---
def simulate_degrading_signal_varied(sampling_rate=20000, duration=10, window_size_sim=1.024):
    """
    Simulates vibration data over time with increasing defect characteristics
    leading to failure at the end, with added randomness in parameters.
    """
    total_windows = int(duration / window_size_sim)
    samples_per_window_sim = int(sampling_rate * window_size_sim)
    full_signal = []
    timestamps = []

    # Randomize degradation parameters for each simulation run
    base_defect_amplitude = np.random.uniform(0.005, 0.015)
    final_defect_amplitude = np.random.uniform(0.08, 0.15)
    defect_frequency = np.random.uniform(1150, 1250)

    initial_impulses = np.random.randint(1, 4)
    max_impulses_increase = np.random.randint(8, 15)

    for i in range(total_windows):
        t = np.linspace(0, window_size_sim, samples_per_window_sim)

        base = 0.1 * np.sin(2 * np.pi * 50 * t)
        defect_strength = base_defect_amplitude + (i / total_windows) * (final_defect_amplitude - base_defect_amplitude)
        defect = defect_strength * np.sin(2 * np.pi * defect_frequency * t)
        impulses = np.zeros_like(t)
        num_impulses = initial_impulses + int((i / total_windows) * max_impulses_increase)
        impulse_indices = np.random.choice(samples_per_window_sim, size=num_impulses, replace=False)
        impulses[impulse_indices] = np.random.uniform(-1.0, 1.0, size=num_impulses)

        signal = base + defect + impulses
        full_signal.append(signal)
        timestamps.extend([i * window_size_sim + x for x in t])

    full_signal = np.concatenate(full_signal)
    timestamps = np.array(timestamps)
    return timestamps, full_signal, duration


# --- Generate Multiple Instances and Create a List of DataFrames ---
num_simulations = 15
list_of_degrading_dfs = []

print(f"Generating {num_simulations} simulated data instances...")
for i in range(num_simulations):
    timestamps_i, degrading_signal_i, total_duration_i = simulate_degrading_signal_varied(sampling_rate=1000, duration=10 + i * 0.5)
    rul_i = total_duration_i - timestamps_i

    df_i = pd.DataFrame({
        'simulation_id': i,
        'time_sec': timestamps_i,
        'amplitude': degrading_signal_i,
        'RUL_sec': rul_i
    })
    list_of_degrading_dfs.append(df_i)

print(f"\nGenerated {len(list_of_degrading_dfs)} individual DataFrames.")

# --- 2. PyTorch Data Preparation (Dataset with timestamp) ---

WINDOW_SIZE_SAMPLES = 2048  # This is our sequence length
STEP_SIZE_SAMPLES = 200


class VibrationDataset(Dataset):
    def __init__(self, data_df, window_size, step_size):
        self.data_df = data_df
        self.window_size = window_size
        self.step_size = step_size
        self.inputs = []
        self.targets = []
        self.window_end_times = []

        num_samples = len(data_df)
        for i in range(0, num_samples - window_size + 1, step_size):
            window_end_idx = i + window_size
            input_window = torch.tensor(self.data_df['amplitude'].iloc[i:window_end_idx].values, dtype=torch.float32)
            self.inputs.append(input_window)

            target_rul = torch.tensor(self.data_df['RUL_sec'].iloc[window_end_idx - 1], dtype=torch.float32)
            self.targets.append(target_rul)

            window_end_time = self.data_df['time_sec'].iloc[window_end_idx - 1]
            self.window_end_times.append(window_end_time)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.window_end_times[idx]


# --- Create Datasets and DataLoaders for Training, Validation, and Testing ---

train_sim_ids = list(range(10))
val_sim_ids = list(range(10, 13))
test_sim_ids = list(range(13, 15))

train_datasets = []
for sim_id in train_sim_ids:
    df_sim = list_of_degrading_dfs[sim_id]
    train_datasets.append(VibrationDataset(df_sim, window_size=WINDOW_SIZE_SAMPLES, step_size=STEP_SIZE_SAMPLES))
train_dataset_combined = torch.utils.data.ConcatDataset(train_datasets)
train_dataloader = DataLoader(train_dataset_combined, batch_size=64, shuffle=True)

val_datasets = []
for sim_id in val_sim_ids:
    df_sim = list_of_degrading_dfs[sim_id]
    val_datasets.append(VibrationDataset(df_sim, window_size=WINDOW_SIZE_SAMPLES, step_size=STEP_SIZE_SAMPLES))
val_dataset_combined = torch.utils.data.ConcatDataset(val_datasets)
val_dataloader = DataLoader(val_dataset_combined, batch_size=64, shuffle=False)

test_datasets = []
for sim_id in test_sim_ids:
    df_sim = list_of_degrading_dfs[sim_id]
    test_datasets.append(VibrationDataset(df_sim, window_size=WINDOW_SIZE_SAMPLES, step_size=STEP_SIZE_SAMPLES))

print(f"\nTraining on {len(train_sim_ids)} simulations. Total training windows: {len(train_dataset_combined)}")
print(f"Validation on {len(val_sim_ids)} simulations. Total validation windows: {len(val_dataset_combined)}")
print(f"Testing on {len(test_sim_ids)} simulations.")


# --- 3. PyTorch Model Definition (Transformer Encoder for Time Series) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        # Add positional encoding
        x = x + self.pe[:, :x.size(1)]  # Only use up to sequence length
        return x


class RULPredictorTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout_rate=0.1):
        super(RULPredictorTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Input embedding: Map univariate time series to d_model
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=WINDOW_SIZE_SAMPLES)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Regression Head
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, src):
        # src shape: (batch_size, sequence_length) e.g., (64, 2048)

        # Add feature dimension and embed
        # (batch_size, sequence_length, input_dim) -> (batch_size, sequence_length, d_model)
        src = src.unsqueeze(-1)  # Add last dimension for input_dim=1
        src = self.embedding(src)  # (batch_size, sequence_length, d_model)

        # Add positional encoding
        src = self.positional_encoding(src)  # (batch_size, sequence_length, d_model)

        # Pass through Transformer Encoder
        # Output of encoder is (batch_size, sequence_length, d_model)
        output = self.transformer_encoder(src)

        # For regression, we can take the output of the last token (time step)
        # or average/pool over the sequence dimension.
        # Taking the representation of the last element in the sequence
        output = output[:, -1, :]  # (batch_size, d_model)

        # Pass through regression head
        output = self.fc_out(output)  # (batch_size, output_dim)

        return output


# Hyperparameters for Transformer
INPUT_DIM = 1  # Univariate time series (amplitude)
D_MODEL = 64  # Embedding dimension / feature dimension in Transformer
NHEAD = 8  # Number of attention heads
NUM_ENCODER_LAYERS = 4  # Number of Transformer encoder layers
DIM_FEEDFORWARD = 128  # Dimension of the feedforward network model in Transformer layers
OUTPUT_DIM = 1  # RUL prediction
DROPOUT_RATE = 0.1  # Dropout rate

model = RULPredictorTransformer(INPUT_DIM, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, OUTPUT_DIM,
                                DROPOUT_RATE)
print(f"\nModel Architecture (Transformer):\n{model}")

# --- 4. Loss Function and Optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate for Transformers sometimes helps

# --- Device Setup (for Mac M3) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU) for acceleration.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU) for acceleration.")
else:
    device = torch.device("cpu")
    print("Using CPU for training.")

device = torch.device("cpu")
print(device)

model.to(device)

# --- 5. Training Loop with Early Stopping ---
NUM_EPOCHS = 100
PATIENCE = 15  # Slightly increased patience as Transformers can be slower to converge
MIN_DELTA = 0.001

best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

training_losses = []
validation_losses = []

MODEL_SAVE_PATH = 'best_rul_model_transformer.pth'

print("\nStarting Training with Early Stopping (Transformer Model)...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets, _) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_dataloader)
    training_losses.append(avg_train_loss)

    # --- Validation Phase ---
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for inputs, targets, _ in val_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs.squeeze(), targets)
            val_running_loss += val_loss.item()

    avg_val_loss = val_running_loss / len(val_dataloader)
    validation_losses.append(avg_val_loss)

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # --- Early Stopping Logic ---
    if avg_val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  --> Best model saved with validation loss: {best_val_loss:.4f}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            early_stop = True
            break

    if early_stop:
        break

print("Training finished.")

# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.title("Training and Validation Loss Over Epochs (Transformer with Early Stopping)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

# --- 6. Load Best Model and Evaluate on Test Simulations ---
print(f"\nLoading best model from {MODEL_SAVE_PATH} for final evaluation...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

print("\nStarting Evaluation on Test Simulations...")

all_test_predictions = []
all_test_true_ruls = []
all_test_window_end_times = []

# Note: BATCH_SIZE is not defined globally, using a default value or inferring from DataLoader
# For test evaluation, we can use a smaller batch size or just process one by one if preferred.
# Here, we'll re-use the train_dataloader's batch size of 64 for evaluation as well.
BATCH_SIZE = 64

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
plt.title("True vs. Predicted RUL (Transformer) on Unseen Test Simulations")
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