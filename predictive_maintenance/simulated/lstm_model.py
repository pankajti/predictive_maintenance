# Simulate time-degrading vibration signal leading to failure
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error  # For evaluation metrics


# --- 1. Simulate Data (from your previous code) ---
def simulate_degrading_signal(sampling_rate=20000, duration=10, window_size_sim=1.024):
    """
    Simulates vibration data over time with increasing defect characteristics
    leading to failure at the end.
    """
    total_windows = int(duration / window_size_sim)
    samples_per_window_sim = int(sampling_rate * window_size_sim)
    full_signal = []
    timestamps = []

    for i in range(total_windows):
        t = np.linspace(0, window_size_sim, samples_per_window_sim)

        # Healthy base signal
        base = 0.1 * np.sin(2 * np.pi * 50 * t)

        # Gradually increasing defect (higher frequency)
        defect_strength = 0.01 + (i / total_windows) * 0.1
        defect = defect_strength * np.sin(2 * np.pi * 1200 * t)

        # Gradually increasing impulse noise
        impulses = np.zeros_like(t)
        num_impulses = 2 + int((i / total_windows) * 10)
        impulse_indices = np.random.choice(samples_per_window_sim, size=num_impulses, replace=False)
        impulses[impulse_indices] = np.random.uniform(-1.0, 1.0, size=num_impulses)

        signal = base + defect + impulses
        full_signal.append(signal)
        timestamps.extend([i * window_size_sim + x for x in t])

    full_signal = np.concatenate(full_signal)
    timestamps = np.array(timestamps)
    return timestamps, full_signal, duration


# Simulate degrading signal over 10 seconds
timestamps, degrading_signal, total_duration = simulate_degrading_signal(duration=10)

# Calculate RUL
rul = total_duration - timestamps

# Create DataFrame
degrading_df = pd.DataFrame({'time_sec': timestamps, 'amplitude': degrading_signal, 'RUL_sec': rul})

print("Data simulation complete. Head of DataFrame:")
print(degrading_df.head())
print("\nTail of DataFrame:")
print(degrading_df.tail())

# --- 2. PyTorch Data Preparation ---

# Define windowing parameters for the model
# For LSTM, the input sequence length is important
WINDOW_SIZE_SAMPLES = 2048  # Number of samples in each input window
STEP_SIZE_SAMPLES = 200  # How much the window slides (can be overlapping)


# Custom Dataset for Time Series
class VibrationDataset(Dataset):
    def __init__(self, data_df, window_size, step_size):
        self.data_df = data_df
        self.window_size = window_size
        self.step_size = step_size
        self.inputs = []
        self.targets = []

        # Generate windows
        num_samples = len(data_df)
        for i in range(0, num_samples - window_size + 1, step_size):
            window_end_idx = i + window_size

            # Input: Amplitude values within the window
            # Convert to float32 for PyTorch
            input_window = torch.tensor(self.data_df['amplitude'].iloc[i:window_end_idx].values, dtype=torch.float32)
            self.inputs.append(input_window)

            # Target: RUL at the end of the window
            # Ensure RUL is also float32
            target_rul = torch.tensor(self.data_df['RUL_sec'].iloc[window_end_idx - 1], dtype=torch.float32)
            self.targets.append(target_rul)

        print(f"Generated {len(self.inputs)} windows for the dataset.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# Create dataset
dataset = VibrationDataset(degrading_df, window_size=WINDOW_SIZE_SAMPLES, step_size=STEP_SIZE_SAMPLES)

# Create DataLoader
BATCH_SIZE = 64
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# --- 3. PyTorch Model Definition (LSTM Example) ---
class RULPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RULPredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer: input_size is the feature dimension (1 for raw amplitude)
        # hidden_size is the number of features in the hidden state h
        # batch_first=True means input tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to map LSTM output to RUL
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length) -> e.g., (64, 2048)
        # LSTM expects input shape: (batch_size, sequence_length, input_size)
        # input_size here is 1 because each time step has one feature (amplitude)
        x = x.unsqueeze(2)  # Add feature dimension: (batch_size, sequence_length, 1)

        # Initialize hidden and cell states (can also be done with zeros by default if not passed)
        # (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_size)
        # _: (hn, cn) - hn is hidden state for last time step

        # Get the output from the last time step
        # out[:, -1, :] takes the hidden state of the last time step for all batches
        out = self.fc(out[:, -1, :])

        return out


# Instantiate the model
INPUT_SIZE = 1  # Each time step has 1 feature (amplitude)
HIDDEN_SIZE = 128  # Number of features in the hidden state
NUM_LAYERS = 2  # Number of recurrent layers
OUTPUT_SIZE = 1  # Predicting a single RUL value

model = RULPredictorLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
print(f"\nModel Architecture:\n{model}")

# --- 4. Loss Function and Optimizer ---
criterion = nn.MSELoss()  # Mean Squared Error for regression task
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. Training Loop ---
NUM_EPOCHS = 20  # You might need more or fewer epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\nUsing device: {device}")

training_losses = []

print("\nStarting Training...")
for epoch in range(NUM_EPOCHS):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss (ensure targets are of the same shape as outputs)
        loss = criterion(outputs.squeeze(), targets)  # Squeeze output to match target shape

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_epoch_loss = running_loss / len(dataloader)
    training_losses.append(avg_epoch_loss)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.4f}")

print("Training finished.")

# Plotting training loss
plt.figure(figsize=(10, 5))
plt.plot(training_losses)
plt.title("Training Loss Over Epochs (LSTM)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

# --- 6. Evaluation (Making Predictions) ---
model.eval()  # Set model to evaluation mode
all_predictions = []
all_true_ruls = []

with torch.no_grad():  # Disable gradient calculations during evaluation
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)

        all_predictions.extend(outputs.squeeze().cpu().numpy())
        all_true_ruls.extend(targets.cpu().numpy())

# Convert to numpy arrays for easier plotting/analysis
all_predictions = np.array(all_predictions)
all_true_ruls = np.array(all_true_ruls)

# Plotting predictions vs true RUL
plt.figure(figsize=(12, 6))
plt.plot(all_true_ruls, label='True RUL')
plt.plot(all_predictions, label='Predicted RUL', alpha=0.7)
plt.title("True vs. Predicted RUL (LSTM)")
plt.xlabel("Window Index (approximates Time Progression)")
plt.ylabel("RUL (seconds)")
plt.legend()
plt.grid(True)
plt.show()

# Calculate metrics
mae = mean_absolute_error(all_true_ruls, all_predictions)
rmse = np.sqrt(mean_squared_error(all_true_ruls, all_predictions))

print(f"\nMean Absolute Error (MAE): {mae:.4f} seconds")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} seconds")