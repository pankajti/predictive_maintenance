import torch
import torch.nn as nn
import torch.optim as optim




# --- 3. PyTorch Model Definition (LSTM Example - IMPROVED LAYERS) ---
class RULPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(RULPredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer: increased num_layers, added dropout to LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)

        # Added an additional fully connected layer
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)  # Reduce dimensions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Dropout after ReLU

        self.fc2 = nn.Linear(hidden_size // 2, output_size)  # Final output layer

    def forward(self, x):
        x = x.unsqueeze(2)  # Add feature dimension: (batch_size, sequence_length, 1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        out = out[:, -1, :]

        # Pass through the new fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)  # Final prediction
        return out


import torch
import torch.nn as nn

# --- Enhanced BiLSTM Model with LayerNorm and Dropout ---
class RULPredictorBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(RULPredictorBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate,
                            bidirectional=True)

        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout_rate)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if missing
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, hidden*2)
        last_time_step = lstm_out[:, -1, :]  # Take last time step
        normed = self.norm(last_time_step)
        dropped = self.dropout(normed)
        return self.regressor(dropped)



def get_lstm_model():
    INPUT_SIZE = 1
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3  # Increased from 2 to 3 LSTM layers
    OUTPUT_SIZE = 1
    DROPOUT_RATE = 0.5  # Dropout rate for LSTM and FC layers

    model = RULPredictorLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT_RATE)
    print(f"\nModel Architecture:\n{model}")


    return model

def get_bi_lstm_model():
    model = RULPredictorBiLSTM(
        input_size=2048,  # or the actual size of each input window
        hidden_size=128,
        num_layers=2,
        output_size=1,
        dropout_rate=0.3
    )
    return  model

if __name__ == '__main__':

    model = get_lstm_model()