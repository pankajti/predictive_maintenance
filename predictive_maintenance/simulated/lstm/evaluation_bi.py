import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from predictive_maintenance.simulated.lstm.model import RULPredictorBiLSTM
from predictive_maintenance.simulated.lstm.data_prep import get_data_loaders
from predictive_maintenance.simulated.lstm.utils import get_device

MODEL_SAVE_PATH = 'best_rul_model_deeper.pth'

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(targets.squeeze().cpu().numpy())

    return np.array(predictions), np.array(actuals)

def plot_rul_results(predictions, actuals):
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predicted RUL')
    plt.plot(actuals, label='Actual RUL')
    plt.xlabel('Sample Index')
    plt.ylabel('Remaining Useful Life')
    plt.title('RUL Prediction vs. Actual')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    print("Loading best model from", MODEL_SAVE_PATH)
    device = get_device()

    model = RULPredictorBiLSTM(input_size=2048, hidden_size=128, num_layers=2, output_size=1, dropout_rate=0.3)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)

    train_dataloader, val_dataloader, test_datasets, test_sim_ids = get_data_loaders()
    predictions, actuals = evaluate_model(model, test_loader, device)

    print("MAE:", mean_absolute_error(actuals, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(actuals, predictions)))
    print("R2 Score:", r2_score(actuals, predictions))

    plot_rul_results(predictions, actuals)

if __name__ == "__main__":
    main()
