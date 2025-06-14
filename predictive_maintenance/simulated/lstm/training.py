import torch
import torch.nn as nn
import torch.optim as optim
from predictive_maintenance.simulated.lstm.model import get_lstm_model,get_bi_lstm_model
import matplotlib.pyplot as plt
from predictive_maintenance.simulated.lstm.data_prep import get_data_loaders
from predictive_maintenance.simulated.lstm.utils import get_device

MODEL_SAVE_PATH = 'best_rul_model_deeper.pth'  # Changed save path


def main ():
    # --- 5. Training Loop with Early Stopping ---
    NUM_EPOCHS = 20
    PATIENCE = 10
    MIN_DELTA = 0.001

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    training_losses = []
    validation_losses = []
    model = get_bi_lstm_model()


    print("\nStarting Training with Early Stopping (Deeper Model)...")

    # --- 4. Loss Function and Optimizer ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = get_device()

    model.to(device)

    train_dataloader, val_dataloader,test_datasets,test_sim_ids = get_data_loaders()

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
    plt.title("Training and Validation Loss Over Epochs (Deeper LSTM with Early Stopping)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()

