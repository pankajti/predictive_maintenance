# predictive_maintenance


## Sources refered to understand predictive maintenance

https://www.youtube.com/watch?v=BPMjYJ_HoWk&list=PLDNHqPpwBs8O2QIGHdi8Bwu3p-WbTLsXG

AN INTRODUCTION  TO PREDICTIVE  MAINTENANCE Second Edition
R. Keith Mobley

# Remaining Useful Life (RUL) Prediction with LSTM

This project implements a Long Short-Term Memory (LSTM) neural network to predict the Remaining Useful Life (RUL) of a machine based on simulated vibration data. The code simulates degrading signals, prepares data for training and testing, trains an LSTM model, and evaluates its performance on unseen data.

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Data Simulation](#data-simulation)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [License](#license)

## Project Overview
The goal is to predict the RUL of a machine using vibration signals that degrade over time, simulating real-world mechanical wear. The project:
- Generates synthetic vibration data with increasing defect characteristics.
- Uses PyTorch to create a custom dataset and train an LSTM model.
- Evaluates the model on unseen simulations to mimic real-world generalization.
- Visualizes training loss and prediction performance with metrics like MAE, RMSE, and R-squared.

## Dependencies
To run the code, install the following Python libraries:
```bash
pip install torch pandas numpy matplotlib scikit-learn
```

- **Python**: 3.8+
- **PyTorch**: For LSTM model and data handling
- **Pandas**: For data manipulation
- **NumPy**: For numerical operations
- **Matplotlib**: For plotting
- **Scikit-learn**: For evaluation metrics

Optional: GPU with MPS support (Apple silicon) or CUDA for faster training.

## Usage
1. Clone or download the repository.
2. Install the required dependencies (see above).
3. Run the script:
   ```bash
   python rul_prediction_lstm.py
   ```
4. The script will:
   - Generate 10 simulated datasets.
   - Train the LSTM model on 8 simulations.
   - Evaluate on 2 unseen simulations.
   - Display plots for training loss and test predictions.
   - Print evaluation metrics (MAE, RMSE, R-squared).

## Code Structure
- **Data Simulation**: Generates synthetic vibration signals with increasing defects.
- **Dataset Preparation**: Creates a custom PyTorch `VibrationDataset` for windowed data.
- **Model Definition**: Defines an LSTM-based `RULPredictorLSTM` model.
- **Training Loop**: Trains the model using MSE loss and Adam optimizer.
- **Evaluation**: Tests on unseen simulations and computes performance metrics.
- **Visualization**: Plots training loss and true vs. predicted RUL.

Key parameters:
- `WINDOW_SIZE_SAMPLES`: 2048 (window size for input data)
- `STEP_SIZE_SAMPLES`: 200 (step size for sliding window)
- `NUM_EPOCHS`: 20 (training epochs)
- `HIDDEN_SIZE`: 128 (LSTM hidden units)
- `NUM_LAYERS`: 2 (LSTM layers)

## Data Simulation
The `simulate_degrading_signal_varied` function generates vibration signals with:
- Randomized defect amplitude, frequency, and impulse characteristics.
- 10 simulations with durations from 10 to 14.5 seconds.
- Sampling rate: 20,000 Hz.
- Output: DataFrames with `time_sec`, `amplitude`, `RUL_sec`, and `simulation_id`.

Each simulation mimics a machine degrading until failure, with RUL calculated as the time remaining until the end of the simulation.

## Model Architecture
The `RULPredictorLSTM` model is an LSTM network with:
- Input size: 1 (single amplitude value per time step)
- Hidden size: 128
- Number of layers: 2
- Output size: 1 (predicted RUL)
- Fully connected layer to map LSTM output to RUL.

The model processes sequences of 2048 samples and predicts the RUL for each window.

## Training and Evaluation
- **Training**: Uses 8 simulations (IDs 0–7) with a batch size of 64 and shuffling.
- **Testing**: Evaluates on 2 unseen simulations (IDs 8–9) without shuffling to preserve temporal order.
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam with a learning rate of 0.001.
- **Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared.

## Results
After training, the script generates:
- A plot of training loss over epochs.
- A plot comparing true vs. predicted RUL for test simulations.
- Test set metrics (e.g., MAE, RMSE, R-squared).

Example output:
```
Overall Test Set Metrics (on 2 unseen simulations):
  MAE: X.XXXX seconds
  RMSE: X.XXXX seconds
  R-squared: X.XXXX
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.