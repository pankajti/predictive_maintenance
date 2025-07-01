
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor # Example regression model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
# Import TimesFM
import timesfm
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
import torch # TimesFM uses PyTorch backend
from tqdm import tqdm


def main():
    CONTEXT_LEN = 64
    HORIZON_LEN = 16
    FREQ = 1
    N_COMPONENTS = 10
    RUL_THRESHOLD = 130
    DATA_PATH = (r''
                 r'/Users/pankajti/dev/data/kaggle/nasa/CMaps')
    columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    dataset_id = 2;
    run_model(dataset_id, context_len= CONTEXT_LEN, horizon_len= HORIZON_LEN, freq= FREQ, n_components= N_COMPONENTS,
              rul_threshold= RUL_THRESHOLD, data_path= DATA_PATH, columns = columns)

def run_model(dataset_id, context_len, horizon_len,
              freq,
              n_components,
              rul_threshold,
              data_path, columns):

    features_to_use, rul_test_df, test_df, train_df = load_and_preprocess_data(dataset_id, data_path, columns, rul_threshold)
    predict_rul_from_features(features_to_use, test_df, train_df, n_components)
    tfm = get_timesfm_model(context_len, horizon_len)
    rul_regressor = train_rul_regressor(train_df, n_components)
    r2, rmse, y_pred_final, y_true_final = predict_rul(rul_regressor, rul_test_df, test_df, tfm, n_components, context_len, freq)
    nasa_score = nasa_rul_score(y_true_final, y_pred_final)
    plot_predictions(nasa_score, r2, rmse, y_pred_final, y_true_final)
    ret_val = {"r2": r2, "rmse": rmse, "nasa_score": nasa_score,}
    print(ret_val)
    return  ret_val


def plot_predictions(nasa_score, r2, rmse, y_pred_final, y_true_final):
    print(f"\n--- Model Evaluation (TimesFM + PCA) ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print(f"NASA RUL Score: {nasa_score:.2f}")
    # Plotting Results
    plt.figure(figsize=(15, 6))
    plt.plot(y_true_final, label='True RUL', color='blue')
    plt.plot(y_pred_final, label='Predicted RUL', color='red', linestyle='--')
    plt.title('CMAPSS FD001 RUL Prediction (TimesFM + PCA)')
    plt.xlabel('Engine Unit Index (Test Set)')
    plt.ylabel('RUL (Cycles)')
    plt.legend()
    plt.grid(True)
    plt.show()


# NASA RUL Scoring Function
def nasa_rul_score(y_true, y_pred):
    d = y_pred - y_true
    score = np.sum(np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1))
    return score


def predict_rul(rul_regressor, rul_test_df, test_df, tfm,N_COMPONENTS,CONTEXT_LEN,FREQ):
    predicted_ruls = []
    # Iterate through each test engine
    for unit_no in tqdm(test_df['unit_number'].unique()):
        unit_df = test_df[test_df['unit_number'] == unit_no].copy()

        # Get the last CONTEXT_LEN PCs for this unit
        # Ensure there are enough cycles to form the context window
        if len(unit_df) < CONTEXT_LEN:
            # Pad if the unit has fewer cycles than context_len
            # Simple padding: repeat the first available PC vector
            padding_needed = CONTEXT_LEN - len(unit_df)
            last_n_pcs = np.vstack([unit_df[[f'PC_{i + 1}' for i in range(N_COMPONENTS)]].iloc[0].values[np.newaxis,
                                    :] * np.ones((padding_needed, N_COMPONENTS)),
                                    unit_df[[f'PC_{i + 1}' for i in range(N_COMPONENTS)]].values])[-CONTEXT_LEN:]
        else:
            last_n_pcs = unit_df[[f'PC_{i + 1}' for i in range(N_COMPONENTS)]].iloc[-CONTEXT_LEN:].values

        # TimesFM expects input shape (batch_size, num_channels, context_len)
        # Our `last_n_pcs` is (context_len, num_channels)
        # Need to reshape to (1, num_channels, context_len)
        timesfm_input = torch.tensor(last_n_pcs, dtype=torch.float32).T.unsqueeze(0)  # Transpose and add batch dim

        # Forecast future PCs using TimesFM
        # The output is (batch_size, num_channels, horizon_len)
        forecast_pcs_torch, _ = tfm.forecast(timesfm_input.squeeze(), freq=[FREQ] * N_COMPONENTS)
        forecast_pcs = forecast_pcs_torch  # Convert back to (horizon_len, num_channels)
        # Predict RUL using the trained regression model
        predicted_rul = rul_regressor.predict(forecast_pcs.T)
        most_Recent_predicted_rul = predicted_rul[-1]
        predicted_ruls.append(most_Recent_predicted_rul)

    y_pred_final = np.maximum(0, np.array(predicted_ruls)).flatten()
    # Get true RUL for comparison
    y_true_final = rul_test_df['RUL'].values
    # --- Evaluation ---
    rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
    r2 = r2_score(y_true_final, y_pred_final)
    return r2, rmse, y_pred_final, y_true_final


def train_rul_regressor(train_df,N_COMPONENTS):
    # Prepare training data for regression model (PC -> RUL mapping)
    # We use the *current* PCs to predict the *current* RUL for training this part
    X_reg_train = train_df[[f'PC_{i + 1}' for i in range(N_COMPONENTS)]].values
    y_reg_train = train_df['RUL'].values
    # --- 5. Regression Model Training (PCs to RUL) ---
    # Using RandomForestRegressor as an example. You could use MLP (Keras) or others.
    rul_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rul_regressor.fit(X_reg_train, y_reg_train)
    print("\nRegression model (PCs to RUL) trained.")
    return rul_regressor


def get_timesfm_model(CONTEXT_LEN,HORIZON_LEN):
    try:
        hparams = TimesFmHparams(
            backend="torch",
            context_len=CONTEXT_LEN,
            horizon_len=HORIZON_LEN,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
        )
        checkpoint = TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
        tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

    except Exception as e:
        print(f"\nError loading TimesFM: {e}")
        print("Please ensure 'timesfm' is installed and checkpoint files are accessible.")
        print("You might need to manually download 'timesfm-1.0-200m-cpu.ckpt' if it doesn't auto-download.")
        exit()
    return tfm


def predict_rul_from_features(features_to_use, test_df, train_df,N_COMPONENTS):
    pca = PCA(n_components=N_COMPONENTS)
    # Create a list of PCs for each engine in training data
    train_pcs_list = []
    train_rul_list = []  # Corresponding RUL for each PC vector
    for unit_no in train_df['unit_number'].unique():
        unit_features = train_df[train_df['unit_number'] == unit_no][features_to_use].values
        unit_rul = train_df[train_df['unit_number'] == unit_no]['RUL'].values

        # Fit PCA on the *entire* training data (all engines features concatenated)
        # This is important: PCA should capture variance across the entire fleet
        if unit_no == train_df['unit_number'].unique()[0]:
            pass
    # Fit PCA on ALL training features (concatenated from all units)
    pca.fit(train_df[features_to_use])
    print(f"\nPCA explained variance ratio (first {N_COMPONENTS} components):")
    print(pca.explained_variance_ratio_[:N_COMPONENTS])
    print(
        f"Total explained variance by {N_COMPONENTS} components: {np.sum(pca.explained_variance_ratio_[:N_COMPONENTS]):.2f}")
    # Transform both train and test features into PCs
    train_pcs = pca.transform(train_df[features_to_use])
    test_pcs = pca.transform(test_df[features_to_use])
    # Add PCs back to dataframes (for easier grouping)
    train_df[[f'PC_{i + 1}' for i in range(N_COMPONENTS)]] = train_pcs
    test_df[[f'PC_{i + 1}' for i in range(N_COMPONENTS)]] = test_pcs
    print("\nTransformed data with PCA components:")


def load_and_preprocess_data(dataset_id,DATA_PATH,columns,RUL_THRESHOLD):
    try:
        train_path = os.path.join(DATA_PATH, f'train_FD00{dataset_id}.txt')
        test_path = os.path.join(DATA_PATH, f'test_FD00{dataset_id}.txt')
        rul_path = os.path.join(DATA_PATH, f'RUL_FD00{dataset_id}.txt')
        train_df = pd.read_csv(train_path, sep=' ', header=None, names=columns, index_col=False)
        test_df = pd.read_csv(test_path, sep=' ', header=None, names=columns, index_col=False)
        rul_test_df = pd.read_csv(rul_path, sep=' ', header=None, names=['RUL'], index_col=False)
    except FileNotFoundError:
        print("CMAPSS data not found. Please ensure 'CMAPSSData' folder is in the same directory as this script.")
        print("Download from: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan")
        exit()  # Exit if files are not found

    print("Train Data Shape:", train_df.shape)
    print("Test Data Shape:", test_df.shape)
    print("RUL Test Data Shape:", rul_test_df.shape)
    # Calculate RUL for training data
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.rename(columns={'time_in_cycles': 'max_time_in_cycles'}, inplace=True)
    train_df = train_df.merge(max_cycles, on='unit_number', how='left')
    train_df['RUL'] = train_df['max_time_in_cycles'] - train_df['time_in_cycles']
    train_df.drop(columns=['max_time_in_cycles'], inplace=True)
    train_df['RUL'] = train_df['RUL'].apply(lambda x: min(x, RUL_THRESHOLD))
    print("\nTrain data with RUL calculated (first 5 rows):")
    # print(train_df.head())
    features_to_use = [col for col in columns[:-2] if col not in ['unit_number', 'time_in_cycles']]
    # Remove features with zero variance in the training data
    constant_features = train_df[features_to_use].std()[train_df[features_to_use].std() == 0].index.tolist()
    if constant_features:
        print(f"\nDropping constant features (zero variance): {constant_features}")
        features_to_use = [f for f in features_to_use if f not in constant_features]
    else:
        print("\nNo constant features found to drop.")
    print(f"Features selected for scaling and PCA: {len(features_to_use)} features.")
    print(features_to_use)
    scaler = MinMaxScaler()
    train_df[features_to_use] = scaler.fit_transform(train_df[features_to_use])
    test_df[features_to_use] = scaler.transform(test_df[features_to_use])
    return features_to_use, rul_test_df, test_df, train_df


if __name__ == '__main__':
    main()

