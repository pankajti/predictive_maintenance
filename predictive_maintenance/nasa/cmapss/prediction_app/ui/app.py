
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import torch
import asyncio
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm

from nicegui import ui

try:
    from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
except ImportError as e:
    import sys
    print("The 'timesfm' library is not installed. Please install it using 'pip install timesfm'.", file=sys.stderr)
    raise e

# --- Helper Functions ---

def nasa_rul_score(y_true, y_pred):
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))

def plot_predictions(nasa_score, r2, rmse, y_pred_final, y_true_final, dataset_id):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_true_final, label='True RUL', color='blue')
    ax.plot(y_pred_final, label='Predicted RUL', color='red', linestyle='--')
    ax.set_title(f'CMAPSS FD00{dataset_id} RUL Prediction (RMSE: {rmse:.2f}, R2: {r2:.2f}, NASA Score: {nasa_score:.2f})')
    ax.set_xlabel('Engine Unit Index (Test Set)')
    ax.set_ylabel('RUL (Cycles)')
    ax.legend()
    ax.grid(True)
    return fig

def get_timesfm_model(CONTEXT_LEN, HORIZON_LEN):
    hparams = TimesFmHparams(
        backend="cpu",
        context_len=CONTEXT_LEN,
        horizon_len=HORIZON_LEN,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
    )
    checkpoint = TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
    return TimesFm(hparams=hparams, checkpoint=checkpoint)

def train_rul_regressor(train_df_pcs, y_reg_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(train_df_pcs, y_reg_train)
    return model

def predict_rul(rul_regressor, rul_test_df, test_df_pcs_only, tfm, N_COMPONENTS, CONTEXT_LEN, FREQ):
    predicted_ruls = []
    for unit_no in tqdm(test_df_pcs_only['unit_number'].unique(), desc="Predicting RUL"):
        unit_df = test_df_pcs_only[test_df_pcs_only['unit_number'] == unit_no].copy()
        if len(unit_df) < CONTEXT_LEN:
            pad_len = CONTEXT_LEN - len(unit_df)
            pad = np.tile(unit_df[[f'PC_{i + 1}' for i in range(N_COMPONENTS)]].iloc[0].values, (pad_len, 1))
            pcs = np.vstack([pad, unit_df[[f'PC_{i + 1}' for i in range(N_COMPONENTS)]].values])[-CONTEXT_LEN:]
        else:
            pcs = unit_df[[f'PC_{i + 1}' for i in range(N_COMPONENTS)]].iloc[-CONTEXT_LEN:].values

        input_tensor = torch.tensor(pcs, dtype=torch.float32).T.unsqueeze(0)
        forecast_pcs, _ = tfm.forecast(input_tensor.squeeze(0), freq=[FREQ] * N_COMPONENTS)
        pred_pc_vector = forecast_pcs.T[-1:, :]
        predicted_ruls.append(rul_regressor.predict(pred_pc_vector)[0])

    y_pred = np.maximum(0, np.array(predicted_ruls)).flatten()
    y_true = rul_test_df['RUL'].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return r2, rmse, y_pred, y_true

def load_and_preprocess_data(dataset_id, data_path, all_columns, rul_threshold):
    train_path = os.path.join(data_path, f'train_FD00{dataset_id}.txt')
    test_path = os.path.join(data_path, f'test_FD00{dataset_id}.txt')
    rul_path = os.path.join(data_path, f'RUL_FD00{dataset_id}.txt')
    train_df = pd.read_csv(train_path, sep=' ', header=None, names=all_columns, index_col=False)
    test_df = pd.read_csv(test_path, sep=' ', header=None, names=all_columns, index_col=False)
    rul_df = pd.read_csv(rul_path, sep=' ', header=None, names=['RUL'], index_col=False)

    # for df in (train_df, test_df):
    #     df.drop(columns=['sensor_20', 'sensor_21'], inplace=True, errors='ignore')

    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    train_df = train_df.merge(max_cycles.rename(columns={'time_in_cycles': 'max_cycles'}), on='unit_number')
    train_df['RUL'] = (train_df['max_cycles'] - train_df['time_in_cycles']).clip(upper=rul_threshold)
    train_df.drop(columns='max_cycles', inplace=True)

    return train_df, test_df, rul_df

def apply_pca_and_add_pcs(train_df, test_df, selected_features, n_components):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[selected_features])
    test_scaled = scaler.transform(test_df[selected_features])

    pca = PCA(n_components=n_components)
    train_pcs = pca.fit_transform(train_scaled)
    test_pcs = pca.transform(test_scaled)

    pc_cols = [f'PC_{i + 1}' for i in range(n_components)]
    train_df[pc_cols] = train_pcs
    test_df[pc_cols] = test_pcs
    return train_df, test_df

ALL_COLUMNS = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
SELECTABLE_FEATURES_FOR_UI = [col for col in ALL_COLUMNS if col not in ['unit_number', 'time_in_cycles']]
DEFAULT_FEATURES_FOR_UI = [f for f in [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17',
    'sensor_20', 'sensor_21'
] if f in SELECTABLE_FEATURES_FOR_UI]

import plotly.graph_objects as go

def plot_predictions_plotly(nasa_score, r2, rmse, y_pred_final, y_true_final, dataset_id):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true_final, mode='lines', name='True RUL', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=y_pred_final, mode='lines', name='Predicted RUL', line=dict(color='red', dash='dash')))
    fig.update_layout(
        title=f'CMAPSS FD00{dataset_id} RUL Prediction',
        xaxis_title='Engine Unit Index (Test Set)',
        yaxis_title='RUL (Cycles)',
        legend_title='Legend',
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        annotations=[
            dict(
                text=f"RMSE: {rmse:.2f}, R2: {r2:.2f}, NASA Score: {nasa_score:.2f}",
                xref="paper", yref="paper",
                x=0, y=1.1, showarrow=False
            )
        ]
    )
    return fig

async def run_prediction_pipeline(dataset_id, selected_features, context_len, horizon_len, freq,
                                  n_components, rul_threshold, data_path, results_label, plot_card):
    async def update_ui(content=None, notify_message=None, notify_type='info', clear_plot=False, plot_fig=None):
        if content:
            results_label.set_content(content)
        if notify_message:
            ui.notify(notify_message, type=notify_type, timeout=7000)
        if clear_plot:
            plot_card.clear()
        if plot_fig:
            with plot_card:
                with ui.pyplot() as p:
                    p.figure = plot_fig

    await update_ui(content="Running prediction... Please wait.", clear_plot=True)
    with plot_card:
        ui.label("Processing...").classes('text-gray-500')

    loop = asyncio.get_running_loop()
    try:
        train_df, test_df, rul_df = await loop.run_in_executor(None, load_and_preprocess_data,
                                                               dataset_id, data_path, ALL_COLUMNS, rul_threshold)
        await update_ui(notify_message="Data loaded and preprocessed.")

        train_df, test_df = await loop.run_in_executor(None, apply_pca_and_add_pcs,
                                                       train_df, test_df, selected_features, n_components)
        await update_ui(notify_message="PCA applied.")

        X_train = train_df[[f'PC_{i + 1}' for i in range(n_components)]].values
        y_train = train_df['RUL'].values
        regressor = await loop.run_in_executor(None, train_rul_regressor, X_train, y_train)
        tfm = await loop.run_in_executor(None, get_timesfm_model, context_len, horizon_len)

        test_df_pcs = test_df[['unit_number'] + [f'PC_{i + 1}' for i in range(n_components)]]
        r2, rmse, y_pred, y_true = await loop.run_in_executor(None, predict_rul,
                                                              regressor, rul_df, test_df_pcs, tfm,
                                                              n_components, context_len, freq)

        score = nasa_rul_score(y_true, y_pred)
        await update_ui(content=f"<h3>Evaluation Metrics</h3><ul><li>RMSE: {rmse:.2f}</li><li>R2: {r2:.2f}</li><li>NASA Score: {score:.2f}</li></ul>",
                        notify_message="Prediction complete.",
                        notify_type='positive',
                        plot_fig=plot_predictions(score, r2, rmse, y_pred, y_true, dataset_id))

    except Exception as e:
        await update_ui(content=f"Error: {str(e)}", notify_message="An error occurred.", notify_type='negative', clear_plot=True)

@ui.page('/')
async def main_page():
    with ui.header(elevated=True).classes('bg-blue-600 text-white'):
        ui.label('CMAPSS RUL Prediction Dashboard').classes('text-3xl font-bold')

    with ui.card().classes('w-full max-w-4xl mx-auto my-6 p-6'):
        ui.label('Configure Prediction').classes('text-2xl font-semibold text-gray-800 mb-4')
        with ui.grid(columns=2).classes('w-full gap-4'):
            with ui.column():
                dataset_id_input = ui.select(options={1: 'FD001', 2: 'FD002', 3: 'FD003', 4: 'FD004'}, value=2, label='Dataset ID')
                features_input = ui.select(options=SELECTABLE_FEATURES_FOR_UI, value=DEFAULT_FEATURES_FOR_UI,
                                           multiple=True, label='Features', validation={'Select at least one': lambda val: len(val) > 0})
                data_path_input = ui.input(label='CMAPSS Data Path', value='CMAPSSData')
            with ui.column():
                context_input = ui.number(label='Context Length', value=64, min=32)
                horizon_input = ui.number(label='Horizon Length', value=16, min=1)
                n_comp_input = ui.number(label='PCA Components', value=10, min=1)
                rul_thresh_input = ui.number(label='RUL Threshold', value=130, min=0)
                freq_input = ui.number(label='TimesFM Frequency', value=1, min=0, max=2)

        run_button = ui.button('Run Prediction', icon='play_arrow', on_click=lambda: run_prediction_pipeline(
            dataset_id_input.value, features_input.value, int(context_input.value), int(horizon_input.value),
            int(freq_input.value), int(n_comp_input.value), int(rul_thresh_input.value), data_path_input.value,
            results_label, plot_card)).classes('w-full bg-green-600 text-white')

    ui.separator()
    results_label = ui.markdown("Results will appear here.")
    ui.separator()
    with ui.card() as plot_card:
        ui.label("Prediction plot will appear here.")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run()
