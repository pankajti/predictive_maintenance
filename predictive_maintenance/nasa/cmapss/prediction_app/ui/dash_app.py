import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import torch
import sys  # Import sys for printing error messages

try:
    from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
except ImportError as e:
    # Print error to stderr and re-raise to indicate a critical dependency is missing
    print("The 'timesfm' library is not installed. Please install it using 'pip install timesfm'.", file=sys.stderr)
    raise e

# --- Configuration and Helper Functions ---

# All possible columns in CMAPSS data
ALL_COLUMNS = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]

# Features that can be selected by the user for PCA.
# Exclude 'unit_number' and 'time_in_cycles' as they are not features for PCA.
SELECTABLE_FEATURES_FOR_UI = [
    col for col in ALL_COLUMNS if col not in ['unit_number', 'time_in_cycles']
]

# Default features to be used in PCA. These are commonly used and generally not constant.
# This list will be the initial selection in the UI.
DEFAULT_FEATURES = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17',
    'sensor_20', 'sensor_21'
]
# Ensure default features are part of selectable features to avoid issues
DEFAULT_FEATURES = [f for f in DEFAULT_FEATURES if f in SELECTABLE_FEATURES_FOR_UI]


def nasa_rul_score(y_true, y_pred):
    """
    Calculates the NASA RUL scoring function.
    Penalizes late predictions more severely than early predictions.
    """
    d = y_pred - y_true
    # Apply asymmetric penalty: exp(-d/13) - 1 for d < 0 (underestimation), exp(d/10) - 1 for d >= 0 (overestimation)
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))


def get_timesfm_model(context_len, horizon_len):
    """
    Initializes and loads the pre-trained TimesFM model.
    Uses CPU backend for broader compatibility in web applications.
    """
    # Define TimesFM hyperparameters based on the pre-trained model (1.0-200m)
    hparams = TimesFmHparams(
        backend="cpu",  # Using CPU for compatibility. Change to "cuda" if a GPU is available and configured.
        context_len=context_len,
        horizon_len=horizon_len,
        input_patch_len=32,  # Specific to TimesFM 1.0-200m
        output_patch_len=128,  # Specific to TimesFM 1.0-200m
        num_layers=20,  # Specific to TimesFM 1.0-200m
        model_dims=1280,  # Specific to TimesFM 1.0-200m
    )
    # Define the checkpoint to load from Hugging Face Hub
    checkpoint = TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
    # Initialize TimesFM model
    return TimesFm(hparams=hparams, checkpoint=checkpoint)


def load_data(dataset_id, data_path, rul_threshold):
    """
    Loads CMAPSS raw data, calculates RUL for training data, and performs initial cleaning.
    """
    train_file = os.path.join(data_path, f'train_FD00{dataset_id}.txt')
    test_file = os.path.join(data_path, f'test_FD00{dataset_id}.txt')
    rul_file = os.path.join(data_path, f'RUL_FD00{dataset_id}.txt')

    # Check if files exist
    if not all(os.path.exists(f) for f in [train_file, test_file, rul_file]):
        missing_files = [f for f in [train_file, test_file, rul_file] if not os.path.exists(f)]
        raise FileNotFoundError(f"Required CMAPSS data files not found: {', '.join(missing_files)}. "
                                "Please ensure the 'CMAPSSData' folder (or specified path) contains all necessary files for FD00{dataset_id}.")

    train_df = pd.read_csv(train_file, sep=' ', header=None, names=ALL_COLUMNS, index_col=False)
    test_df = pd.read_csv(test_file, sep=' ', header=None, names=ALL_COLUMNS, index_col=False)
    rul_test_df = pd.read_csv(rul_file, sep=' ', header=None, names=['RUL'], index_col=False)

    # Drop last two columns which are usually empty/NaN in CMAPSS files
    # This is handled here. If a user selects sensor_20/21, they should exist in the raw data.
    # The 'errors'='ignore' will prevent issues if they are already dropped or not present.
    # train_df.drop(columns=['sensor_20', 'sensor_21'], inplace=True, errors='ignore')
    # test_df.drop(columns=['sensor_20', 'sensor_21'], inplace=True, errors='ignore')

    # Calculate RUL for training data based on max cycles per unit
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    train_df = train_df.merge(max_cycles.rename(columns={'time_in_cycles': 'max_cycles'}), on='unit_number')
    # Apply piecewise linear RUL threshold: RUL is capped at 'rul_threshold'
    train_df['RUL'] = (train_df['max_cycles'] - train_df['time_in_cycles']).clip(upper=rul_threshold)
    train_df.drop(columns='max_cycles', inplace=True)  # Clean up temporary column

    return train_df, test_df, rul_test_df


def apply_pca_and_preprocess_features(train_df, test_df, selected_features, n_components):
    """
    Applies MinMaxScaler and PCA on the selected features.
    Returns DataFrames with added PCA components and updates features_to_use.
    """
    # Validate if all selected features exist in the dataframes
    for df_name, df in [("training", train_df), ("test", test_df)]:
        missing_cols = [col for col in selected_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Selected features not found in {df_name} data: {missing_cols}. "
                             "Please check your selection or data files.")

    # Remove features with zero variance in the *selected training features*
    # This ensures PCA doesn't fail on constant columns from the user's selection
    current_features = [f for f in selected_features if f in train_df.columns]  # Ensure feature is actually in df

    # Filter out features that are constant AFTER data loading, even if user selected them
    constant_features_in_selection = train_df[current_features].std()[
        train_df[current_features].std() == 0].index.tolist()

    features_for_pca = [f for f in current_features if f not in constant_features_in_selection]

    if not features_for_pca:
        raise ValueError("No valid non-constant features remaining after selection and variance check. "
                         "Please adjust your feature selection.")

    # Initialize and fit MinMaxScaler on the selected, non-constant training features
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[features_for_pca])

    # Transform test features using the scaler fitted on training data
    test_scaled = scaler.transform(test_df[features_for_pca])

    # Initialize and fit PCA on the scaled training features
    pca = PCA(n_components=n_components)
    train_pcs = pca.fit_transform(train_scaled)

    # Transform test features using the PCA fitted on training data
    test_pcs = pca.transform(test_scaled)

    # Create copies of the original DataFrames to add PC columns without modifying originals by reference
    train_df_with_pcs = train_df.copy()
    test_df_with_pcs = test_df.copy()

    # Generate column names for the new PCA components
    pc_columns = [f'PC_{i + 1}' for i in range(n_components)]
    train_df_with_pcs[pc_columns] = train_pcs
    test_df_with_pcs[pc_columns] = test_pcs

    return train_df_with_pcs, test_df_with_pcs, pc_columns  # Return pc_columns for later use


def train_regressor_model(X_train_pcs, y_train_rul):
    """
    Trains a RandomForestRegressor model to map PCA components to RUL.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_pcs, y_train_rul)
    return model


def predict_rul_pipeline(rul_regressor, test_df_with_pcs, rul_test_df, tfm, n_components, context_len, freq,
                         pc_columns):
    """
    Executes the prediction pipeline for RUL on test data.
    """
    predicted_ruls = []
    # Get unique unit numbers from the test data that includes PC columns
    test_unit_numbers = test_df_with_pcs['unit_number'].unique()

    for unit_no in test_unit_numbers:
        unit_df = test_df_with_pcs[test_df_with_pcs['unit_number'] == unit_no].copy()

        # Get the last CONTEXT_LEN PCs for this unit
        if len(unit_df) < context_len:
            # Pad with the first observed PC vector if the unit history is shorter than CONTEXT_LEN
            padding_needed = context_len - len(unit_df)
            first_pc_vector_repeated = np.tile(unit_df[pc_columns].iloc[0].values, (padding_needed, 1))
            last_n_pcs = np.vstack([first_pc_vector_repeated, unit_df[pc_columns].values])[-context_len:]
        else:
            # Take the last CONTEXT_LEN PC vectors
            last_n_pcs = unit_df[pc_columns].iloc[-context_len:].values

        # TimesFM expects input shape (batch_size, num_channels, context_len)
        # Our `last_n_pcs` is (context_len, num_channels).
        # Transpose and add batch dimension: (1, num_channels, context_len)
        timesfm_input = torch.tensor(last_n_pcs, dtype=torch.float32).T.unsqueeze(0)

        # tfm.forecast expects (num_channels, context_len) for single sample input if batch_size=1
        forecast_pcs_torch, _ = tfm.forecast(timesfm_input.squeeze(0), freq=[freq] * n_components)

        # The output `forecast_pcs_torch` will have shape (num_channels, horizon_len)
        # Transpose it to (horizon_len, num_channels) for the regressor
        forecast_pcs_np = forecast_pcs_torch.T  # Ensure conversion to numpy for sklearn regressor

        # For RUL prediction, we are interested in the RUL at the end of the forecasted horizon.
        # This corresponds to the last forecasted PC vector: forecast_pcs_np[-1, :].
        # Reshape it to (1, num_channels) to be compatible with `rul_regressor.predict()`
        most_recent_forecasted_pc_vector = forecast_pcs_np[-1:, :]

        # Predict RUL for this single PC vector
        predicted_rul_value = rul_regressor.predict(most_recent_forecasted_pc_vector)[0]
        predicted_ruls.append(predicted_rul_value)

    # Ensure predictions are non-negative and flatten the array
    y_pred_final = np.maximum(0, np.array(predicted_ruls)).flatten()
    # Get true RUL for comparison from the loaded RUL test file
    y_true_final = rul_test_df['RUL'].values

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
    r2 = r2_score(y_true_final, y_pred_final)
    nasa_score = nasa_rul_score(y_true_final, y_pred_final)

    return y_true_final, y_pred_final, rmse, r2, nasa_score


def create_plotly_figure(y_true, y_pred, rmse, r2, nasa_score, dataset_id):
    """
    Creates an interactive Plotly graph of True vs. Predicted RUL.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, mode='lines', name='True RUL', line=dict(color='blue', width=2)))
    fig.add_trace(
        go.Scatter(y=y_pred, mode='lines', name='Predicted RUL', line=dict(color='red', dash='dash', width=2)))
    fig.update_layout(
        title={
            'text': f'CMAPSS FD00{dataset_id} RUL Prediction<br><sup>RMSE: {rmse:.2f}, R2: {r2:.2f}, NASA Score: {nasa_score:.2f}</sup>',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Engine Unit Index (Test Set)',
        yaxis_title='RUL (Cycles)',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1),
        margin=dict(l=40, r=40, t=80, b=40),  # Adjust margins for better fit
        hovermode="x unified",  # Shows tooltip for all traces at a given x
        height=500,  # Increased height for better visibility
        template="plotly_white"  # Clean white background theme
    )
    return fig


# --- Dash Application Setup ---

app = dash.Dash(__name__)

# Main application layout
app.layout = html.Div(className='min-h-screen bg-gray-100 p-4 font-sans', children=[
    # Header
    html.Div(className='bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-6 rounded-lg shadow-lg mb-6',
             children=[
                 html.H1('CMAPSS RUL Prediction Dashboard', className='text-4xl font-bold text-center'),
                 html.P('Predicting Remaining Useful Life of Turbofan Engines using PCA and TimesFM',
                        className='text-center text-lg mt-2 opacity-90')
             ]),

    # Main Content Area (Controls and Parameters)
    html.Div(className='flex flex-wrap lg:flex-nowrap gap-6 max-w-6xl mx-auto mb-6', children=[
        # Left Panel: Configuration Card
        html.Div(className='w-full lg:w-1/2 bg-white p-6 rounded-lg shadow-md', children=[
            html.H2('Prediction Configuration', className='text-2xl font-semibold text-gray-800 mb-4 border-b pb-2'),

            html.Div(className='grid grid-cols-1 md:grid-cols-2 gap-4 mb-4', children=[
                html.Div(children=[
                    html.Label('Dataset ID:', className='block text-gray-700 text-sm font-bold mb-1'),
                    dcc.Dropdown(
                        id='dataset-id',
                        options=[{'label': f'FD00{i}', 'value': i} for i in range(1, 5)],
                        value=2,  # Default to FD002
                        clearable=False,
                        className='rounded-md shadow-sm'
                    ),
                ]),
                html.Div(children=[
                    html.Label('RUL Threshold (Cycles):', className='block text-gray-700 text-sm font-bold mb-1'),
                    dcc.Input(id='rul-thresh', type='number', value=130, min=0, step=5,
                              className='w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                ]),
                html.Div(children=[
                    html.Label('Context Length (TimesFM):', className='block text-gray-700 text-sm font-bold mb-1'),
                    dcc.Input(id='context-len', type='number', value=64, min=32, step=32,
                              className='w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                ]),
                html.Div(children=[
                    html.Label('Horizon Length (TimesFM):', className='block text-gray-700 text-sm font-bold mb-1'),
                    dcc.Input(id='horizon-len', type='number', value=16, min=1, step=1,
                              className='w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                ]),
                html.Div(children=[
                    html.Label('Num PCA Components:', className='block text-gray-700 text-sm font-bold mb-1'),
                    dcc.Input(id='pca-n', type='number', value=10, min=1, step=1,
                              className='w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                ]),
                html.Div(children=[
                    html.Label('TimesFM Frequency (0, 1, or 2):',
                               className='block text-gray-700 text-sm font-bold mb-1'),
                    dcc.Input(id='freq', type='number', value=1, min=0, max=2, step=1,
                              className='w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                ]),
            ]),
            html.Div(className='mb-4', children=[
                html.Label('CMAPSS Data Path:', className='block text-gray-700 text-sm font-bold mb-1'),
                dcc.Input(id='data-path', type='text', value='/Users/pankajti/dev/data/kaggle/nasa/CMaps', placeholder='e.g., /path/to/CMAPSSData',
                          className='w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                html.P('Ensure CMAPSS data files (e.g., train_FD001.txt, RUL_FD001.txt) are in this directory.',
                       className='text-xs text-gray-500 mt-1')
            ]),
            html.Button('Run Prediction', id='run-btn', n_clicks=0,
                        className='w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-md shadow-md '
                                  'transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-75'),
        ]),

        # Right Panel: Dynamic Feature Selection
        html.Div(className='w-full lg:w-1/2 bg-white p-6 rounded-lg shadow-md', children=[
            html.H2('Select Features for PCA', className='text-2xl font-semibold text-gray-800 mb-4 border-b pb-2'),
            dcc.Checklist(
                id='features-checklist',
                options=[{'label': col, 'value': col} for col in SELECTABLE_FEATURES_FOR_UI],
                value=DEFAULT_FEATURES,  # Initial selected features
                inline=False,  # Stack checkboxes vertically
                className='max-h-64 overflow-y-auto bg-gray-50 p-3 rounded-md border border-gray-200 text-gray-700'
            ),
            html.P('Select which sensor and operational setting features to use for PCA.',
                   className='text-sm text-gray-600 mt-3')
        ]),
    ]),

    # Loading Spinner (Wraps results and plot)
    dcc.Loading(
        id="loading-output",
        type="circle",
        children=[
            # Metrics Output Section
            html.Div(className='bg-white p-6 rounded-lg shadow-md max-w-6xl mx-auto mb-6', children=[
                html.H2('Prediction Results', className='text-2xl font-semibold text-gray-800 mb-4 border-b pb-2'),
                html.Div(id='metrics-output', className='text-lg text-gray-700'),
            ]),

            # Plot Section
            html.Div(className='bg-white p-6 rounded-lg shadow-md max-w-6xl mx-auto mb-6', children=[
                html.H2('True vs. Predicted RUL', className='text-2xl font-semibold text-gray-800 mb-4 border-b pb-2'),
                dcc.Graph(id='rul-plot', style={'width': '100%', 'height': '500px'}),
            ]),
        ]
    )
])


# --- Callbacks ---

# No longer need update_selected_features_display as dcc.Checklist handles it directly.
# The selected features are passed via State in the main run_prediction callback.

@app.callback(
    [Output('metrics-output', 'children'),
     Output('rul-plot', 'figure')],
    Input('run-btn', 'n_clicks'),
    State('dataset-id', 'value'),
    State('rul-thresh', 'value'),
    State('context-len', 'value'),
    State('horizon-len', 'value'),
    State('pca-n', 'value'),
    State('freq', 'value'),
    State('data-path', 'value'),
    State('features-checklist', 'value')  # New State to get selected features
)
def run_prediction(n_clicks, dataset_id, rul_threshold, context_len, horizon_len, pca_n, freq, data_path,
                   selected_features):
    if n_clicks == 0:
        # Return empty plot and default message on initial load
        return html.P('Click "Run Prediction" to see results.'), go.Figure()

    # Input validation for selected_features
    if not selected_features:
        error_message = html.Div(className='bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative',
                                 children=[
                                     html.Strong('Configuration Error: '),
                                     html.Span("Please select at least one feature for PCA.")
                                 ])
        return error_message, go.Figure()

    try:
        # Load and preprocess data
        train_df, test_df, rul_test_df = load_data(dataset_id, data_path, rul_threshold)

        # Apply PCA and preprocess features (including constant feature removal)
        # Pass the dynamically selected_features from the UI
        train_df_with_pcs, test_df_with_pcs, pc_columns = apply_pca_and_preprocess_features(
            train_df, test_df, selected_features, pca_n
        )

        # Train RUL Regressor
        X_reg_train = train_df_with_pcs[pc_columns].values
        y_reg_train = train_df_with_pcs['RUL'].values
        rul_regressor = train_regressor_model(X_reg_train, y_reg_train)

        # Get TimesFM model
        tfm = get_timesfm_model(context_len, horizon_len)

        # Predict RUL
        y_true, y_pred, rmse, r2, nasa_score = predict_rul_pipeline(
            rul_regressor, test_df_with_pcs, rul_test_df, tfm, pca_n, context_len, freq, pc_columns
        )

        # Create metrics display
        metrics_output = html.Div(children=[
            html.P(f"RMSE: {rmse:.2f}", className='text-xl font-medium text-gray-800'),
            html.P(f"R2 Score: {r2:.2f}", className='text-xl font-medium text-gray-800'),
            html.P(f"NASA RUL Score: {nasa_score:.2f}", className='text-xl font-medium text-gray-800'),
            html.P(f"Dataset: FD00{dataset_id}", className='text-md text-gray-600 mt-2'),
            html.P(f"PCA Components: {pca_n}", className='text-md text-gray-600'),
            html.P(f"Features Used: {len(selected_features)} selected ({len(pc_columns)} non-constant PCs)",
                   className='text-md text-gray-600'),
        ])

        # Create Plotly figure
        figure = create_plotly_figure(y_true, y_pred, rmse, r2, nasa_score, dataset_id)

        return metrics_output, figure

    except FileNotFoundError as e:
        error_message = html.Div(className='bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative',
                                 children=[
                                     html.Strong('Error: '),
                                     html.Span(str(e))
                                 ])
        return error_message, go.Figure()  # Return empty figure on error
    except ValueError as e:
        error_message = html.Div(className='bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative',
                                 children=[
                                     html.Strong('Configuration Error: '),
                                     html.Span(str(e))
                                 ])
        return error_message, go.Figure()
    except Exception as e:
        # Catch any other unexpected errors during execution
        error_message = html.Div(className='bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative',
                                 children=[
                                     html.Strong('An unexpected error occurred: '),
                                     html.Span(str(e)),
                                     html.P(
                                         "Please check your input parameters, data path, and ensure all libraries are correctly installed and model weights are downloaded.",
                                         className='text-sm mt-2')
                                 ])
        return error_message, go.Figure()


if __name__ == '__main__':
    # Add a custom CSS file for Tailwind CSS (if you use it locally).
    # If deploying, you might use a CDN or compile Tailwind CSS into a static asset.
    # For a simple local setup, we can use a small inline style or a local CSS file.
    # For full Tailwind, you'd typically need a build step or a CDN.
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    app.run_server(debug=True)

