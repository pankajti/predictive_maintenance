from predictive_maintenance.nasa.cmapss.prediction_app.CMAPSS_RUL_Prediction_TimesFM_PCA import run_model
import pandas as pd
from itertools import product

if __name__ == '__main__':

    RUL_THRESHOLD = 125
    DATA_PATH = r'/Users/pankajti/dev/data/kaggle/nasa/CMaps'
    COLUMNS = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    DATASET_ID = 2;
    run_model

    corrected_param_grid = {
        "context_len": [64, 128],
        "horizon_len": [16, 32],
        "freq": [1],
        "n_components": [5, 10],
        "rul_threshold" : [90,100,110,120,130]
    }

    results = []
    for values in product(*corrected_param_grid.values()):
        params = dict(zip(corrected_param_grid.keys(), values))
        result = run_model(**params,
                           data_path=DATA_PATH,
                           dataset_id=DATASET_ID,
                           columns=COLUMNS,
                           )
        result['params'] = params
        print(params)
        results.append(result)

    # Sort by best R² or NASA score
    sorted_by_r2 = sorted(results, key=lambda x: x["r2"], reverse=True)

    df_r2 = pd.DataFrame([{
        **res["params"],
        "R2": res["r2"],
        "RMSE": res["rmse"],
        "NASA_Score": res["nasa_score"]
    } for res in sorted_by_r2])

    sorted_by_r2 = sorted(results, key=lambda x: x["r2"], reverse=True)
    sorted_by_nasa = sorted(results, key=lambda x: x["nasa_score"])

    df_r2 = pd.DataFrame([{
        **res["params"],
        "R2": res["r2"],
        "RMSE": res["rmse"],
        "NASA_Score": res["nasa_score"]
    } for res in sorted_by_r2])

    print("Grid Search Results (Sorted by R2)",df_r2)

    best_config = sorted_by_r2[0]["params"]
    best_score = sorted_by_r2[0]["r2"], sorted_by_r2[0]["nasa_score"]
    print(best_config, best_score)

#{'context_len': 128, 'horizon_len': 16, 'freq': 0, 'n_components': 10} (0.7190976440443193, 2976.7230789251676) for #3
