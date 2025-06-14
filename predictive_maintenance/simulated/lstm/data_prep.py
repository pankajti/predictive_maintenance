import torch
from torch.utils.data import Dataset, DataLoader
from simulated.lstm.data_gen import get_simulated_data

# --- 2. PyTorch Data Preparation (Dataset with timestamp) ---

WINDOW_SIZE_SAMPLES = 2048
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

def get_data_loaders ():
    train_sim_ids = list(range(10))
    val_sim_ids = list(range(10, 13))
    test_sim_ids = list(range(13, 15))
    list_of_degrading_dfs = get_simulated_data(15)

    train_datasets = []
    for sim_id in train_sim_ids:
        df_sim = list_of_degrading_dfs[sim_id]
        train_datasets.append(VibrationDataset(df_sim, window_size=WINDOW_SIZE_SAMPLES, step_size=STEP_SIZE_SAMPLES))
    train_dataset_combined = torch.utils.data.ConcatDataset(train_datasets)

    val_datasets = []
    for sim_id in val_sim_ids:
        df_sim = list_of_degrading_dfs[sim_id]
        val_datasets.append(VibrationDataset(df_sim, window_size=WINDOW_SIZE_SAMPLES, step_size=STEP_SIZE_SAMPLES))
    val_dataset_combined = torch.utils.data.ConcatDataset(val_datasets)
    val_dataloader = DataLoader(val_dataset_combined, batch_size=64, shuffle=False)
    train_dataloader = DataLoader(train_dataset_combined, batch_size=64, shuffle=True)

    test_datasets = []
    for sim_id in test_sim_ids:
        df_sim = list_of_degrading_dfs[sim_id]
        test_datasets.append(VibrationDataset(df_sim, window_size=WINDOW_SIZE_SAMPLES, step_size=STEP_SIZE_SAMPLES))

    print(f"\nTraining on {len(train_sim_ids)} simulations. Total training windows: {len(train_dataset_combined)}")
    print(f"Validation on {len(val_sim_ids)} simulations. Total validation windows: {len(val_dataset_combined)}")
    print(f"Testing on {len(test_sim_ids)} simulations.")

    return train_dataloader, val_dataloader,test_datasets,test_sim_ids


def main():
    get_data_loaders()



if __name__ == '__main__':
    main()


