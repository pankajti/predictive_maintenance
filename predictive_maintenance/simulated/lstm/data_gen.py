import numpy as np
import pandas as pd

# --- 1. Simulate Data (Modified for Varied Simulations) ---
def simulate_degrading_signal_varied(sampling_rate=20000, duration=10, window_size_sim=1):
    """
    Simulates vibration data over time with increasing defect characteristics
    leading to failure at the end, with added randomness in parameters.
    """
    total_windows = int(duration / window_size_sim)
    samples_per_window_sim = int(sampling_rate * window_size_sim)
    full_signal = []
    timestamps = []

    # Randomize degradation parameters for each simulation run
    base_defect_amplitude = np.random.uniform(0.005, 0.015)
    final_defect_amplitude = np.random.uniform(0.08, 0.15)
    defect_frequency = np.random.uniform(1150, 1250)

    initial_impulses = np.random.randint(1, 4)
    max_impulses_increase = np.random.randint(8, 15)

    for i in range(total_windows):
        t = np.linspace(0, window_size_sim, samples_per_window_sim)

        base = 0.1 * np.sin(2 * np.pi * 50 * t)
        defect_strength = base_defect_amplitude + (i / total_windows) * (final_defect_amplitude - base_defect_amplitude)
        defect = defect_strength * np.sin(2 * np.pi * defect_frequency * t)
        impulses = np.zeros_like(t)
        num_impulses = initial_impulses + int((i / total_windows) * max_impulses_increase)
        impulse_indices = np.random.choice(samples_per_window_sim, size=num_impulses, replace=False)
        impulses[impulse_indices] = np.random.uniform(-1.0, 1.0, size=num_impulses)

        signal = base + defect + impulses
        full_signal.append(signal)
        timestamps.extend([i * window_size_sim + x for x in t])

    full_signal = np.concatenate(full_signal)
    timestamps = np.array(timestamps)
    return timestamps, full_signal, duration

def main():
    num_simulations = 15  # Keep same number of simulations
    list_of_degrading_dfs = get_simulated_data(num_simulations)

    print(f"\nGenerated {len(list_of_degrading_dfs)} individual DataFrames.")


def get_simulated_data(num_simulations):
    list_of_degrading_dfs = []
    print(f"Generating {num_simulations} simulated data instances...")
    for i in range(num_simulations):
        timestamps_i, degrading_signal_i, total_duration_i = simulate_degrading_signal_varied(sampling_rate=5000,
                                                                                              duration=10 + i * 0.5)
        rul_i = total_duration_i - timestamps_i

        df_i = pd.DataFrame({
            'simulation_id': i,
            'time_sec': timestamps_i,
            'amplitude': degrading_signal_i,
            'RUL_sec': rul_i
        })
        list_of_degrading_dfs.append(df_i)
    return list_of_degrading_dfs


if __name__ == '__main__':
    main()

