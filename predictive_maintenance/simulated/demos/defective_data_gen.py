# Simulate time-degrading vibration signal leading to failure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def simulate_degrading_signal(sampling_rate=20000, duration=10, window_size=1.024):
    """
    Simulates vibration data over time with increasing defect characteristics
    leading to failure at the end.
    """
    total_windows = int(duration / window_size)
    samples_per_window = int(sampling_rate * window_size)
    full_signal = []
    timestamps = []

    for i in range(total_windows):
        t = np.linspace(0, window_size, samples_per_window)

        # Healthy base signal
        base = 0.1 * np.sin(2 * np.pi * 50 * t)

        # Gradually increasing defect (higher frequency)
        defect_strength = 0.01 + (i / total_windows) * 0.1
        defect = defect_strength * np.sin(2 * np.pi * 1200 * t)

        # Gradually increasing impulse noise
        impulses = np.zeros_like(t)
        num_impulses = 2 + int((i / total_windows) * 10)
        impulse_indices = np.random.choice(samples_per_window, size=num_impulses, replace=False)
        impulses[impulse_indices] = np.random.uniform(-1.0, 1.0, size=num_impulses)

        signal = base + defect + impulses
        full_signal.append(signal)
        timestamps.extend([i * window_size + x for x in t])

    full_signal = np.concatenate(full_signal)
    timestamps = np.array(timestamps)
    return timestamps, full_signal

# Simulate degrading signal over 10 seconds (approx 10 snapshots)
timestamps, degrading_signal = simulate_degrading_signal()

# Plot full signal (zoomed out)
plt.figure(figsize=(12, 4))
plt.plot(timestamps, degrading_signal, linewidth=0.5)
plt.title("Simulated Vibration Signal Over Time (Degrading to Failure)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Show sample data as DataFrame
degrading_df = pd.DataFrame({'time_sec': timestamps, 'amplitude': degrading_signal})
degrading_df.head()
