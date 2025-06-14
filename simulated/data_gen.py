import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set parameters for synthetic vibration signal
sampling_rate = 20000  # Hz
duration = 1.024       # seconds (to match 20480 samples)
n_samples = int(sampling_rate * duration)

# Simulate 3 components of the signal
t = np.linspace(0, duration, n_samples)

# 1. Base healthy vibration signal (sinusoidal)
base_signal = 0.1 * np.sin(2 * np.pi * 50 * t)

# 2. Add defect-like high frequency component
defect_signal = 0.05 * np.sin(2 * np.pi * 1200 * t)

# 3. Add random impulsive noise (like bearing fault impact)
impulses = np.zeros_like(t)
impulse_indices = np.random.choice(n_samples, size=10, replace=False)
impulses[impulse_indices] = np.random.uniform(-1.0, 1.0, size=10)

# Final synthetic vibration signal
synthetic_signal = base_signal + defect_signal + impulses

# Plot a short segment
plt.figure(figsize=(12, 4))
plt.plot(t[:1000], synthetic_signal[:1000], label="Synthetic Vibration")
plt.title("Sample Synthetic Vibration Signal (first 1000 samples)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Create a dataframe for export if needed
vibration_df = pd.DataFrame({'time_sec': t, 'amplitude': synthetic_signal})
vibration_df.head()
