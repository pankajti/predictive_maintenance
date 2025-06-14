import torch

def get_device():
    # --- Device Setup (for Mac M3) ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) for acceleration.")
    elif torch.cuda.is_available():  # Fallback for other systems
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU) for acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")
    return device
