import os
import sys
import time
import json
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Auto-detect Google Colab and mount Google Drive
if 'google.colab' in sys.modules:
    print("[Colab] Google Colab environment detected.")
    from google.colab import drive
    drive.mount('/content/drive')

    candidates = [
        '/content/drive/MyDrive/user/goit/Diplom/var1_Di/code/Di',
        '/content/drive/MyDrive/Diplom/var1_Di/code/Di',
        '/content/drive/MyDrive/var1_Di/code/Di',
        '/content/drive/MyDrive/code/Di'
    ]
    target_dir = None
    for cand in candidates:
        if os.path.isdir(cand):
            target_dir = cand
            break

    if target_dir is None:
        print("[Colab] Searching for project folder in MyDrive...")
        for root, dirs, files in os.walk('/content/drive/MyDrive'):
            if 'cifar10_experiment.py' in files and 'di_algorithm.py' in files:
                target_dir = root
                break

    if target_dir:
        os.chdir(target_dir)
        if target_dir not in sys.path:
            sys.path.insert(0, target_dir)
        print(f"[Colab] Working directory set to: {os.getcwd()}")
else:
    current_dir = os.path.dirname(os.path.abspath('__file__')) if '__file__' in locals() else os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    print(f"[Local] Working directory: {os.getcwd()}")

# Hardware accelerator check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# Reproducibility: fix all random generators (CPU, NumPy, Python, CUDA)
# ============================================================================
import random

SEED = 42

# Python built-in random
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# PyTorch CPU
torch.manual_seed(SEED)

# PyTorch CUDA (all devices)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# CuDNN: deterministic mode, disable benchmark for reproducible results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed fixed: {SEED}")
