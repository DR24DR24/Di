import os
import json
from datetime import datetime

TASK_DESCRIPTION_ENGLISH = """# Pretrained Transfer Learning with Source Pretraining, Target Fine-Tuning, and Layer-wise PC Tracking

## Task Description
Implementation of a two-stage transfer learning pipeline based on Di et al. (2025):
1. **Target Fixed Subsets Setup**:
   - At initialization, draw and fix `SAMPLE_SIZES` instances from the target dataset `NUM_REPLICATIONS` times with fixed random seeds for reproducible target validation loss and layer-wise Projection Correlation (PC) evaluation with confidence intervals.
2. **Stage 1 — Source Pretraining with Early Stopping & Control**:
   - Pretrain a 6-layer CNN on the source dataset with early stopping monitored on source validation loss.
   - Upon triggering early stopping, continue training for an additional control period defined by the named constant `POST_EARLY_STOP_EPOCHS = 10`.
   - Log each epoch: Train Loss, Source Validation Loss, Source Test Accuracy, Target Validation Loss, and PC for all layers across replications.
   - Plot loss/accuracy curves and PC layer trajectories.
   - Save the pretrained model checkpoint and metrics to `pretrained/` with timestamps.
3. **Stage 2 — Target Fine-Tuning with Early Stopping**:
   - Fine-tune the model on the fixed `SAMPLE_SIZES` target subset using early stopping on target validation loss.
   - Validate on the target validation set.
   - Log each epoch: Train Loss, Target Validation Loss, Test Accuracy, and PC for all layers.
   - Plot training curves and PC trajectories.
   - Save the fine-tuned model checkpoint and metrics to `pretrained/` with timestamps.
4. **Artifacts & Logging**:
   - All experiment outputs (checkpoints, metrics JSON, visualization plots) are stored in a dedicated folder matching the notebook name (`pretrained/`), with an execution timestamp appended to every artifact filename.
"""

def make_notebook(cells, metadata=None):
    if metadata is None:
        metadata = {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        }
    nb_cells = []
    for c in cells:
        raw_source = c["source"]
        if isinstance(raw_source, str):
            lines = raw_source.splitlines(keepends=True)
        else:
            lines = raw_source

        cell_dict = {
            "cell_type": c["cell_type"],
            "metadata": c.get("metadata", {}),
            "source": lines
        }
        if c["cell_type"] == "code":
            cell_dict["execution_count"] = None
            cell_dict["outputs"] = []
        nb_cells.append(cell_dict)

    return {
        "cells": nb_cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 4
    }

def generate_pretrained_notebook():
    cells = [
        # Cell 0: Task Description
        {
            "cell_type": "markdown",
            "source": TASK_DESCRIPTION_ENGLISH.strip() + "\n"
        },
        # Cell 1: Environment Setup Markdown
        {
            "cell_type": "markdown",
            "source": "## 1. Environment Setup, Google Colab Integration & GPU Verification"
        },
        # Cell 2: Environment Setup Code
        {
            "cell_type": "code",
            "source": """import os
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
"""
        },
        # Cell 3: Parameters Markdown
        {
            "cell_type": "markdown",
            "source": "## 2. Experiment Parameters, Named Constants & Output Directory"
        },
        # Cell 4: Parameters Code
        {
            "cell_type": "code",
            "source": """# Experiment Mode and Subsets
EXPERIMENT_CASE = 'imbalance'   # 'imbalance' (5 animal -> 5 vehicle classes) or 'balance' (50%/50% split)
SAMPLE_SIZES = [500, 1000]      # Target sample sizes for PC tracking and fine-tuning
NUM_REPLICATIONS = 2            # Number of replications for PC confidence interval estimation
USE_REAL_CIFAR10 = True         # Set True for official CIFAR-10; False for quick synthetic mode

# Stage 1: Pretraining Hyperparameters
PRETRAIN_EPOCHS = 50            # Maximum pretraining epochs
EARLY_STOP_PATIENCE = 5         # Patience for early stopping on source validation loss
POST_EARLY_STOP_EPOCHS = 10     # Named constant: extra control epochs after early stopping trigger
LEARNING_RATE_SOURCE = 0.001    # RMSProp learning rate for pretraining (Di et al. 2025 Sec 4.1)

# Stage 2: Target Fine-Tuning Hyperparameters
TARGET_EPOCHS = 50              # Maximum target adaptation epochs
TARGET_PATIENCE = 5             # Patience for target early stopping
LEARNING_RATE_TARGET = 0.001    # RMSProp learning rate for target fine-tuning

# Directory for experiment results (matches notebook name: 'pretrained')
NOTEBOOK_NAME = "pretrained"
RESULTS_DIR = os.path.join(os.getcwd(), NOTEBOOK_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_timestamp() -> str:
    \"\"\"Return a compact timestamp string YYYYMMDD_HHMMSS.\"\"\"
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def make_result_path(base_name: str, ext: str, ts: Optional[str] = None) -> str:
    \"\"\"Generate a full output file path with timestamp in the results directory.\"\"\"
    if ts is None:
        ts = get_timestamp()
    filename = f"{base_name}_{ts}.{ext}"
    return os.path.join(RESULTS_DIR, filename)

print(f"Results Directory: {RESULTS_DIR}")
print(f"Mode: {EXPERIMENT_CASE} | Sample sizes: {SAMPLE_SIZES} | Replications: {NUM_REPLICATIONS}")
print(f"Post early-stopping control epochs: {POST_EARLY_STOP_EPOCHS}")
"""
        },
        # Cell 5: Data Loading Markdown
        {
            "cell_type": "markdown",
            "source": "## 3. Data Loading, Class Splitting, and DataLoaders"
        },
        # Cell 6: Data Loading Code
        {
            "cell_type": "code",
            "source": """from models import CIFAR6CNN, create_transferred_model, set_trainable_layers
from dataset_splits import (
    split_cifar_imbalance,
    split_cifar_balance,
    generate_synthetic_transfer_dataset,
    make_dataloaders
)
from di_algorithm import default_eval_loss
from projection_correlation import projection_correlation
from layer_extractor import extract_layer_activations

if USE_REAL_CIFAR10:
    from cifar10_experiment import load_cifar10_raw
    print("Loading CIFAR-10 dataset...")
    raw_x, raw_y = load_cifar10_raw(data_dir='./data')
    if EXPERIMENT_CASE == 'imbalance':
        splits = split_cifar_imbalance(raw_x, raw_y)
    else:
        splits = split_cifar_balance(raw_x, raw_y)
else:
    print("Generating synthetic transfer dataset...")
    splits = generate_synthetic_transfer_dataset(
        n_source=1200, n_target=800,
        num_classes=5 if EXPERIMENT_CASE == 'imbalance' else 10,
        transition_layer=3 if EXPERIMENT_CASE == 'imbalance' else 4,
        seed=42
    )

loaders = make_dataloaders(splits, batch_size=64)
target_train_x, target_train_y = splits['target_train']
target_test_loader = loaders['target_test']

print(f"Source train size: {len(splits['source_train'][0])} | test: {len(splits['source_test'][0])}")
print(f"Target train size: {len(target_train_x)} | test: {len(splits['target_test'][0])}")
print(f"Source classes: {splits['num_source_classes']} | Target classes: {splits['num_target_classes']}")
"""
        },
        # Cell 7: Fixed Subsets Markdown
        {
            "cell_type": "markdown",
            "source": """## 4. Selecting and Fixing Target Subsets for Projection Correlation Tracking
To calculate Projection Correlation profiles $\\omega_j$ and target validation metrics with statistical confidence intervals without random sampling variance across epochs, we select and fix `SAMPLE_SIZES` items from the target dataset `NUM_REPLICATIONS` times at the beginning.
"""
        },
        # Cell 8: Fixed Subsets Code
        {
            "cell_type": "code",
            "source": """def select_fixed_subsets(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    sample_sizes: List[int],
    num_replications: int,
    base_seed: int = 42
) -> Dict[int, List[Dict[str, Any]]]:
    \"\"\"Pre-sample and fix deterministic subsets for PC and target validation.\"\"\"
    n_total = x_data.shape[0]
    subsets_by_size = {}

    for n_s in sample_sizes:
        subsets_by_size[n_s] = []
        eff_n = min(n_s, n_total)
        for rep in range(num_replications):
            seed = base_seed + rep * 1000 + n_s
            gen = torch.Generator().manual_seed(seed)
            indices = torch.randperm(n_total, generator=gen)[:eff_n]
            sub_x, sub_y = x_data[indices], y_data[indices]
            sub_loader = DataLoader(
                TensorDataset(sub_x, sub_y),
                batch_size=64,
                shuffle=False
            )
            subsets_by_size[n_s].append({
                'rep': rep,
                'seed': seed,
                'indices': indices,
                'x': sub_x,
                'y': sub_y,
                'loader': sub_loader
            })
    return subsets_by_size

# Pre-select and fix target subsets
fixed_target_subsets = select_fixed_subsets(
    target_train_x,
    target_train_y,
    sample_sizes=SAMPLE_SIZES,
    num_replications=NUM_REPLICATIONS,
    base_seed=42
)

print(f"Fixed {len(fixed_target_subsets)} sample size configurations with {NUM_REPLICATIONS} replications each.")

def evaluate_layer_pc_profiles(
    model: nn.Module,
    subsets_dict: Dict[int, List[Dict[str, Any]]],
    device: torch.device,
    pooling: str = 'gap'
) -> Dict[int, Dict[str, Dict[str, float]]]:
    \"\"\"Compute layer-wise PC on the pre-fixed subsets across replications.\"\"\"
    model.eval()
    layer_names = CIFAR6CNN.LAYER_NAMES
    pc_summary_by_size = {}

    for n_s, rep_list in subsets_dict.items():
        layer_rep_values = {l: [] for l in layer_names}
        for item in rep_list:
            acts, labels = extract_layer_activations(
                model=model,
                dataloader=item['loader'],
                layer_names=layer_names,
                pooling=pooling,
                device=device,
                max_samples=n_s
            )
            for l in layer_names:
                pc_val = projection_correlation(acts[l], labels)
                layer_rep_values[l].append(float(pc_val))

        stats = {}
        for l in layer_names:
            arr = np.array(layer_rep_values[l])
            stats[l] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'values': [float(v) for v in arr]
            }
        pc_summary_by_size[n_s] = stats
    return pc_summary_by_size

def format_pc_summary(pc_stats: Dict[int, Dict[str, Dict[str, float]]], target_size: int) -> str:
    \"\"\"Format PC stats as a readable string for logging.\"\"\"
    stats = pc_stats.get(target_size, {})
    parts = []
    for l in CIFAR6CNN.LAYER_NAMES:
        if l in stats:
            parts.append(f\"{l}={stats[l]['mean']:.3f}±{stats[l]['std']:.3f}\")
    return \" | \".join(parts)
"""
        },
        # Cell 9: Stage 1 Markdown
        {
            "cell_type": "markdown",
            "source": """## 5. Stage 1: Source Pretraining with Early Stopping and Post-Patience Control
- Monitored metric: **Source Validation Loss**.
- Upon triggering early stopping, training continues for **`POST_EARLY_STOP_EPOCHS` (=10)** control epochs.
- Logged each epoch:
  - `Epoch ep:02d/PRETRAIN_EPOCHS:02d`
  - `Train Loss: avg_loss:.4f`
  - `Source Test Acc: v_acc * 100:.1f%`
  - `Source Val Loss: s_val_loss:.4f`
  - `Target Val Loss: t_val_loss:.4f`
  - `PC (conv1..conv6)` with mean ± std
"""
        },
        # Cell 10: Stage 1 Code
        {
            "cell_type": "code",
            "source": """print(\"=\" * 75)
print(\"[Stage 1] Pretraining 6-layer CNN on Source Dataset\")
print(f\"Early stopping patience: {EARLY_STOP_PATIENCE} | Post-early-stop control: {POST_EARLY_STOP_EPOCHS} epochs\")
print(\"=\" * 75)

source_model = CIFAR6CNN(num_classes=splits['num_source_classes']).to(device)
optimizer = torch.optim.RMSprop(source_model.parameters(), lr=LEARNING_RATE_SOURCE, alpha=0.9)
criterion = nn.CrossEntropyLoss()

stage1_history = []
best_s_val_loss = float('inf')
best_source_state = None
early_stop_triggered = False
early_stop_epoch = None
extra_epochs_left = POST_EARLY_STOP_EPOCHS
patience_counter = 0

primary_sample_size = SAMPLE_SIZES[0]

for ep in range(1, PRETRAIN_EPOCHS + POST_EARLY_STOP_EPOCHS + 1):
    # Training pass
    source_model.train()
    tr_loss = 0.0
    for x, y in loaders['source_train']:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = source_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * x.size(0)
    avg_train_loss = tr_loss / len(loaders['source_train'].dataset)

    # Source validation & test accuracy
    s_val_loss, s_test_acc = default_eval_loss(source_model, loaders['source_test'], criterion, device)

    # Target validation loss on target test set
    t_val_loss, _ = default_eval_loss(source_model, target_test_loader, criterion, device)

    # Layer-wise Projection Correlation on fixed subsets
    pc_stats = evaluate_layer_pc_profiles(source_model, fixed_target_subsets, device=device)
    pc_str = format_pc_summary(pc_stats, target_size=primary_sample_size)

    # Logging line
    prefix_tag = \"[Control]\" if early_stop_triggered else \"[Pretrain]\"
    print(f\"{prefix_tag} Epoch {ep:02d}/{PRETRAIN_EPOCHS:02d} | \"
          f\"Train Loss: {avg_train_loss:.4f} | \"
          f\"Source Test Acc: {s_test_acc * 100:.1f}% | \"
          f\"Source Val Loss: {s_val_loss:.4f} | \"
          f\"Target Val Loss: {t_val_loss:.4f} | \"
          f\"PC: {pc_str}\")

    stage1_history.append({
        'epoch': ep,
        'train_loss': float(avg_train_loss),
        'source_val_loss': float(s_val_loss),
        'source_test_acc': float(s_test_acc * 100.0),
        'target_val_loss': float(t_val_loss),
        'pc_profiles': pc_stats,
        'is_control_phase': early_stop_triggered
    })

    # Early stopping check on Source Val Loss
    if not early_stop_triggered:
        if s_val_loss < best_s_val_loss:
            best_s_val_loss = s_val_loss
            best_source_state = {k: v.cpu().clone() for k, v in source_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                early_stop_triggered = True
                early_stop_epoch = ep
                print(f\"\\n>>> [Early Stop Triggered] at epoch {ep} (Patience={EARLY_STOP_PATIENCE}).\")
                print(f\">>> Running {POST_EARLY_STOP_EPOCHS} additional control epochs...\\n\")
    else:
        extra_epochs_left -= 1
        if extra_epochs_left <= 0:
            print(f\"\\n>>> Completed {POST_EARLY_STOP_EPOCHS} post-early-stop control epochs. Stopping Stage 1.\\n\")
            break

# Save Stage 1 artifacts with timestamp
stage1_ts = get_timestamp()

model_save_path = make_result_path("pretrained_model", "pt", stage1_ts)
torch.save({
    'best_state_dict': best_source_state,
    'final_state_dict': source_model.state_dict(),
    'early_stop_epoch': early_stop_epoch,
    'best_s_val_loss': best_s_val_loss
}, model_save_path)
print(f\"Pretrained model checkpoint saved to: {model_save_path}\")

history_save_path = make_result_path("stage1_history", "json", stage1_ts)
with open(history_save_path, 'w', encoding='utf-8') as f:
    json.dump({
        'stage': 'stage1_pretraining',
        'experiment_case': EXPERIMENT_CASE,
        'early_stop_epoch': early_stop_epoch,
        'post_early_stop_epochs': POST_EARLY_STOP_EPOCHS,
        'best_s_val_loss': best_s_val_loss,
        'history': stage1_history
    }, f, indent=2)
print(f\"Stage 1 metrics history saved to: {history_save_path}\")
"""
        },
        # Cell 11: Stage 1 Plots Markdown
        {
            "cell_type": "markdown",
            "source": "## 6. Stage 1 Visualizations: Training Dynamics and Layer-wise PC Evolution"
        },
        # Cell 12: Stage 1 Plots Code
        {
            "cell_type": "code",
            "source": """epochs_arr = [h['epoch'] for h in stage1_history]
tr_loss_arr = [h['train_loss'] for h in stage1_history]
s_val_loss_arr = [h['source_val_loss'] for h in stage1_history]
t_val_loss_arr = [h['target_val_loss'] for h in stage1_history]
s_acc_arr = [h['source_test_acc'] for h in stage1_history]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Losses
axes[0].plot(epochs_arr, tr_loss_arr, 'o-', color='#1f77b4', label='Source Train Loss', linewidth=2, markersize=4)
axes[0].plot(epochs_arr, s_val_loss_arr, 's-', color='#d62728', label='Source Val Loss', linewidth=2, markersize=4)
axes[0].plot(epochs_arr, t_val_loss_arr, '^--', color='#9467bd', label='Target Val Loss', linewidth=2, markersize=4)
if early_stop_epoch:
    axes[0].axvline(early_stop_epoch, color='black', linestyle=':', linewidth=2, label=f'Early Stop (ep {early_stop_epoch})')
    axes[0].axvspan(early_stop_epoch, epochs_arr[-1], alpha=0.12, color='orange', label=f'+{POST_EARLY_STOP_EPOCHS} Control Epochs')
axes[0].set_xlabel('Epoch', fontweight='bold')
axes[0].set_ylabel('Loss', fontweight='bold')
axes[0].set_title('Stage 1: Training & Validation Losses', fontweight='bold')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# Plot 2: Source Accuracy
axes[1].plot(epochs_arr, s_acc_arr, 'o-', color='#2ca02c', label='Source Test Accuracy', linewidth=2, markersize=4)
if early_stop_epoch:
    axes[1].axvline(early_stop_epoch, color='black', linestyle=':', linewidth=2, label=f'Early Stop (ep {early_stop_epoch})')
    axes[1].axvspan(early_stop_epoch, epochs_arr[-1], alpha=0.12, color='orange')
axes[1].set_xlabel('Epoch', fontweight='bold')
axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
axes[1].set_title('Stage 1: Source Classification Accuracy', fontweight='bold')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.tight_layout()
curves_path = make_result_path("stage1_curves", "png", stage1_ts)
plt.savefig(curves_path, dpi=150)
plt.show()
print(f\"Stage 1 training curves saved to: {curves_path}\")

# Plot 3: Projection Correlation Evolution per Layer
plt.figure(figsize=(12, 6))
layer_names = CIFAR6CNN.LAYER_NAMES
cmap = plt.cm.get_cmap('tab10', len(layer_names))

for l_idx, layer in enumerate(layer_names):
    means = [h['pc_profiles'][primary_sample_size][layer]['mean'] for h in stage1_history]
    stds = [h['pc_profiles'][primary_sample_size][layer]['std'] for h in stage1_history]
    plt.errorbar(
        epochs_arr, means, yerr=stds,
        fmt='o-', color=cmap(l_idx),
        linewidth=2, markersize=4, capsize=3,
        label=f'{layer}'
    )

if early_stop_epoch:
    plt.axvline(early_stop_epoch, color='black', linestyle=':', linewidth=2, label=f'Early Stop (ep {early_stop_epoch})')
    plt.axvspan(early_stop_epoch, epochs_arr[-1], alpha=0.1, color='orange')

plt.xlabel('Pretraining Epoch', fontweight='bold', fontsize=11)
plt.ylabel(f'Projection Correlation (n={primary_sample_size})', fontweight='bold', fontsize=11)
plt.title(f'Stage 1: Layer-wise PC Evolution over Pretraining Epochs ({EXPERIMENT_CASE.upper()} Case)', fontweight='bold', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

pc_evolution_path = make_result_path("stage1_pc_evolution", "png", stage1_ts)
plt.savefig(pc_evolution_path, dpi=150)
plt.show()
print(f\"Stage 1 PC evolution plot saved to: {pc_evolution_path}\")
"""
        },
        # Cell 13: Stage 2 Markdown
        {
            "cell_type": "markdown",
            "source": """## 7. Stage 2: Target Fine-Tuning on Fixed SAMPLE_SIZES with Early Stopping
- Fine-tune the model using the fixed `SAMPLE_SIZES` target training subset.
- Evaluate on the full target validation/test set (`target_test_loader`).
- Monitored metric: **Target Validation Loss** (with early stopping).
- Logged each epoch:
  - `Epoch ep:02d/EPOCHS:02d`
  - `Train Loss: avg_loss:.4f`
  - `Test Acc: v_acc * 100:.1f%`
  - `Val Loss: val_loss:.4f`
  - `PC (conv1..conv6)` for all layers
"""
        },
        # Cell 14: Stage 2 Code
        {
            "cell_type": "code",
            "source": """print(\"=\" * 75)
print(f\"[Stage 2] Fine-tuning Model on Fixed Target Subset (n={primary_sample_size})\")
print(f\"Target early stopping patience: {TARGET_PATIENCE} | Target epochs: {TARGET_EPOCHS}\")
print(\"=\" * 75)

# Load best checkpoint from Stage 1 into target model
if best_source_state is not None:
    source_model.load_state_dict(best_source_state)

target_model = create_transferred_model(
    pretrained_model=source_model,
    k=len(CIFAR6CNN.LAYER_NAMES),
    num_target_classes=splits['num_target_classes'],
    freeze_transferred=False
).to(device)
set_trainable_layers(target_model, k_freeze=0)

target_optimizer = torch.optim.RMSprop(target_model.parameters(), lr=LEARNING_RATE_TARGET, alpha=0.9)
target_criterion = nn.CrossEntropyLoss()

# Fixed target training loader (using the first replication of primary_sample_size)
target_train_subset_loader = fixed_target_subsets[primary_sample_size][0]['loader']

stage2_history = []
best_t_val_loss = float('inf')
best_target_state = None
target_patience_counter = 0
target_early_stop_epoch = None

for ep in range(1, TARGET_EPOCHS + 1):
    target_model.train()
    tr_loss = 0.0
    for x, y in target_train_subset_loader:
        x, y = x.to(device), y.to(device)
        target_optimizer.zero_grad()
        out = target_model(x)
        loss = target_criterion(out, y)
        loss.backward()
        target_optimizer.step()
        tr_loss += loss.item() * x.size(0)
    avg_train_loss = tr_loss / len(target_train_subset_loader.dataset)

    # Validate on target validation/test set
    val_loss, test_acc = default_eval_loss(target_model, target_test_loader, target_criterion, device)

    # Compute PC on fixed subsets during fine-tuning
    pc_stats = evaluate_layer_pc_profiles(target_model, fixed_target_subsets, device=device)
    pc_str = format_pc_summary(pc_stats, target_size=primary_sample_size)

    # Logging line
    print(f\"Epoch {ep:02d}/{TARGET_EPOCHS:02d} | \"
          f\"Train Loss: {avg_train_loss:.4f} | \"
          f\"Test Acc: {test_acc * 100:.1f}% | \"
          f\"Val Loss: {val_loss:.4f} | \"
          f\"PC: {pc_str}\")

    stage2_history.append({
        'epoch': ep,
        'train_loss': float(avg_train_loss),
        'val_loss': float(val_loss),
        'test_acc': float(test_acc * 100.0),
        'pc_profiles': pc_stats
    })

    # Early stopping check on Target Validation Loss
    if val_loss < best_t_val_loss:
        best_t_val_loss = val_loss
        best_target_state = {k: v.cpu().clone() for k, v in target_model.state_dict().items()}
        target_patience_counter = 0
    else:
        target_patience_counter += 1
        if target_patience_counter >= TARGET_PATIENCE:
            target_early_stop_epoch = ep
            print(f\"\\n>>> [Target Early Stop] Triggered at epoch {ep} (Patience={TARGET_PATIENCE}). Stopping Stage 2.\\n\")
            break

# Save Stage 2 artifacts with timestamp
stage2_ts = get_timestamp()

ft_model_path = make_result_path("finetuned_model", "pt", stage2_ts)
torch.save({
    'best_state_dict': best_target_state,
    'final_state_dict': target_model.state_dict(),
    'best_val_loss': best_t_val_loss,
    'early_stop_epoch': target_early_stop_epoch
}, ft_model_path)
print(f\"Fine-tuned target model checkpoint saved to: {ft_model_path}\")

stage2_history_path = make_result_path("stage2_history", "json", stage2_ts)
with open(stage2_history_path, 'w', encoding='utf-8') as f:
    json.dump({
        'stage': 'stage2_fine_tuning',
        'experiment_case': EXPERIMENT_CASE,
        'target_sample_size': primary_sample_size,
        'best_val_loss': best_t_val_loss,
        'early_stop_epoch': target_early_stop_epoch,
        'history': stage2_history
    }, f, indent=2)
print(f\"Stage 2 metrics history saved to: {stage2_history_path}\")
"""
        },
        # Cell 15: Stage 2 Plots Markdown
        {
            "cell_type": "markdown",
            "source": "## 8. Stage 2 Visualizations: Fine-Tuning Curves & PC Trajectories"
        },
        # Cell 16: Stage 2 Plots Code
        {
            "cell_type": "code",
            "source": """epochs_ft = [h['epoch'] for h in stage2_history]
tr_loss_ft = [h['train_loss'] for h in stage2_history]
val_loss_ft = [h['val_loss'] for h in stage2_history]
test_acc_ft = [h['test_acc'] for h in stage2_history]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Fine-tuning Loss & Accuracy
axes[0].plot(epochs_ft, tr_loss_ft, 'o-', color='#1f77b4', label='Train Loss', linewidth=2, markersize=4)
axes[0].plot(epochs_ft, val_loss_ft, 's-', color='#d62728', label='Val Loss', linewidth=2, markersize=4)
if target_early_stop_epoch:
    axes[0].axvline(target_early_stop_epoch, color='black', linestyle=':', linewidth=2, label=f'Early Stop (ep {target_early_stop_epoch})')
axes[0].set_xlabel('Epoch', fontweight='bold')
axes[0].set_ylabel('Loss', fontweight='bold')
axes[0].set_title('Stage 2: Target Loss Curves', fontweight='bold')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

axes[1].plot(epochs_ft, test_acc_ft, '^-', color='#2ca02c', label='Target Test Accuracy', linewidth=2, markersize=4)
if target_early_stop_epoch:
    axes[1].axvline(target_early_stop_epoch, color='black', linestyle=':', linewidth=2, label=f'Early Stop (ep {target_early_stop_epoch})')
axes[1].set_xlabel('Epoch', fontweight='bold')
axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
axes[1].set_title('Stage 2: Target Classification Accuracy', fontweight='bold')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.tight_layout()
ft_curves_path = make_result_path("stage2_curves", "png", stage2_ts)
plt.savefig(ft_curves_path, dpi=150)
plt.show()
print(f\"Stage 2 curves saved to: {ft_curves_path}\")

# Plot 2: PC evolution during fine-tuning
plt.figure(figsize=(12, 6))
cmap = plt.cm.get_cmap('tab10', len(layer_names))

for l_idx, layer in enumerate(layer_names):
    means = [h['pc_profiles'][primary_sample_size][layer]['mean'] for h in stage2_history]
    stds = [h['pc_profiles'][primary_sample_size][layer]['std'] for h in stage2_history]
    plt.errorbar(
        epochs_ft, means, yerr=stds,
        fmt='s--', color=cmap(l_idx),
        linewidth=2, markersize=4, capsize=3,
        label=f'{layer}'
    )

if target_early_stop_epoch:
    plt.axvline(target_early_stop_epoch, color='black', linestyle=':', linewidth=2, label=f'Early Stop (ep {target_early_stop_epoch})')

plt.xlabel('Fine-Tuning Epoch', fontweight='bold', fontsize=11)
plt.ylabel(f'Projection Correlation (n={primary_sample_size})', fontweight='bold', fontsize=11)
plt.title(f'Stage 2: Layer-wise PC Evolution during Fine-Tuning ({EXPERIMENT_CASE.upper()} Case)', fontweight='bold', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

ft_pc_path = make_result_path("stage2_pc_evolution", "png", stage2_ts)
plt.savefig(ft_pc_path, dpi=150)
plt.show()
print(f\"Stage 2 PC evolution saved to: {ft_pc_path}\")
"""
        },
        # Cell 17: Summary Markdown
        {
            "cell_type": "markdown",
            "source": "## 9. Experiment Summary and Saved Artifacts Inventory"
        },
        # Cell 18: Summary Code
        {
            "cell_type": "code",
            "source": """print(\"=\" * 75)
print(\"EXPERIMENT COMPLETED SUCCESSFULLY\")
print(f\"Results stored in catalog matching notebook name: '{RESULTS_DIR}'\")
print(\"=\" * 75)

artifacts = sorted(glob.glob(os.path.join(RESULTS_DIR, "*")))
print(f\"Found {len(artifacts)} saved artifacts in '{RESULTS_DIR}':\")
for art in artifacts:
    size_kb = os.path.getsize(art) / 1024.0
    print(f\"  - {os.path.basename(art)} ({size_kb:.1f} KB)\")
print(\"=\" * 75)
"""
        }
    ]

    nb = make_notebook(cells)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Generated notebook successfully: {out_path}")

if __name__ == "__main__":
    generate_pretrained_notebook()
