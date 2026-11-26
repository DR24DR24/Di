import json

with open('d:/googledisk8J/git/Di/pretrained_stat.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def set_src(idx, text):
    nb['cells'][idx]['source'] = text.splitlines(keepends=True)

def save():
    with open('d:/googledisk8J/git/Di/pretrained_stat.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

# Cell 3: setup_environment
set_src(3, """import os
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


def setup_environment(seed: int = 42) -> Tuple[torch.device, int]:
    """Setup device, random seeds, Colab mount. Returns (device, seed)."""
    import random
    if "google.colab" in sys.modules:
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
            print("[Colab] Searching for project folder...")
            for root, dirs, files in os.walk('/content/drive/MyDrive'):
                if 'cifar10_experiment.py' in files and 'di_algorithm.py' in files:
                    target_dir = root
                    break
        if target_dir:
            os.chdir(target_dir)
            if target_dir not in sys.path:
                sys.path.insert(0, target_dir)
            print(f"[Colab] Working directory: {os.getcwd()}")
        else:
            cd = os.path.dirname(os.path.abspath("__file__")) if "__file__" in locals() else os.getcwd()
            if cd not in sys.path:
                sys.path.insert(0, cd)
            print(f"[Local] Working directory: {os.getcwd()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed fixed: {seed}")
    return device, seed
""")
save()
print("Cell 3 done")

# Cell 5: get_experiment_params()
set_src(5, """def get_experiment_params() -> Dict[str, Any]:
    """Return all experiment parameters as a dictionary."""
    params = {
        "EXPERIMENT_CASE": 'imbalance',  # 'imbalance' or 'balance'
        "SAMPLE_SIZES": [200],
        "NUM_REPLICATIONS": 10,
        "USE_REAL_CIFAR10": True,
        "PRETRAIN_EPOCHS": 50,
        "EARLY_STOP_PATIENCE": 5,
        "POST_EARLY_STOP_EPOCHS": 1,
        "LEARNING_RATE_SOURCE": 0.001,
        "TARGET_EPOCHS": 50,
        "TARGET_PATIENCE": 5,
        "LEARNING_RATE_TARGET": 0.001,
        "FEATURE_EXTRACTION": True,
        "FT_SAMPLE_SIZE": 1000,
        "NOTEBOOK_NAME": "pretrained",
    }
    params["RESULTS_DIR"] = os.path.join(os.getcwd(), params["NOTEBOOK_NAME"])
    os.makedirs(params["RESULTS_DIR"], exist_ok=True)
    print(f"Results Directory: {params['RESULTS_DIR']}")
    print(f"Mode: {params['EXPERIMENT_CASE']} | Sample sizes: {params['SAMPLE_SIZES']} | Replications: {params['NUM_REPLICATIONS']}")
    print(f"Post early-stopping control epochs: {params['POST_EARLY_STOP_EPOCHS']}")
    return params
""")
save()
print("Cell 5 done")

# Cell 7: load_experiment_data()
set_src(7, """from models import CIFAR6CNN, create_transferred_model, set_trainable_layers
from dataset_splits import (
    split_cifar_imbalance,
    split_cifar_balance,
    generate_synthetic_transfer_dataset,
    make_dataloaders,
    create_ft_dataloaders_from_subsets
)
from di_algorithm import default_eval_loss
from projection_correlation import projection_correlation, save_experiment_results
from layer_extractor import extract_layer_activations


def load_experiment_data(experiment_case: str, seed: int, use_real_cifar10: bool,
                         n_source: int = 1200, n_target: int = 800,
                         batch_size: int = 64) -> Dict[str, Any]:
    """Load data, split into source/target, create dataloaders."""
    if use_real_cifar10:
        from cifar10_experiment import load_cifar10_raw
        print("Loading CIFAR-10 dataset...")
        raw_x, raw_y = load_cifar10_raw(data_dir='./data')
        if experiment_case == 'imbalance':
            splits = split_cifar_imbalance(raw_x, raw_y, seed=seed)
        else:
            splits = split_cifar_balance(raw_x, raw_y, seed=seed)
    else:
        print("Generating synthetic transfer dataset...")
        num_classes = 5 if experiment_case == 'imbalance' else 10
        transition_layer = 3 if experiment_case == 'imbalance' else 4
        splits = generate_synthetic_transfer_dataset(
            n_source=n_source, n_target=n_target,
            num_classes=num_classes,
            transition_layer=transition_layer,
            seed=seed
        )
    generator = torch.Generator().manual_seed(seed)
    loaders = make_dataloaders(splits, batch_size=batch_size, generator=generator)
    target_train_x, target_train_y = splits['target_train']
    target_test_loader = loaders['target_test']
    print(f"Source train size: {len(splits['source_train'][0])} | test: {len(splits['source_test'][0])}")
    print(f"Target train size: {len(target_train_x)} | test: {len(splits['target_test'][0])}")
    print(f"Source classes: {splits['num_source_classes']} | Target classes: {splits['num_target_classes']}")
    return {
        "splits": splits,
        "loaders": loaders,
        "target_train_x": target_train_x,
        "target_train_y": target_train_y,
        "target_test_loader": target_test_loader,
        "generator": generator,
        "num_source_classes": splits['num_source_classes'],
        "num_target_classes": splits['num_target_classes'],
    }
""")
save()
print("Cell 7 done")

# Cell 9: select_fixed_subsets call
set_src(9, """# Pre-select and fix target subsets
fixed_target_subsets = select_fixed_subsets(
    target_train_x,
    target_train_y,
    sample_sizes=SAMPLE_SIZES,
    num_replications=NUM_REPLICATIONS,
    base_seed=42
)

print(f"Fixed {len(fixed_target_subsets)} sample size configurations with {NUM_REPLICATIONS} replications each.")
""")
save()
print("Cell 9 done")

# Cell 11: run_stage1_pretraining (part 1)
set_src(11, """def run_stage1_pretraining(
    device: torch.device,
    seed: int,
    loaders: Dict[str, Any],
    num_source_classes: int,
    fixed_target_subsets: Dict[int, List[Dict[str, Any]]],
    primary_sample_size: int,
    num_target_classes: int,
    pretrain_epochs: int = 50,
    post_early_stop_epochs: int = 1,
    early_stop_patience: int = 5,
    learning_rate_source: float = 0.001,
    best_model_criterion: str = 'pc',
    one_hot: bool = True,
) -> Dict[str, Any]:
    """Run Stage 1: Source pretraining with early stopping and model selection."""
    print("Stage 1.4")
    print("=" * 75)
    print("[Stage 1] Pretraining 6-layer CNN on Source Dataset")
    print(f"Early stopping patience: {early_stop_patience} | Post-early-stop control: {post_early_stop_epochs} epochs")
    print("=" * 75)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    source_model = CIFAR6CNN(num_classes=num_source_classes).to(device)
    optimizer = torch.optim.RMSprop(source_model.parameters(), lr=learning_rate_source, alpha=0.9)
    criterion = nn.CrossEntropyLoss()
    stage1_history = []
    epoch_states = []
    best_s_val_loss = float('inf')
    best_source_state = None
    best_pc_epoch = None
    best_ipc_epoch = None
    best_pc_value = float('-inf')
    best_ipc_value = float('-inf')
    early_stop_triggered = False
    early_stop_epoch = None
    extra_epochs_left = post_early_stop_epochs
    patience_counter = 0
    debug_saved = False

    layer_names = CIFAR6CNN.LAYER_NAMES
""")
# Cell 11 part 2a: training loop
curr11 = ''.join(nb['cells'][11]['source'])
add11a = '''
    for ep in range(1, pretrain_epochs + post_early_stop_epochs + 1):
        time_0 = time.perf_counter()
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
        time_1 = time.perf_counter()
        if 1 == ep:
            print(f"train time{time_1 - time_0}")
        s_val_loss, s_test_acc = default_eval_loss(source_model, loaders['source_test'], criterion, device)
        time_2 = time.perf_counter()
        if 1 == ep:
            print(f"eval_loss {time_2 - time_1}")
        t_val_loss = 0
        if (early_stop_triggered or ep == 1) and not debug_saved:
            pc_stats, ipc_stats = evaluate_layer_pc_profiles(
                source_model, fixed_target_subsets, device=device,
                epoch=ep, Debug_Info=1,
                one_hot=one_hot, num_classes=num_target_classes
            )
            debug_saved = True
        else:
            pc_stats, ipc_stats = evaluate_layer_pc_profiles(
                source_model, fixed_target_subsets, device=device,
                one_hot=one_hot, num_classes=num_target_classes
            )
        pc_str = format_pc_summary(pc_stats, ipc_stats, target_size=primary_sample_size)
        time_3 = time.perf_counter()
        if 1 == ep:
            print(f"PC evaluation {time_3 - time_2}")
        prefix_tag = '[Control]' if early_stop_triggered else '[Pretrain]'
        print(f"{prefix_tag} Epoch {ep:02d}/{pretrain_epochs:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Source Test Acc: {s_test_acc * 100:.1f}% | "
              f"Source Val Loss: {s_val_loss:.4f} | "
              f"Target Val Loss: {t_val_loss:.4f} | "
              f"PC: {pc_str}")
        stage1_history.append({
            'epoch': ep,
            'train_loss': float(avg_train_loss),
            'source_val_loss': float(s_val_loss),
            'source_test_acc': float(s_test_acc * 100.0),
            'target_val_loss': float(t_val_loss),
            'pc_profiles': pc_stats,
            'is_control_phase': early_stop_triggered
        })
        epoch_states.append({k: v.cpu().clone() for k, v in source_model.state_dict().items()})
        pc_means = [pc_stats[primary_sample_size][l]['mean'] for l in layer_names]
        ipc_means = [ipc_stats[primary_sample_size][l]['mean'] for l in layer_names]
        max_pc = max(pc_means)
        max_ipc = max(ipc_means)
        if max_pc > best_pc_value:
            best_pc_value = max_pc
            best_pc_epoch = ep - 1
        if max_ipc > best_ipc_value:
            best_ipc_value = max_ipc
            best_ipc_epoch = ep - 1
        if not early_stop_triggered:
            if s_val_loss < best_s_val_loss:
                best_s_val_loss = s_val_loss
                best_source_state = {k: v.cpu().clone() for k, v in source_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    early_stop_triggered = True
                    early_stop_epoch = ep
                    print(f"\\n>>> [Early Stop Triggered] at epoch {ep} (Patience={early_stop_patience}).")
                    print(f">>> Running {post_early_stop_epochs} additional control epochs...")
        else:
            extra_epochs_left -= 1
            if extra_epochs_left <= 0:
                print(f"\\n>>> Completed {post_early_stop_epochs} post-early-stop control epochs. Stopping Stage 1.")
                break
'''
nb['cells'][11]['source'] = (curr11 + add11a).splitlines(keepends=True)

# Cell 11 part 2b: model selection + saving + return
add11b = '''
    if best_model_criterion == 'source_val_loss':
        print(f"Best model selected by Source Val Loss (epoch {early_stop_epoch if early_stop_epoch else 'N/A'}, best_s_val_loss={best_s_val_loss:.4f})")
    elif best_model_criterion == 'pc':
        best_source_state = {k: v.cpu().clone() for k, v in epoch_states[best_pc_epoch].items()}
        print(f"Best model selected by PC (epoch {best_pc_epoch + 1}, max PC={best_pc_value:.4f})")
    elif best_model_criterion == 'ipc':
        best_source_state = {k: v.cpu().clone() for k, v in epoch_states[best_ipc_epoch].items()}
        print(f"Best model selected by IPC (epoch {best_ipc_epoch + 1}, max IPC={best_ipc_value:.4f})")
    elif best_model_criterion == 'frankenstein':
        pc_per_epoch = []
        for h in stage1_history:
            pc_per_epoch.append(h['pc_profiles'][primary_sample_size])
        selected_epochs = {}
        selected_epochs['conv1'] = max(range(len(epoch_states)), key=lambda e: pc_per_epoch[e]['conv1']['mean'])
        for j in range(1, len(layer_names)):
            prev_layer = layer_names[j-1]
            curr_layer = layer_names[j]
            selected_epochs[curr_layer] = max(range(len(epoch_states)), key=lambda e: abs(pc_per_epoch[e][prev_layer]['mean'] - pc_per_epoch[e][curr_layer]['mean']))
        frankenstein_state_dict = {}
        for j, layer_name in enumerate(layer_names):
            src_epoch = selected_epochs[layer_name]
            prefixes = (f'layers.{j}.', f'{layer_name}.')
            for key, value in epoch_states[src_epoch].items():
                if key.startswith(prefixes):
                    frankenstein_state_dict[key] = value.clone()
        for key, value in epoch_states[-1].items():
            if key.startswith('classifier'):
                frankenstein_state_dict[key] = value.clone()
        best_source_state = frankenstein_state_dict
        print(f"Frankenstein model assembled: layer epochs = {selected_epochs}")
    else:
        raise ValueError(f"Unknown BEST_MODEL_CRITERION: {best_model_criterion}")

    stage1_ts = get_timestamp()
    model_save_path = make_result_path("pretrained_model", "pt", stage1_ts)
    torch.save({
        'best_state_dict': best_source_state,
        'final_state_dict': source_model.state_dict(),
        'early_stop_epoch': early_stop_epoch,
        'best_s_val_loss': best_s_val_loss}, model_save_path)
    print(f"Pretrained model checkpoint saved to: {model_save_path}")
    history_save_path = make_result_path("stage1_history", "json", stage1_ts)
    with open(history_save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'stage': 'stage1_pretraining',
            'experiment_case': EXPERIMENT_CASE if 'EXPERIMENT_CASE' in dir() else 'unknown',
            'early_stop_epoch': early_stop_epoch,
            'post_early_stop_epochs': post_early_stop_epochs,
            'best_s_val_loss': best_s_val_loss,
            'history': stage1_history}, f, indent=2)
    print(f"Stage 1 metrics history saved to: {history_save_path}")

    return {
        'best_source_state': best_source_state,
        'stage1_history': stage1_history,
        'stage1_ts': stage1_ts,
        'source_model': source_model,
        'early_stop_epoch': early_stop_epoch,
        'best_s_val_loss': best_s_val_loss,
    }
'''
nb['cells'][11]['source'] = (curr11 + add11a + add11b).splitlines(keepends=True)
# Cell 13: plot_stage1_visualizations
set_src(13, """def plot_stage1_visualizations(
    stage1_history: List[Dict[str, Any]],
    stage1_ts: str,
    results_dir: str,
    primary_sample_size: int,
    experiment_case: str,
    early_stop_epoch: Optional[int] = None,
    post_early_stop_epochs: int = 1,
) -> None:
    """Plot Stage 1 training curves and PC evolution."""
    epochs_arr = [h['epoch'] for h in stage1_history]
    tr_loss_arr = [h['train_loss'] for h in stage1_history]
    s_val_loss_arr = [h['source_val_loss'] for h in stage1_history]
    t_val_loss_arr = [h['target_val_loss'] for h in stage1_history]
    s_acc_arr = [h['source_test_acc'] for h in stage1_history]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(epochs_arr, tr_loss_arr, 'o-', color='#1f77b4', label='Source Train Loss', linewidth=2, markersize=4)
    axes[0].plot(epochs_arr, s_val_loss_arr, 's-', color='#d62728', label='Source Val Loss', linewidth=2, markersize=4)
    axes[0].plot(epochs_arr, t_val_loss_arr, '^--', color='#9467bd', label='Target Val Loss', linewidth=2, markersize=4)
    if early_stop_epoch:
        axes[0].axvline(early_stop_epoch, color='black', linestyle=':', linewidth=2, label=f'Early Stop (ep {early_stop_epoch})')
        axes[0].axvspan(early_stop_epoch, epochs_arr[-1], alpha=0.12, color='orange', label=f'+{post_early_stop_epochs} Control Epochs')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Stage 1: Training & Validation Losses', fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

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
    print(f"Stage 1 training curves saved to: {curves_path}")

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
    plt.title(f'Stage 1: Layer-wise PC Evolution over Pretraining Epochs ({experiment_case.upper()} Case)', fontweight='bold', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    pc_evolution_path = make_result_path("stage1_pc_evolution", "png", stage1_ts)
    plt.savefig(pc_evolution_path, dpi=150)
    plt.show()
    print(f"Stage 1 PC evolution plot saved to: {pc_evolution_path}")
""")

# Cell 15: run_stage2_finetuning (part 1)
set_src(15, """def run_stage2_finetuning(
    best_source_state: Optional[Dict[str, torch.Tensor]],
    source_model: CIFAR6CNN,
    loaders: Dict[str, Any],
    target_test_loader: DataLoader,
    fixed_target_subsets: Dict[int, List[Dict[str, Any]]],
    device: torch.device,
    target_train_x: torch.Tensor,
    target_train_y: torch.Tensor,
    num_target_classes: int,
    target_epochs: int = 50,
    target_patience: int = 5,
    learning_rate_target: float = 0.001,
    feature_extraction: bool = True,
    primary_sample_size: int = 200,
    ft_sample_size: Optional[int] = None,
    seed: int = 42,
    generator: Optional[torch.Generator] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run Stage 2: Target fine-tuning on all 7 candidate models M_0...M_6."""
    if verbose:
        print("=" * 75)
        print("[Stage 2] Target Fine-Tuning on Fixed SAMPLE_SIZES for All 7 Cases (M_0 ... M_6)")
        print(f"Fixed sample size: n={primary_sample_size} | Early stopping patience: {target_patience} | Max epochs: {target_epochs}")
        print("=" * 75)

    ft_sample_size = ft_sample_size if ft_sample_size is not None else primary_sample_size
    target_train_subset_loader, target_val_subset_loader = create_ft_dataloaders_from_subsets(
        fixed_target_subsets=fixed_target_subsets,
        target_train_x=target_train_x,
        target_train_y=target_train_y,
        sample_size=primary_sample_size,
        ft_sample_size=ft_sample_size,
        batch_size=64,
        seed=seed,
        generator=generator,
    )
    target_criterion = nn.CrossEntropyLoss()
    stage2_results = {}
    stage2_ts = get_timestamp()
    layer_names = CIFAR6CNN.LAYER_NAMES

    for k in range(0, 7):
        model_name = f"M_{k}"
        if verbose:
            print("\\n" + "=" * 75)
            print(f"[Stage 2] Model {model_name} (transferred k={k} layers from pre-trained source model)")
            print(f"Fine-tuning on fixed target subset (n={primary_sample_size}), monitored metric: Target Validation Loss")
            print("=" * 75)
        if best_source_state is not None:
            source_model.load_state_dict(best_source_state)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        target_model = create_transferred_model(
            pretrained_model=source_model,
            k=k,
            num_target_classes=num_target_classes,
            freeze_transferred=feature_extraction
        ).to(device)
        set_trainable_layers(target_model, k_freeze=k if feature_extraction else 0)
        target_optimizer = torch.optim.RMSprop(
            target_model.parameters(),
            lr=learning_rate_target,
            alpha=0.9
        )
        stage2_history_k = []
        best_t_val_loss = float('inf')
        best_t_val_acc = 0.0
        best_target_state = None
        target_patience_counter = 0
        target_early_stop_epoch = None

# Cell 15 part 2: training loop body
set_src(15, """                avg_train_loss = tr_loss / len(target_train_subset_loader.dataset)
                val_loss, val_acc = default_eval_loss(target_model, target_val_subset_loader, target_criterion, device)
                test_loss, test_acc = default_eval_loss(target_model, target_test_loader, target_criterion, device)
                try:
                    pc_stats, ipc_stats = evaluate_layer_pc_profiles(
                        target_model, fixed_target_subsets, device=device,
                        one_hot=True, num_classes=num_target_classes
                    )
                except TypeError:
                    pc_stats, ipc_stats = evaluate_layer_pc_profiles(target_model, fixed_target_subsets, device=device)
                pc_str = format_pc_summary(pc_stats, ipc_stats, target_size=primary_sample_size)
                if verbose:
                    print(f"[{model_name}] Epoch {ep:02d}/{target_epochs:02d} | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.1f}% | "
                          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.1f}% | "
                          f"PC: {pc_str}")
                stage2_history_k.append({
                    'epoch': ep,
                    'train_loss': float(avg_train_loss),
                    'val_loss': float(val_loss),
                    'val_acc': float(val_acc * 100.0),
                    'test_loss': float(test_loss),
                    'test_acc': float(test_acc * 100.0),
                    'pc_profiles': pc_stats,
                })
                if val_loss < best_t_val_loss:
                    best_t_val_loss = val_loss
                    best_t_val_acc = val_acc
                    best_target_state = {k2: v.cpu().clone() for k2, v in target_model.state_dict().items()}
                    target_patience_counter = 0
                else:
                    target_patience_counter += 1
                    if target_patience_counter >= target_patience:
                        target_early_stop_epoch = ep
                        if verbose:
                            print(f"\\n>>> [{model_name}] Early stopping at epoch {ep} (Patience={target_patience}).")
                        break
""", 0)
""")