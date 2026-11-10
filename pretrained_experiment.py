"""
Pretrained Transfer Learning Experiment - Di et al. (2025).

Task description (English):
==========================
Implementation of a two-stage transfer learning pipeline based on Di et al. (2025):
1. Target Fixed Subsets Setup:
   - At initialization, draw and fix SAMPLE_SIZES instances from the target dataset
     NUM_REPLICATIONS times with fixed random seeds for reproducible target validation
     loss and layer-wise Projection Correlation (PC) evaluation with confidence intervals.
2. Stage 1 - Source Pretraining with Early Stopping & Control:
   - Pretrain a 6-layer CNN on the source dataset with early stopping monitored on source validation loss.
   - Upon triggering early stopping, continue training for an additional control period
     defined by the named constant POST_EARLY_STOP_EPOCHS = 10.
   - Log each epoch: Train Loss, Source Validation Loss, Source Test Accuracy,
     Target Validation Loss, and PC for all layers across replications.
   - Plot loss/accuracy curves and PC layer trajectories.
   - Save the pretrained model checkpoint and metrics to pretrained/ with timestamps.
3. Stage 2 - Target Fine-Tuning with Early Stopping:
   - Fine-tune the model on the fixed SAMPLE_SIZES target subset using early stopping on target validation loss.
   - Validate on the target validation set.
   - Log each epoch: Train Loss, Target Validation Loss, Test Accuracy, and PC for all layers.
   - Plot training curves and PC trajectories.
   - Save the fine-tuned model checkpoint and metrics to pretrained/ with timestamps.
4. Artifacts & Logging:
   - All experiment outputs (checkpoints, metrics JSON, visualization plots) are stored
     in a dedicated folder matching the notebook name ('pretrained/'), with an execution
     timestamp appended to every artifact filename.
"""

import argparse
import sys
import os
import json
import time
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .models import CIFAR6CNN, create_transferred_model, set_trainable_layers
    from .projection_correlation import projection_correlation
    from .layer_extractor import extract_layer_activations
    from .dataset_splits import (
        split_cifar_imbalance,
        split_cifar_balance,
        generate_synthetic_transfer_dataset,
        make_dataloaders,
        CIFAR_SOURCE_CLASSES,
        CIFAR_TARGET_CLASSES,
    )
    from .di_algorithm import default_eval_loss
except ImportError:
    from models import CIFAR6CNN, create_transferred_model, set_trainable_layers
    from projection_correlation import projection_correlation
    from layer_extractor import extract_layer_activations
    from dataset_splits import (
        split_cifar_imbalance,
        split_cifar_balance,
        generate_synthetic_transfer_dataset,
        make_dataloaders,
        CIFAR_SOURCE_CLASSES,
        CIFAR_TARGET_CLASSES,
    )
    from di_algorithm import default_eval_loss

# Named constants
POST_EARLY_STOP_EPOCHS = 10
EARLY_STOP_PATIENCE = 5


def _timestamp() -> str:
    """Return a compact timestamp string YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _results_dir(folder_name: str = "pretrained") -> str:
    """Return results directory path, ensuring it exists."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(base_dir, folder_name)
    os.makedirs(d, exist_ok=True)
    return d


def _save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {path}")


def select_fixed_subsets(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    sample_sizes: List[int],
    num_replications: int,
    base_seed: int = 42
) -> Dict[int, List[Dict[str, Any]]]:
    """Pre-sample and fix deterministic subsets for PC and target validation."""
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
                batch_size=128 if torch.cuda.is_available() else 64,
                shuffle=False,
                pin_memory=torch.cuda.is_available()
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


def evaluate_layer_pc_profiles(
    model: nn.Module,
    subsets_dict: Dict[int, List[Dict[str, Any]]],
    device: torch.device,
    pooling: str = 'gap',
    one_hot: bool = False,
    num_classes: Optional[int] = None,
    **kwargs
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Compute layer-wise PC on the pre-fixed subsets across replications."""
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
                if one_hot and num_classes is not None:
                    labels_one_hot = torch.zeros(len(labels), num_classes, device=device)
                    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                    labels_for_pc = labels_one_hot
                else:
                    labels_for_pc = labels
                pc_val = projection_correlation(acts[l], labels_for_pc, device=device)
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
    """Format PC stats as a readable string for logging."""
    stats = pc_stats.get(target_size, {})
    parts = []
    for l in CIFAR6CNN.LAYER_NAMES:
        if l in stats:
            parts.append(f"{l}={stats[l]['mean']:.3f}±{stats[l]['std']:.3f}")
    return " | ".join(parts)


def _plot_training_curves(
    history: List[dict],
    save_path: str,
    title_suffix: str = "",
    early_stop_epoch: Optional[int] = None,
    post_epochs: int = 0
) -> None:
    """Plot train loss, val loss, and accuracy over epochs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[Plot Warning] matplotlib not available. Skipping plot: {save_path}")
        return

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h.get('val_loss', h.get('source_val_loss', np.nan)) for h in history]
    test_acc = [h.get('test_acc', h.get('source_test_acc', np.nan)) for h in history]
    target_val_loss = [h.get('target_val_loss', np.nan) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, train_loss, 'o-', color='#1f77b4', linewidth=2, markersize=4, label='Train Loss')
    axes[0].set_xlabel("Epoch", fontweight='bold')
    axes[0].set_ylabel("Loss", fontweight='bold')
    axes[0].set_title("Training Loss", fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    if early_stop_epoch:
        axes[0].axvline(early_stop_epoch, color='black', linestyle=':', label='Early Stop')
    axes[0].legend()

    axes[1].plot(epochs, val_loss, 's-', color='#d62728', linewidth=2, markersize=4, label='Val Loss')
    if any(not np.isnan(v) for v in target_val_loss):
        axes[1].plot(epochs, target_val_loss, '^--', color='#9467bd', linewidth=2, markersize=4, label='Target Val Loss')
    axes[1].set_xlabel("Epoch", fontweight='bold')
    axes[1].set_ylabel("Loss", fontweight='bold')
    axes[1].set_title("Validation Loss", fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    if early_stop_epoch:
        axes[1].axvline(early_stop_epoch, color='black', linestyle=':', label='Early Stop')
        if post_epochs > 0:
            axes[1].axvspan(early_stop_epoch, epochs[-1], alpha=0.15, color='orange', label=f'+{post_epochs} Control')
    axes[1].legend()

    axes[2].plot(epochs, test_acc, '^-', color='#2ca02c', linewidth=2, markersize=4, label='Accuracy')
    axes[2].set_xlabel("Epoch", fontweight='bold')
    axes[2].set_ylabel("Accuracy (%)", fontweight='bold')
    axes[2].set_title("Test Accuracy", fontweight='bold')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    if early_stop_epoch:
        axes[2].axvline(early_stop_epoch, color='black', linestyle=':', label='Early Stop')
    axes[2].legend()

    plt.suptitle(f"Training Curves{title_suffix}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {save_path}")


def _plot_pc_evolution(
    history: List[dict],
    save_path: str,
    target_size: int,
    title_suffix: str = "",
    early_stop_epoch: Optional[int] = None
) -> None:
    """Plot layer-wise PC trajectory across epochs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[Plot Warning] matplotlib not available. Skipping plot: {save_path}")
        return

    epochs = [h['epoch'] for h in history]
    layer_names = CIFAR6CNN.LAYER_NAMES
    cmap = plt.cm.get_cmap('tab10', len(layer_names))

    plt.figure(figsize=(12, 6))
    for idx, layer in enumerate(layer_names):
        means = [h['pc_profiles'][target_size][layer]['mean'] for h in history]
        stds = [h['pc_profiles'][target_size][layer]['std'] for h in history]
        plt.errorbar(
            epochs, means, yerr=stds,
            fmt='o-', color=cmap(idx),
            linewidth=2, markersize=4, capsize=3,
            label=f"{layer}"
        )

    if early_stop_epoch:
        plt.axvline(early_stop_epoch, color='black', linestyle=':', linewidth=2, label=f'Early Stop (ep {early_stop_epoch})')
        plt.axvspan(early_stop_epoch, epochs[-1], alpha=0.1, color='orange', label='Control Phase')

    plt.xlabel("Epoch", fontweight='bold')
    plt.ylabel(f"Projection Correlation (n={target_size})", fontweight='bold')
    plt.title(f"Layer-wise PC Evolution{title_suffix}", fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PC evolution plot saved to: {save_path}")


def _plot_stage2_comparison(
    stage2_results: Dict[int, dict],
    save_path: str,
    case_name: str = "imbalance"
) -> None:
    """Plot comparative loss, accuracy, and depth performance across all 7 models."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[Plot Warning] matplotlib not available. Skipping plot: {save_path}")
        return

    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    model_markers = ['o', 's', '^', 'v', 'D', 'p', '*']
    ks = list(range(0, 7))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Val Loss vs Epoch
    for k in ks:
        hist_k = stage2_results[k]['history']
        eps = [h['epoch'] for h in hist_k]
        v_loss = [h['val_loss'] for h in hist_k]
        es = stage2_results[k]['early_stop_epoch']
        axes[0].plot(eps, v_loss, marker=model_markers[k], color=model_colors[k],
                     linewidth=1.8, markersize=4, label=f"M_{k} (k={k})")
        if es and es <= len(v_loss):
            axes[0].scatter([es], [v_loss[es-1]], color='black', s=50, zorder=5, marker='x')

    axes[0].set_xlabel('Fine-Tuning Epoch', fontweight='bold')
    axes[0].set_ylabel('Target Validation Loss', fontweight='bold')
    axes[0].set_title('Target Val Loss vs Epoch (M_0..M_6)', fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(fontsize=9, loc='upper right')

    # 2. Test Accuracy vs Epoch
    for k in ks:
        hist_k = stage2_results[k]['history']
        eps = [h['epoch'] for h in hist_k]
        t_acc = [h['test_acc'] for h in hist_k]
        axes[1].plot(eps, t_acc, marker=model_markers[k], color=model_colors[k],
                     linewidth=1.8, markersize=4, label=f"M_{k} (k={k})")

    axes[1].set_xlabel('Fine-Tuning Epoch', fontweight='bold')
    axes[1].set_ylabel('Target Test Accuracy (%)', fontweight='bold')
    axes[1].set_title('Target Test Accuracy vs Epoch (M_0..M_6)', fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(fontsize=9, loc='lower right')

    # 3. Performance vs Transferred Depth (k)
    best_losses = [stage2_results[k]['best_val_loss'] for k in ks]
    best_accs = [stage2_results[k]['best_test_acc'] for k in ks]
    best_k_loss = min(ks, key=lambda k: best_losses[k])

    ax_loss = axes[2]
    ax_acc = ax_loss.twinx()

    line1 = ax_loss.plot(ks, best_losses, 's-', color='#d62728', linewidth=2.5, markersize=7, label='Best Val Loss')
    line2 = ax_acc.plot(ks, best_accs, 'o--', color='#2ca02c', linewidth=2.5, markersize=7, label='Best Test Acc (%)')
    ax_loss.axvline(best_k_loss, color='#333333', linestyle=':', linewidth=2, label=f'Optimal k*={best_k_loss}')

    ax_loss.set_xlabel('Number of Transferred Layers (k)', fontweight='bold')
    ax_loss.set_ylabel('Best Target Val Loss', color='#d62728', fontweight='bold')
    ax_acc.set_ylabel('Best Target Test Accuracy (%)', color='#2ca02c', fontweight='bold')
    ax_loss.set_title('Performance vs Transferred Depth (k)', fontweight='bold')
    ax_loss.set_xticks(ks)
    ax_loss.set_xticklabels([f"M_{k}\n(k={k})" for k in ks])
    ax_loss.grid(True, linestyle=':', alpha=0.6)

    lines = line1 + line2 + [plt.Line2D([0], [0], color='#333333', linestyle=':', label=f'Optimal k*={best_k_loss}')]
    labels = [l.get_label() for l in lines]
    ax_loss.legend(lines, labels, loc='center right', fontsize=9)

    plt.suptitle(f"Stage 2: Target Fine-Tuning Comparison across All 7 Models ({case_name.upper()} Case)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Stage 2 comparative curves saved to: {save_path}")


def _plot_stage2_individual_curves(
    stage2_results: Dict[int, dict],
    save_path: str,
    case_name: str = "imbalance"
) -> None:
    """Plot individual fine-tuning loss and accuracy curves for each of the 7 models."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[Plot Warning] matplotlib not available. Skipping plot: {save_path}")
        return

    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    ks = list(range(0, 7))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes_flat = axes.flatten()

    for k in ks:
        ax = axes_flat[k]
        hist_k = stage2_results[k]['history']
        eps = [h['epoch'] for h in hist_k]
        tr_loss = [h['train_loss'] for h in hist_k]
        v_loss = [h['val_loss'] for h in hist_k]
        t_acc = [h['test_acc'] for h in hist_k]
        es = stage2_results[k]['early_stop_epoch']

        ax2 = ax.twinx()
        l1 = ax.plot(eps, tr_loss, 'o-', color='#1f77b4', linewidth=1.8, markersize=3, label='Train Loss')
        l2 = ax.plot(eps, v_loss, 's-', color='#d62728', linewidth=1.8, markersize=3, label='Val Loss')
        l3 = ax2.plot(eps, t_acc, '^--', color='#2ca02c', linewidth=1.5, markersize=3, label='Test Acc (%)')

        if es:
            ax.axvline(es, color='black', linestyle=':', linewidth=1.8, label=f'Early Stop (ep {es})')

        ax.set_title(f"Model M_{k} (k={k} transferred)", fontweight='bold', fontsize=11)
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Loss', fontsize=9)
        ax2.set_ylabel('Accuracy (%)', color='#2ca02c', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)

        if k == 0:
            lines_all = l1 + l2 + l3
            labels_all = [l.get_label() for l in lines_all]
            ax.legend(lines_all, labels_all, loc='upper right', fontsize=8)

    # 8th subplot: Summary Bar Chart
    ax_sum = axes_flat[7]
    best_accs = [stage2_results[k]['best_test_acc'] for k in ks]
    bars = ax_sum.bar(ks, best_accs, color=model_colors, edgecolor='black', alpha=0.85)
    ax_sum.set_xticks(ks)
    ax_sum.set_xticklabels([f"M_{k}" for k in ks])
    ax_sum.set_xlabel('Model', fontweight='bold')
    ax_sum.set_ylabel('Best Test Accuracy (%)', fontweight='bold')
    ax_sum.set_title('Best Test Accuracy per Model', fontweight='bold', fontsize=11)
    ax_sum.grid(True, axis='y', linestyle=':', alpha=0.6)
    for bar, acc in zip(bars, best_accs):
        ax_sum.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{acc:.1f}%",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle(f"Stage 2: Individual Fine-Tuning Curves for All 7 Cases ({case_name.upper()} Case)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Stage 2 individual curves saved to: {save_path}")


def _plot_stage2_pc_trajectories(
    stage2_results: Dict[int, dict],
    save_path: str,
    primary_size: int,
    case_name: str = "imbalance"
) -> None:
    """Plot layer-wise PC trajectories across epochs for all 7 models."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[Plot Warning] matplotlib not available. Skipping plot: {save_path}")
        return

    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    model_markers = ['o', 's', '^', 'v', 'D', 'p', '*']
    ks = list(range(0, 7))
    layer_names = CIFAR6CNN.LAYER_NAMES
    try:
        cmap = plt.colormaps['tab10']
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('tab10')

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes_flat = axes.flatten()

    for k in ks:
        ax = axes_flat[k]
        hist_k = stage2_results[k]['history']
        eps = [h['epoch'] for h in hist_k]
        es = stage2_results[k]['early_stop_epoch']

        for l_idx, layer in enumerate(layer_names):
            means = [h['pc_profiles'][primary_size][layer]['mean'] for h in hist_k]
            stds = [h['pc_profiles'][primary_size][layer]['std'] for h in hist_k]
            ax.errorbar(
                eps, means, yerr=stds,
                fmt='o-', color=cmap(l_idx),
                linewidth=1.8, markersize=3, capsize=2,
                label=f"{layer}"
            )

        if es:
            ax.axvline(es, color='black', linestyle=':', linewidth=1.8, label=f'Early Stop (ep {es})')

        ax.set_title(f"PC Trajectories: Model M_{k} (k={k})", fontweight='bold', fontsize=11)
        ax.set_xlabel('Fine-Tuning Epoch', fontsize=9)
        ax.set_ylabel(f'PC (n={primary_size})', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)

    # 8th subplot: Final PC generality profiles across all 7 models
    ax_pc_comp = axes_flat[7]
    for k in ks:
        hist_k = stage2_results[k]['history']
        last_profile = hist_k[-1]['pc_profiles'][primary_size]
        final_means = [last_profile[l]['mean'] for l in layer_names]
        ax_pc_comp.plot(range(1, 7), final_means, marker=model_markers[k], color=model_colors[k],
                        linewidth=1.8, markersize=4, label=f"M_{k} (k={k})")

    ax_pc_comp.set_xticks(range(1, 7))
    ax_pc_comp.set_xticklabels(layer_names, rotation=25)
    ax_pc_comp.set_xlabel('Layer', fontweight='bold', fontsize=9)
    ax_pc_comp.set_ylabel(f'Final PC (n={primary_size})', fontweight='bold', fontsize=9)
    ax_pc_comp.set_title('Final PC Generality Profiles', fontweight='bold', fontsize=11)
    ax_pc_comp.grid(True, linestyle=':', alpha=0.5)
    ax_pc_comp.legend(fontsize=8, loc='upper right')

    axes_flat[0].legend(loc='upper right', fontsize=8)

    plt.suptitle(f"Stage 2: Layer-wise PC Trajectories for All 7 Cases ({case_name.upper()} Case, n={primary_size})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Stage 2 PC trajectories for all 7 cases saved to: {save_path}")


def run_pretrained_experiment(
    case: str = 'imbalance',
    sample_sizes: Optional[List[int]] = None,
    num_replications: int = 2,
    pretrain_epochs: int = 50,
    target_epochs: int = 50,
    use_synthetic: bool = True,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
    max_images: Optional[int] = None,
    data_dir: str = './data',
    use_hf: bool = True,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    post_early_stop_epochs: int = POST_EARLY_STOP_EPOCHS,
):
    if sample_sizes is None:
        sample_sizes = [500, 1000]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results_dir = output_dir if output_dir else _results_dir("pretrained")
    os.makedirs(results_dir, exist_ok=True)
    ts = _timestamp()

    print("=" * 75)
    print(f"Pretrained Transfer Learning Pipeline: {case.upper()} Case")
    print(f"Results Directory: {results_dir}")
    print(f"Sample sizes: {sample_sizes} | Replications: {num_replications}")
    print(f"Patience: {early_stop_patience} | Post early-stop control: {post_early_stop_epochs} epochs")
    print("=" * 75)

    if use_synthetic:
        splits = generate_synthetic_transfer_dataset(
            n_source=1200, n_target=800,
            num_classes=5 if case == 'imbalance' else 10,
            transition_layer=3 if case == 'imbalance' else 4,
            seed=42,
        )
    else:
        from cifar10_experiment import load_cifar10_raw
        raw_x, raw_y = load_cifar10_raw(
            data_dir=data_dir,
            max_images=max_images,
            use_hf=use_hf,
        )
        if case == 'imbalance':
            splits = split_cifar_imbalance(raw_x, raw_y)
        else:
            splits = split_cifar_balance(raw_x, raw_y)

    loaders = make_dataloaders(splits, batch_size=64)
    target_train_x, target_train_y = splits['target_train']
    target_test_loader = loaders['target_test']

    # Step 1: Pre-select and fix target subsets
    print("\n[Setup] Fixing target subsets across replications...")
    fixed_subsets = select_fixed_subsets(
        target_train_x, target_train_y,
        sample_sizes=sample_sizes,
        num_replications=num_replications,
        base_seed=42
    )
    primary_size = sample_sizes[0]

    # Stage 1: Source Pretraining
    print("\n[Stage 1] Pretraining Source 6-layer CNN with early stopping...")
    source_model = CIFAR6CNN(num_classes=splits['num_source_classes']).to(device)
    optimizer = torch.optim.RMSprop(source_model.parameters(), lr=0.001, alpha=0.9)
    criterion = nn.CrossEntropyLoss()

    stage1_history = []
    best_s_val_loss = float('inf')
    best_source_state = None
    early_stop_triggered = False
    early_stop_epoch = None
    extra_left = post_early_stop_epochs
    patience_counter = 0

    for ep in range(1, pretrain_epochs + post_early_stop_epochs + 1):
        source_model.train()
        tr_loss = 0.0
        for x, y in loaders['source_train']:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = source_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
        avg_train_loss = tr_loss / len(loaders['source_train'].dataset)

        s_val_loss, s_test_acc = default_eval_loss(source_model, loaders['source_test'], criterion, device)
        t_val_loss, _ = default_eval_loss(source_model, target_test_loader, criterion, device)

        pc_stats = evaluate_layer_pc_profiles(source_model, fixed_subsets, device=device)
        pc_str = format_pc_summary(pc_stats, target_size=primary_size)

        prefix_tag = "[Control]" if early_stop_triggered else "[Pretrain]"
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
                    print(f"\n>>> [Early Stop Triggered] at epoch {ep} (Patience={early_stop_patience}).")
                    print(f">>> Running {post_early_stop_epochs} additional control epochs...\n")
        else:
            extra_left -= 1
            if extra_left <= 0:
                print(f"\n>>> Completed {post_early_stop_epochs} control epochs. Stopping Stage 1.\n")
                break

    # Save Stage 1 artifacts
    pt_model_path = os.path.join(results_dir, f"pretrained_model_{ts}.pt")
    torch.save({
        'best_state_dict': best_source_state,
        'final_state_dict': source_model.state_dict(),
        'early_stop_epoch': early_stop_epoch,
        'best_s_val_loss': best_s_val_loss
    }, pt_model_path)
    print(f"Pretrained model saved to: {pt_model_path}")

    st1_json_path = os.path.join(results_dir, f"stage1_history_{ts}.json")
    _save_json(st1_json_path, {
        'stage': 'stage1_pretraining',
        'early_stop_epoch': early_stop_epoch,
        'post_early_stop_epochs': post_early_stop_epochs,
        'best_s_val_loss': best_s_val_loss,
        'history': stage1_history
    })

    _plot_training_curves(
        stage1_history,
        os.path.join(results_dir, f"stage1_curves_{ts}.png"),
        title_suffix=" (Stage 1 Pretraining)",
        early_stop_epoch=early_stop_epoch,
        post_epochs=post_early_stop_epochs
    )
    _plot_pc_evolution(
        stage1_history,
        os.path.join(results_dir, f"stage1_pc_evolution_{ts}.png"),
        target_size=primary_size,
        title_suffix=f" (Stage 1, n={primary_size})",
        early_stop_epoch=early_stop_epoch
    )

    # Stage 2: Target Fine-Tuning for All 7 Cases (M_0 ... M_6)
    print("\n" + "=" * 75)
    print(f"[Stage 2] Target Fine-Tuning on Fixed SAMPLE_SIZES for All 7 Cases (M_0 ... M_6)")
    print(f"Fixed sample size: n={primary_size} | Early stopping patience: {early_stop_patience} | Max epochs: {target_epochs}")
    print("=" * 75)

    target_loader_sub = fixed_subsets[primary_size][0]['loader']
    num_target_classes = splits['num_target_classes']

    stage2_results: Dict[int, dict] = {}

    for k in range(0, 7):
        model_name = f"M_{k}"
        print("\n" + "=" * 75)
        print(f"[Stage 2] Model {model_name} (transferred {k} layers from pre-trained source model)")
        print(f"Fine-tuning on fixed target subset (n={primary_size}), monitored metric: Target Validation Loss")
        print("=" * 75)

        if best_source_state is not None:
            source_model.load_state_dict(best_source_state)

        target_model = create_transferred_model(
            pretrained_model=source_model,
            k=k,
            num_target_classes=num_target_classes,
            freeze_transferred=False
        ).to(device)
        set_trainable_layers(target_model, k_freeze=0)

        target_opt = torch.optim.RMSprop(target_model.parameters(), lr=0.001, alpha=0.9)

        stage2_history_k = []
        best_t_val_loss = float('inf')
        best_t_val_acc = 0.0
        best_target_state = None
        target_patience_counter = 0
        target_early_stop_epoch = None

        for ep in range(1, target_epochs + 1):
            target_model.train()
            tr_loss = 0.0
            for x, y in target_loader_sub:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                target_opt.zero_grad(set_to_none=True)
                out = target_model(x)
                loss = criterion(out, y)
                loss.backward()
                target_opt.step()
                tr_loss += loss.item() * x.size(0)
            avg_train_loss = tr_loss / len(target_loader_sub.dataset)

            # Evaluate loss and accuracy on full target validation/test set
            val_loss, test_acc = default_eval_loss(target_model, target_test_loader, criterion, device)

            # Compute PC on fixed subsets during fine-tuning
            try:
                pc_stats = evaluate_layer_pc_profiles(
                    target_model, fixed_subsets, device=device,
                    one_hot=True, num_classes=num_target_classes
                )
            except TypeError:
                pc_stats = evaluate_layer_pc_profiles(target_model, fixed_subsets, device=device)
            pc_str = format_pc_summary(pc_stats, target_size=primary_size)

            # Logged each epoch:
            print(f"[{model_name}] Epoch {ep:02d}/{target_epochs:02d} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Test Acc: {test_acc * 100:.1f}% | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"PC: {pc_str}")

            stage2_history_k.append({
                'epoch': ep,
                'train_loss': float(avg_train_loss),
                'val_loss': float(val_loss),
                'test_acc': float(test_acc * 100.0),
                'pc_profiles': pc_stats
            })

            # Early stopping check on Target Validation Loss
            if val_loss < best_t_val_loss:
                best_t_val_loss = val_loss
                best_t_val_acc = test_acc * 100.0
                best_target_state = {key: val.cpu().clone() for key, val in target_model.state_dict().items()}
                target_patience_counter = 0
            else:
                target_patience_counter += 1
                if target_patience_counter >= early_stop_patience:
                    target_early_stop_epoch = ep
                    print(f"\n>>> [Target Early Stop] {model_name} triggered at epoch {ep} (Patience={early_stop_patience}). Stopping Stage 2 for {model_name}.\n")
                    break

        # Save individual model checkpoint
        ft_model_path = os.path.join(results_dir, f"finetuned_model_{model_name}_{ts}.pt")
        torch.save({
            'k': k,
            'model_name': model_name,
            'best_state_dict': best_target_state,
            'final_state_dict': target_model.state_dict(),
            'best_val_loss': best_t_val_loss,
            'best_test_acc': best_t_val_acc,
            'early_stop_epoch': target_early_stop_epoch
        }, ft_model_path)
        print(f"[{model_name}] Checkpoint saved to: {ft_model_path}")

        stage2_results[k] = {
            'k': k,
            'model_name': model_name,
            'transferred_layers': k,
            'best_val_loss': float(best_t_val_loss),
            'best_test_acc': float(best_t_val_acc),
            'early_stop_epoch': target_early_stop_epoch,
            'total_epochs': len(stage2_history_k),
            'history': stage2_history_k,
            'checkpoint_path': ft_model_path
        }

    # Save Stage 2 artifacts across all 7 cases
    best_k_idx = min(stage2_results.keys(), key=lambda k: stage2_results[k]['best_val_loss'])
    best_target_info = stage2_results[best_k_idx]

    st2_json_path = os.path.join(results_dir, f"stage2_history_all_cases_{ts}.json")
    _save_json(st2_json_path, {
        'stage': 'stage2_fine_tuning_all_cases',
        'experiment_case': case,
        'target_sample_size': primary_size,
        'early_stop_patience': early_stop_patience,
        'target_epochs': target_epochs,
        'best_overall_k': best_k_idx,
        'best_overall_val_loss': best_target_info['best_val_loss'],
        'best_overall_test_acc': best_target_info['best_test_acc'],
        'models': {
            f"M_{k}": {
                'k': res['k'],
                'best_val_loss': res['best_val_loss'],
                'best_test_acc': res['best_test_acc'],
                'early_stop_epoch': res['early_stop_epoch'],
                'total_epochs': res['total_epochs'],
                'history': res['history']
            } for k, res in stage2_results.items()
        }
    })
    print(f"\nStage 2 metrics history (all 7 cases) saved to: {st2_json_path}")

    # Summary table printout
    print("\n" + "=" * 80)
    print("STAGE 2 SUMMARY TABLE (ALL 7 CASES: M_0 ... M_6)")
    print("=" * 80)
    print(f"{'Model':<8} | {'Transferred (k)':<16} | {'Best Val Loss':<14} | {'Best Test Acc (%)':<18} | {'Early Stop Epoch':<16}")
    print("-" * 80)
    for k in range(0, 7):
        res = stage2_results[k]
        es_str = f"Epoch {res['early_stop_epoch']}" if res['early_stop_epoch'] else f"Max ({res['total_epochs']})"
        print(f"{res['model_name']:<8} | {res['transferred_layers']:<16} | {res['best_val_loss']:<14.4f} | {res['best_test_acc']:<18.2f} | {es_str:<16}")
    print("-" * 80)
    print(f"Optimal Model: M_{best_k_idx} (k* = {best_k_idx}) | Val Loss = {best_target_info['best_val_loss']:.4f} | Test Acc = {best_target_info['best_test_acc']:.2f}%")
    print("=" * 80)

    # Visualizations
    _plot_stage2_comparison(
        stage2_results,
        os.path.join(results_dir, f"stage2_curves_comparison_{ts}.png"),
        case_name=case
    )
    _plot_stage2_individual_curves(
        stage2_results,
        os.path.join(results_dir, f"stage2_curves_all_7_cases_{ts}.png"),
        case_name=case
    )
    _plot_stage2_pc_trajectories(
        stage2_results,
        os.path.join(results_dir, f"stage2_pc_trajectories_all_7_cases_{ts}.png"),
        primary_size=primary_size,
        case_name=case
    )

    # Legacy plots for best model (backward compatibility)
    _plot_training_curves(
        best_target_info['history'],
        os.path.join(results_dir, f"stage2_curves_{ts}.png"),
        title_suffix=f" (Stage 2 Best Model M_{best_k_idx})",
        early_stop_epoch=best_target_info['early_stop_epoch']
    )
    _plot_pc_evolution(
        best_target_info['history'],
        os.path.join(results_dir, f"stage2_pc_evolution_{ts}.png"),
        target_size=primary_size,
        title_suffix=f" (Stage 2 Best Model M_{best_k_idx}, n={primary_size})",
        early_stop_epoch=best_target_info['early_stop_epoch']
    )

    print("\n" + "=" * 75)
    print("PRETRAINED EXPERIMENT COMPLETED")
    print(f"All artifacts saved with timestamp '{ts}' in: '{results_dir}'")
    print("=" * 75)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pretrained transfer learning experiment (Di et al. 2025)."
    )
    parser.add_argument('--case', choices=['imbalance', 'balance'], default='imbalance')
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--target_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument('--post_epochs', type=int, default=POST_EARLY_STOP_EPOCHS)
    parser.add_argument('--real_cifar', action='store_true')
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--no_hf', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    run_pretrained_experiment(
        case=args.case,
        sample_sizes=[args.sample_size],
        num_replications=args.reps,
        pretrain_epochs=args.pretrain_epochs,
        target_epochs=args.target_epochs,
        use_synthetic=not args.real_cifar,
        max_images=args.max_images,
        data_dir=args.data_dir,
        use_hf=not args.no_hf,
        early_stop_patience=args.patience,
        post_early_stop_epochs=args.post_epochs,
        output_dir=args.output_dir,
    )
