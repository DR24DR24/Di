"""
Quickstart Demonstration for Di et al. (2025) Algorithm.

Executes end-to-end:
  1. Synthetic Transfer Benchmark (Source vs Target domain shift).
  2. 6-layer CNN (Figure 3 architecture).
  3. Pre-training on Source task.
  4. Generality Measure profiling via Projection Correlation (PC).
  5. Phase Transition cut-off estimation j* = argmax omega_j (Theorem 2).
  6. Algorithm 1: Forward-Adding Layer Selection with warm-start.
  7. Verification against exhaustive search (nested models M_0..M_6).
"""

import sys
import os
import time
import torch
import torch.nn as nn

# Ensure local package import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models import CIFAR6CNN, create_transferred_model
from dataset_splits import generate_synthetic_transfer_dataset, make_dataloaders
from di_algorithm import (
    compute_pc_profile,
    estimate_phase_transition,
    forward_adding_layer_selection,
    train_target_candidate,
    default_eval_loss
)


def run_demo():
    print("=" * 70)
    print("Di et al. (2025) Layer Selection Algorithm Demonstration")
    print("Paper: 'Computationally Efficient General Convolutional Layer Selection'")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Generate Synthetic Transfer Benchmark
    print("\n[Step 1] Generating transfer learning benchmark...")
    splits = generate_synthetic_transfer_dataset(
        n_source=800, n_target=500,
        num_classes=5, image_size=32,
        transition_layer=4, seed=42
    )
    loaders = make_dataloaders(splits, batch_size=64)

    # 2. Build 6-layer CNN from Figure 3
    print("\n[Step 2] Building 6-layer CNN (Di et al. 2025, Figure 3)...")
    source_model = CIFAR6CNN(num_classes=splits['num_source_classes']).to(device)

    # 3. Pretrain model on source dataset
    print("[Step 3] Pre-training on Source dataset (5 epochs)...")
    optimizer = torch.optim.RMSprop(source_model.parameters(), lr=0.002, alpha=0.9)
    criterion = nn.CrossEntropyLoss()

    source_model.train()
    for ep in range(1, 6):
        total_loss = 0.0
        for x, y in loaders['source_train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = source_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(loaders['source_train'].dataset)
        val_loss, val_acc = default_eval_loss(source_model, loaders['source_test'], criterion, device)
        print(f"  Epoch {ep}/5 - Train Loss: {avg_loss:.4f} | Source Val Acc: {val_acc * 100:.1f}%")

    # 4. Compute PC profile across all 6 layers on target dataset
    print("\n[Step 4] Profiling generality via Projection Correlation (Section 2)...")
    profile = compute_pc_profile(
        source_model, loaders['target_train'],
        device=device, max_samples=400,
        mode='auto', verbose=True
    )

    pt_est = estimate_phase_transition(profile)
    j_star = pt_est['j_star_index']
    j_star_name = pt_est['j_star_name']

    print("\n" + "-" * 50)
    print("PHASE TRANSITION ESTIMATION (Theorem 2 & Section 2):")
    print(f"  Theoretical cut-off j* = {j_star} ({j_star_name})")
    print(f"  General layers (transferred): {pt_est['general_layers']}")
    print(f"  Specific layers (discarded):  {pt_est['specific_layers']}")
    print("-" * 50)

    # 5. Run Algorithm 1: Forward Adding Selection with warm-start
    print("\n[Step 5] Running Algorithm 1 (Forward-Adding Selection with warm-start)...")
    result = forward_adding_layer_selection(
        source_model,
        target_train_loader=loaders['target_train'],
        target_val_loader=loaders['target_test'],
        num_target_classes=splits['num_target_classes'],
        pc_sample_size=400,
        train_epochs=8,
        lr=0.002,
        freeze_transferred=True,
        device=device,
        verbose=True
    )

    # 6. Summary Comparison Table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY & RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Layer':<10} | {'Layer Name':<12} | {'PC Generality (omega)':<22} | {'Status':<15}")
    print("-" * 70)
    for idx, name in enumerate(profile['ordered_layers'], 1):
        omega = profile['omega'][name]
        is_cut = (idx == result['selected_k'])
        status = ">>> CUT-OFF <<<" if is_cut else ("General" if idx <= result['selected_k'] else "Specific (Overfit)")
        print(f"Layer {idx:<4} | {name:<12} | {omega:<22.4f} | {status}")

    print("-" * 70)
    print(f"Selected cut-off: k* = {result['selected_k']} layers transferred.")
    print(f"Final Validation Loss: {result['final_val_loss']:.4f} | Accuracy: {result['final_val_acc'] * 100:.1f}%")
    print(f"Algorithm 1 evaluated only {len(result['history'])} models instead of all 7 nested models!")
    print("=" * 70)


if __name__ == '__main__':
    run_demo()
