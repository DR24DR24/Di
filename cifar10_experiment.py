"""
Replication of Section 4.1: CIFAR-10 Experiment from Di et al. (2025).

Recreates:
  - Imbalance Case (5 source classes vs 5 target classes, non-overlapping)
  - Balance Case (10 classes, 50% source, 50% target)
  - 6-layer CNN pretraining on Source dataset (25,000 images)
  - Generality measure profiling across layers j=1..6 with sample sizes n in {500, 1000, 1500}
  - Comparison of nested models M_0..M_6 vs Algorithm 1 (Forward-Adding Selection)
"""

import argparse
import sys
import os
import json
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .models import CIFAR6CNN, create_transferred_model
    from .projection_correlation import projection_correlation
    from .layer_extractor import extract_layer_activations
    from .dataset_splits import (
        split_cifar_imbalance,
        split_cifar_balance,
        generate_synthetic_transfer_dataset,
        make_dataloaders,
        CIFAR_SOURCE_CLASSES,
        CIFAR_TARGET_CLASSES
    )
    from .di_algorithm import (
        compute_pc_profile,
        estimate_phase_transition,
        forward_adding_layer_selection,
        train_target_candidate,
        default_eval_loss
    )
except ImportError:
    from models import CIFAR6CNN, create_transferred_model
    from projection_correlation import projection_correlation
    from layer_extractor import extract_layer_activations
    from dataset_splits import (
        split_cifar_imbalance,
        split_cifar_balance,
        generate_synthetic_transfer_dataset,
        make_dataloaders,
        CIFAR_SOURCE_CLASSES,
        CIFAR_TARGET_CLASSES
    )
    from di_algorithm import (
        compute_pc_profile,
        estimate_phase_transition,
        forward_adding_layer_selection,
        train_target_candidate,
        default_eval_loss
    )


def load_cifar10_raw(data_dir: str = './data') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load full CIFAR-10 dataset (60,000 images, 10 classes) as normalized tensors.
    Supports:
      1. torchvision.datasets.CIFAR10 if available.
      2. Direct native download and extraction from official source without torchvision.
    """
    # 1. Try torchvision if available
    try:
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        cifar_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        cifar_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        loader_tr = DataLoader(cifar_train, batch_size=len(cifar_train), shuffle=False)
        loader_te = DataLoader(cifar_test, batch_size=len(cifar_test), shuffle=False)
        x_tr, y_tr = next(iter(loader_tr))
        x_te, y_te = next(iter(loader_te))
        return torch.cat([x_tr, x_te], dim=0), torch.cat([y_tr, y_te], dim=0)
    except Exception:
        pass

    # 2. Native download and loading from cifar-10-python.tar.gz
    import urllib.request
    import tarfile
    import pickle

    os.makedirs(data_dir, exist_ok=True)
    cifar_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    archive_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')

    if not os.path.exists(cifar_dir):
        if not os.path.exists(archive_path):
            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            print(f"[CIFAR-10] Downloading dataset from {url}...")
            urllib.request.urlretrieve(url, archive_path)
            print("[CIFAR-10] Download completed.")
        print(f"[CIFAR-10] Extracting {archive_path}...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("[CIFAR-10] Extraction completed.")

    # Load 5 train batches + 1 test batch (60,000 images)
    batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    all_data = []
    all_labels = []

    for bf in batch_files:
        fp = os.path.join(cifar_dir, bf)
        with open(fp, 'rb') as f:
            entry = pickle.load(f, encoding='bytes')
            all_data.append(entry[b'data'])
            all_labels.extend(entry[b'labels'])

    data_np = np.concatenate(all_data, axis=0)
    data_np = data_np.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    data_np = (data_np - mean) / std

    data_tensor = torch.from_numpy(data_np)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    print(f"[CIFAR-10] Successfully loaded {data_tensor.shape[0]} images, shape {data_tensor.shape}.")
    return data_tensor, labels_tensor


def run_experiment(case: str = 'imbalance',
                   sample_sizes: Tuple[int, ...] = (500,),
                   num_replications: int = 3,
                   pretrain_epochs: int = 10,
                   target_epochs: int = 10,
                   use_synthetic: bool = True,
                   device: Optional[torch.device] = None,
                   output_dir: Optional[str] = None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    """
    Run full experiment matching Di et al. (2025) Section 4.1.

    Parameters
    ----------
    case : {'imbalance', 'balance'}
    sample_sizes : tuple of ints (e.g. 500, 1000, 1500)
    num_replications : int (number of independent PC evaluation runs, paper used 20)
    pretrain_epochs : int (number of epochs to pretrain source CNN)
    target_epochs : int (epochs to train nested models M_j)
    use_synthetic : bool (fast self-contained synthetic benchmark vs real CIFAR)
    output_dir : str (where to save JSON logs)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print(f"Di et al. (2025) CIFAR-10 Experiment: {case.upper()} CASE")
    print(f"Device: {device} | Sample sizes: {sample_sizes} | Replications: {num_replications}")
    print("=" * 70)

    # 1. Prepare Datasets
    if use_synthetic:
        # Imbalance has transition at Layer 4; Balance has transition at Layer 6 (all general)
        trans_layer = 4 if case == 'imbalance' else 6
        splits = generate_synthetic_transfer_dataset(
            n_source=1500, n_target=1000,
            num_classes=5 if case == 'imbalance' else 10,
            transition_layer=trans_layer, seed=42
        )
    else:
        raw_x, raw_y = load_cifar10_raw()
        if case == 'imbalance':
            splits = split_cifar_imbalance(raw_x, raw_y)
        else:
            splits = split_cifar_balance(raw_x, raw_y)

    loaders = make_dataloaders(splits, batch_size=64)

    # 2. Pre-train 6-layer CNN on Source dataset
    print(f"\n[Step 1] Pre-training Source 6-layer CNN ({pretrain_epochs} epochs)...")
    source_model = CIFAR6CNN(num_classes=splits['num_source_classes']).to(device)
    optimizer = torch.optim.RMSprop(source_model.parameters(), lr=0.001, alpha=0.9)
    criterion = nn.CrossEntropyLoss()

    source_model.train()
    for ep in range(1, pretrain_epochs + 1):
        tr_loss = 0.0
        for x, y in loaders['source_train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = source_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
        avg_loss = tr_loss / len(loaders['source_train'].dataset)
        v_loss, v_acc = default_eval_loss(source_model, loaders['source_test'], criterion, device)
        if ep == 1 or ep % 5 == 0 or ep == pretrain_epochs:
            print(f"  Pretrain Epoch {ep:02d}/{pretrain_epochs:02d} - Loss: {avg_loss:.4f} | Source Test Acc: {v_acc * 100:.1f}%")

    # 3. Compute PC Profile over multiple sample sizes and replications (Figure 4A/B)
    print("\n[Step 2] Computing PC generality profiles (Figures 4A, 4B)...")
    pc_results: Dict[int, Dict[str, List[float]]] = {}

    for n_s in sample_sizes:
        print(f"\n--- Evaluating PC with sample size n = {n_s} across {num_replications} reps ---")
        pc_results[n_s] = {layer: [] for layer in CIFAR6CNN.LAYER_NAMES}

        for rep in range(num_replications):
            prof = compute_pc_profile(
                source_model, loaders['target_train'],
                device=device, max_samples=n_s,
                verbose=False
            )
            for layer, val in prof['omega'].items():
                pc_results[n_s][layer].append(val)

        # Print mean and std
        for layer in CIFAR6CNN.LAYER_NAMES:
            vals = pc_results[n_s][layer]
            print(f"  {layer:<8}: mean = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Theoretical cut-off based on PC profile
    last_n = sample_sizes[-1]
    mean_profile = {l: float(np.mean(pc_results[last_n][l])) for l in CIFAR6CNN.LAYER_NAMES}
    best_layer = max(mean_profile, key=mean_profile.get)
    k_star_theory = CIFAR6CNN.LAYER_NAMES.index(best_layer) + 1

    print(f"\nTheoretical Cut-off (argmax omega_j): k* = {k_star_theory} ({best_layer})")

    # 4. Run Algorithm 1 (Forward Adding Selection)
    print("\n[Step 3] Running Algorithm 1 (Forward-Adding Selection with warm-start)...")
    t0 = time.time()
    algo_result = forward_adding_layer_selection(
        source_model,
        target_train_loader=loaders['target_train'],
        target_val_loader=loaders['target_test'],
        num_target_classes=splits['num_target_classes'],
        pc_sample_size=sample_sizes[0],
        train_epochs=target_epochs,
        lr=0.001,
        freeze_transferred=True,
        device=device,
        verbose=True
    )
    algo_time = time.time() - t0

    # 5. Run Exhaustive Search over nested models M_0..M_6 for comparison (Figure 4C/D)
    print("\n[Step 4] Running Exhaustive Search over nested models M_0..M_6 for validation...")
    exhaustive_results: List[Dict] = []
    t0_ex = time.time()

    for k in range(0, 7):
        print(f"  Training model M_{k} (transferred {k} layers)...")
        _, v_loss, v_acc = train_target_candidate(
            source_model, k=k,
            num_target_classes=splits['num_target_classes'],
            train_loader=loaders['target_train'],
            val_loader=loaders['target_test'],
            epochs=target_epochs, lr=0.001,
            freeze_transferred=True,
            device=device, verbose=False
        )
        exhaustive_results.append({
            'k': k,
            'val_loss': v_loss,
            'val_acc': v_acc
        })
        print(f"    M_{k}: Val Loss = {v_loss:.4f} | Val Acc = {v_acc * 100:.1f}%")

    exhaustive_time = time.time() - t0_ex

    best_exhaustive = min(exhaustive_results, key=lambda x: x['val_loss'])
    print("\n" + "=" * 70)
    print("COMPARISON: Algorithm 1 vs Exhaustive Search")
    print("=" * 70)
    print(f"Algorithm 1 Selected Cut-off:     k* = {algo_result['selected_k']} (Time: {algo_time:.1f}s)")
    print(f"Exhaustive Search Best Model:     k* = {best_exhaustive['k']} (Time: {exhaustive_time:.1f}s)")
    print(f"Theoretical argmax(omega_j):      k* = {k_star_theory}")
    print(f"Speedup Factor:                   {exhaustive_time / max(0.001, algo_time):.1f}x faster!")
    print("=" * 70)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"di_2025_{case}_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'case': case,
            'sample_sizes': sample_sizes,
            'mean_pc_profile': mean_profile,
            'k_star_theory': k_star_theory,
            'algo_selected_k': algo_result['selected_k'],
            'exhaustive_best_k': best_exhaustive['k'],
            'exhaustive_results': exhaustive_results,
            'algo_time_seconds': algo_time,
            'exhaustive_time_seconds': exhaustive_time
        }, f, indent=2)
    print(f"Results saved to: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replicate Di et al. (2025) CIFAR-10 experiments.")
    parser.add_argument('--case', choices=['imbalance', 'balance'], default='imbalance')
    parser.add_argument('--sample_size', type=int, default=400)
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--real_cifar', action='store_true', help="Use real CIFAR-10 if available")
    args = parser.parse_args()

    run_experiment(
        case=args.case,
        sample_sizes=(args.sample_size,),
        num_replications=args.reps,
        pretrain_epochs=args.epochs,
        target_epochs=args.epochs,
        use_synthetic=not args.real_cifar
    )
