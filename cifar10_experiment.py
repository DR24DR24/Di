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


def _load_from_cache(
    cache_path: str,
    max_images: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read existing prepared .pt cache (fastest option)."""
    print(f"[CIFAR-10] Loading cache from {cache_path} ...")
    data_tensor, labels_tensor = torch.load(cache_path, weights_only=True)
    if max_images is not None:
        data_tensor = data_tensor[:max_images]
        labels_tensor = labels_tensor[:max_images]
    print(f"[CIFAR-10] From cache: {data_tensor.shape[0]} images.")
    return data_tensor, labels_tensor


def _load_from_local_archive(
    cifar_dir: str,
    max_images: Optional[int] = None,
    cache_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read from already extracted cifar-10-batches-py folder (no network)."""
    import pickle
    import numpy as np

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

    if max_images is not None:
        data_tensor = data_tensor[:max_images]
        labels_tensor = labels_tensor[:max_images]

    if cache_path:
        torch.save((data_tensor, labels_tensor), cache_path)
        print(f"[CIFAR-10] Cache saved: {cache_path}")

    print(f"[CIFAR-10] Loaded {data_tensor.shape[0]} images from local archive.")
    return data_tensor, labels_tensor


def _load_from_hf(
    max_images: Optional[int] = None,
    cache_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    New option: load via HuggingFace datasets (uoft-cs/cifar10).
    No need to access cs.toronto.edu, downloads from HF.
    """
    from datasets import load_dataset

    print("[CIFAR-10] Loading via HuggingFace datasets (uoft-cs/cifar10) ...")
    train_ds = load_dataset("uoft-cs/cifar10", split="train")
    test_ds = load_dataset("uoft-cs/cifar10", split="test")

    def _ds_to_tensors(ds) -> Tuple[torch.Tensor, torch.Tensor]:
        imgs = []
        labels = []
        for example in ds:
            img = example["img"]  # PIL Image (32x32x3)
            t = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
            imgs.append(t)
            labels.append(example["label"])
        x = torch.stack(imgs)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    x_tr, y_tr = _ds_to_tensors(train_ds)
    x_te, y_te = _ds_to_tensors(test_ds)

    data_tensor = torch.cat([x_tr, x_te], dim=0)
    labels_tensor = torch.cat([y_tr, y_te], dim=0)

    # Same normalization as other options
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    data_tensor = (data_tensor - mean) / std

    if max_images is not None:
        data_tensor = data_tensor[:max_images]
        labels_tensor = labels_tensor[:max_images]

    if cache_path:
        torch.save((data_tensor, labels_tensor), cache_path)
        print(f"[CIFAR-10] HF cache saved: {cache_path}")

    print(f"[CIFAR-10] Loaded {data_tensor.shape[0]} images via HF.")
    return data_tensor, labels_tensor


def load_cifar10_raw(
    data_dir: str = './data',
    max_images: Optional[int] = None,
    use_hf: bool = True,
    cache_name: str = 'cifar10_cached.pt'
) -> Tuple[torch.Tensor, torch.Tensor]:
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, cache_name)
    archive_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    cifar_dir = os.path.join(data_dir, 'cifar-10-batches-py')

    # 1. Cache (fastest)
    if os.path.exists(cache_path):
        return _load_from_cache(cache_path, max_images=max_images)

    # 2. Local archive (already downloaded, no network)
    if os.path.exists(archive_path):
        if not os.path.exists(cifar_dir):
            import tarfile
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=data_dir)
        return _load_from_local_archive(cifar_dir, max_images=max_images, cache_path=cache_path)

    # 3. New option: HuggingFace
    if use_hf:
        try:
            return _load_from_hf(max_images=max_images, cache_path=cache_path)
        except Exception as exc:
            print(f"[CIFAR-10] HF unavailable ({exc}), falling back to legacy option...")

    # 4. Legacy option (torchvision / native download)
    return _load_legacy(data_dir, max_images=max_images, cache_path=cache_path)


def _load_legacy(
    data_dir: str,
    max_images: Optional[int] = None,
    cache_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Legacy option: torchvision.datasets.CIFAR10(download=True)
    or native download of cifar-10-python.tar.gz from cs.toronto.edu.
    """
    # --- Branch A: torchvision ---
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
        data_tensor = torch.cat([x_tr, x_te], dim=0)
        labels_tensor = torch.cat([y_tr, y_te], dim=0)
        if max_images is not None:
            data_tensor = data_tensor[:max_images]
            labels_tensor = labels_tensor[:max_images]
        if cache_path:
            torch.save((data_tensor, labels_tensor), cache_path)
        print(f"[CIFAR-10] (legacy/torchvision) Loaded {data_tensor.shape[0]} images.")
        return data_tensor, labels_tensor
    except Exception:
        pass

    # --- Branch B: native download of cifar-10-python.tar.gz ---
    import urllib.request
    import tarfile

    archive_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    cifar_dir = os.path.join(data_dir, 'cifar-10-batches-py')

    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(cifar_dir):
        if not os.path.exists(archive_path):
            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            print(f"[CIFAR-10] (legacy) Downloading from {url} ...")
            urllib.request.urlretrieve(url, archive_path)
            print("[CIFAR-10] (legacy) Download completed.")
        print(f"[CIFAR-10] (legacy) Extracting {archive_path} ...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("[CIFAR-10] (legacy) Extraction completed.")

    return _load_from_local_archive(cifar_dir, max_images=max_images, cache_path=cache_path)


def run_experiment(case: str = 'imbalance',
                   sample_sizes: Tuple[int, ...] = (500, 1000),
                   num_replications: int = 2,
                   pretrain_epochs: int = 150,
                   target_epochs: int = 150,
                   use_synthetic: bool = True,
                   device: Optional[torch.device] = None,
                   output_dir: Optional[str] = None,
                   max_images: Optional[int] = None,
                   data_dir: str = './data',
                   use_hf: bool = True):
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
    max_images : int, optional (limit the number of CIFAR-10 images)
    data_dir : str (folder with cifar-10-python.tar.gz or cache)
    use_hf : bool (use HuggingFace datasets as a network source)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print(f"Di et al. (2025) CIFAR-10 Experiment: {case.upper()} CASE")
    print(f"Device: {device} | Sample sizes: {sample_sizes} | Replications: {num_replications}")
    print("=" * 70)

    # 1. Prepare Datasets
    if use_synthetic:
        splits = generate_synthetic_transfer_dataset(
            n_source=1200, n_target=800,
            num_classes=5 if case == 'imbalance' else 10,
            transition_layer=3 if case == 'imbalance' else 4, seed=42
        )
    else:
        raw_x, raw_y = load_cifar10_raw(
            data_dir=data_dir,
            max_images=max_images,
            use_hf=use_hf
        )
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
    summary_path = os.path.join(output_dir, f"colab_{case}_replication.json")
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
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--real_cifar', action='store_true', help="Use real CIFAR-10 if available")
    parser.add_argument('--max_images', type=int, default=None, help="Limit number of CIFAR-10 images (debug)")
    parser.add_argument('--data_dir', type=str, default='./data', help="Folder with cifar-10-python.tar.gz or cache")
    parser.add_argument('--no_hf', action='store_true', help="Do not use HuggingFace datasets (legacy option)")
    args = parser.parse_args()

    run_experiment(
        case=args.case,
        sample_sizes=(args.sample_size,),
        num_replications=args.reps,
        pretrain_epochs=args.epochs,
        target_epochs=args.epochs,
        use_synthetic=not args.real_cifar,
        max_images=args.max_images,
        data_dir=args.data_dir,
        use_hf=not args.no_hf
    )
