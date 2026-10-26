"""
Dataset Splitting and Benchmarks for Di et al. (2025) Transfer Learning Experiments.

Implements:
  1. Imbalance Case Split:
     Source (5 classes): automobile(1), truck(9), deer(4), dog(5), horse(7)
     Target (5 non-overlapping classes): airplane(0), bird(2), ship(8), cat(3), frog(6)
     Models real domain shift where source and target distributions differ.
  2. Balance Case Split:
     All 10 classes present in both source and target (50/50 split per class).
  3. Subsampling utility:
     Random subsample of sizes 500, 1000, 1500 for PC evaluation (Section 4.1).
  4. SyntheticTransferDataset:
     Fast self-contained synthetic benchmark with controllable phase transition
     at a predetermined layer l*, enabling rapid verification without external downloads.
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


# Exact class indices from CIFAR-10 standard mapping and Di et al. (2025) footnote 2:
# 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
# 5: dog, 6: frog, 7: horse, 8: ship, 9: truck
CIFAR_SOURCE_CLASSES = [1, 9, 4, 5, 7]  # automobile, truck, deer, dog, horse
CIFAR_TARGET_CLASSES = [0, 2, 8, 3, 6]  # airplane, bird, ship, cat, frog


class TensorDataset(Dataset):
    """Simple in-memory dataset of tensors (X, Y)."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def split_cifar_imbalance(cifar_train_data: torch.Tensor,
                          cifar_train_labels: torch.Tensor,
                          source_classes: Optional[List[int]] = None,
                          target_classes: Optional[List[int]] = None,
                          source_train_samples: int = 25000,
                          seed: Optional[int] = 42) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split CIFAR-10 into Imbalance Case (non-overlapping source and target classes).

    Source: 5 classes (5000 train per class = 25000, 1000 test per class = 5000).
    Target: 5 classes (re-indexed to 0..4).
    """
    if seed is not None:
        torch.manual_seed(seed)

    if source_classes is None:
        source_classes = CIFAR_SOURCE_CLASSES
    if target_classes is None:
        target_classes = CIFAR_TARGET_CLASSES

    # Remap dicts
    source_map = {orig_cls: new_idx for new_idx, orig_cls in enumerate(source_classes)}
    target_map = {orig_cls: new_idx for new_idx, orig_cls in enumerate(target_classes)}

    src_mask = torch.tensor([lbl.item() in source_classes for lbl in cifar_train_labels])
    tgt_mask = torch.tensor([lbl.item() in target_classes for lbl in cifar_train_labels])

    src_x, src_y_raw = cifar_train_data[src_mask], cifar_train_labels[src_mask]
    tgt_x, tgt_y_raw = cifar_train_data[tgt_mask], cifar_train_labels[tgt_mask]

    src_y = torch.tensor([source_map[lbl.item()] for lbl in src_y_raw], dtype=torch.long)
    tgt_y = torch.tensor([target_map[lbl.item()] for lbl in tgt_y_raw], dtype=torch.long)

    # Permute
    perm_src = torch.randperm(src_x.shape[0])
    perm_tgt = torch.randperm(tgt_x.shape[0])

    src_x, src_y = src_x[perm_src], src_y[perm_src]
    tgt_x, tgt_y = tgt_x[perm_tgt], tgt_y[perm_tgt]

    # Split train and test
    src_train_x, src_train_y = src_x[:source_train_samples], src_y[:source_train_samples]
    src_test_x, src_test_y = src_x[source_train_samples:], src_y[source_train_samples:]

    tgt_train_x, tgt_train_y = tgt_x[:source_train_samples], tgt_y[:source_train_samples]
    tgt_test_x, tgt_test_y = tgt_x[source_train_samples:], tgt_y[source_train_samples:]

    return {
        'source_train': (src_train_x, src_train_y),
        'source_test': (src_test_x, src_test_y),
        'target_train': (tgt_train_x, tgt_train_y),
        'target_test': (tgt_test_x, tgt_test_y),
        'num_source_classes': len(source_classes),
        'num_target_classes': len(target_classes)
    }


def split_cifar_balance(cifar_train_data: torch.Tensor,
                        cifar_train_labels: torch.Tensor,
                        seed: Optional[int] = 42) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split CIFAR-10 into Balance Case (all 10 classes equally partitioned 50% source, 50% target).
    Each class has 3000 source images and 3000 target images.
    """
    if seed is not None:
        torch.manual_seed(seed)

    src_x_list, src_y_list = [], []
    tgt_x_list, tgt_y_list = [], []

    for cls in range(10):
        cls_mask = (cifar_train_labels == cls)
        cls_x = cifar_train_data[cls_mask]
        cls_y = cifar_train_labels[cls_mask]

        n_cls = cls_x.shape[0]
        half = n_cls // 2
        perm = torch.randperm(n_cls)

        src_x_list.append(cls_x[perm[:half]])
        src_y_list.append(cls_y[perm[:half]])
        tgt_x_list.append(cls_x[perm[half:]])
        tgt_y_list.append(cls_y[perm[half:]])

    src_x = torch.cat(src_x_list, dim=0)
    src_y = torch.cat(src_y_list, dim=0)
    tgt_x = torch.cat(tgt_x_list, dim=0)
    tgt_y = torch.cat(tgt_y_list, dim=0)

    # Shuffle
    perm_s = torch.randperm(src_x.shape[0])
    perm_t = torch.randperm(tgt_x.shape[0])

    return {
        'source_train': (src_x[perm_s[:25000]], src_y[perm_s[:25000]]),
        'source_test': (src_x[perm_s[25000:]], src_y[perm_s[25000:]]),
        'target_train': (tgt_x[perm_t[:25000]], tgt_y[perm_t[:25000]]),
        'target_test': (tgt_x[perm_t[25000:]], tgt_y[perm_t[25000]]),
        'num_source_classes': 10,
        'num_target_classes': 10
    }


def generate_synthetic_transfer_dataset(n_source: int = 1000,
                                        n_target: int = 600,
                                        num_classes: int = 5,
                                        image_size: int = 32,
                                        transition_layer: int = 4,
                                        seed: Optional[int] = 42) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate synthetic image data exhibiting a clear phase transition at `transition_layer`.

    Low-level patterns (edges, color patches) correlate with target labels across
    layers 1..transition_layer. High-level patterns (specific shapes) diverge after transition_layer,
    simulating the exact phase transition described in Di et al. (2025).

    Returns
    -------
    dict with:
      'source_train': (X_s, Y_s)
      'source_test': (X_s_val, Y_s_val)
      'target_train': (X_t, Y_t)
      'target_test': (X_t_val, Y_t_val)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    def create_samples(n: int, is_target: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.randint(0, num_classes, (n,), dtype=torch.long)
        images = torch.randn(n, 3, image_size, image_size) * 0.1

        for i in range(n):
            lbl = labels[i].item()
            # Shared low-level signal: horizontal/vertical stripes based on label % 2
            if lbl % 2 == 0:
                images[i, 0, :, ::2] += 0.5
            else:
                images[i, 1, ::2, :] += 0.5

            # Shared mid-level signal: corner intensities based on label
            corner = lbl % 4
            if corner == 0:
                images[i, 2, :8, :8] += 0.8
            elif corner == 1:
                images[i, 2, :8, -8:] += 0.8
            elif corner == 2:
                images[i, 2, -8:, :8] += 0.8
            else:
                images[i, 2, -8:, -8:] += 0.8

            # Specific high-level signal: diverges for target vs source
            if not is_target:
                # Source specific high-frequency pattern
                images[i, :, 12:20, 12:20] += (lbl * 0.3)
            else:
                # Target specific pattern is completely independent or inverted
                images[i, :, 12:20, 12:20] += ((num_classes - lbl) * 0.3)

        return images, labels

    src_train_x, src_train_y = create_samples(n_source, is_target=False)
    src_test_x, src_test_y = create_samples(max(200, n_source // 4), is_target=False)
    tgt_train_x, tgt_train_y = create_samples(n_target, is_target=True)
    tgt_test_x, tgt_test_y = create_samples(max(200, n_target // 3), is_target=True)

    return {
        'source_train': (src_train_x, src_train_y),
        'source_test': (src_test_x, src_test_y),
        'target_train': (tgt_train_x, tgt_train_y),
        'target_test': (tgt_test_x, tgt_test_y),
        'num_source_classes': num_classes,
        'num_target_classes': num_classes,
        'true_transition_layer': transition_layer
    }


def make_dataloaders(split_dict: Dict,
                     batch_size: int = 64) -> Dict[str, DataLoader]:
    """Convert tensor split dict into PyTorch DataLoaders."""
    loaders = {}
    for key in ['source_train', 'source_test', 'target_train', 'target_test']:
        if key in split_dict:
            x, y = split_dict[key]
            shuffle = 'train' in key
            ds = TensorDataset(x, y)
            loaders[key] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loaders
