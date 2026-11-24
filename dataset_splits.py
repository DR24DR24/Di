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

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch
from sklearn.model_selection import train_test_split
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
                          source_train_samples: int = 20000,
                          target_train_samples: Optional[int] = None,
                          seed: Optional[int] = 42) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split CIFAR-10 into Imbalance Case (non-overlapping source and target classes).

    Source: 5 classes (4000 train per class = 20000, 1000 test per class = 5000).
    Target: 5 classes (re-indexed to 0..4).
    """
    if target_train_samples is None:
        target_train_samples = source_train_samples

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

    # Стратифицированное разбиение для Source
    src_train_x, src_test_x, src_train_y, src_test_y = train_test_split(
        src_x.numpy(), src_y.numpy(),
        train_size=source_train_samples,
        stratify=src_y.numpy(),
        random_state=seed,
    )
    src_train_x = torch.from_numpy(src_train_x)
    src_test_x = torch.from_numpy(src_test_x)
    src_train_y = torch.from_numpy(src_train_y).long()
    src_test_y = torch.from_numpy(src_test_y).long()

    # Стратифицированное разбиение для Target
    tgt_train_x, tgt_test_x, tgt_train_y, tgt_test_y = train_test_split(
        tgt_x.numpy(), tgt_y.numpy(),
        train_size=target_train_samples,
        stratify=tgt_y.numpy(),
        random_state=seed,
    )
    tgt_train_x = torch.from_numpy(tgt_train_x)
    tgt_test_x = torch.from_numpy(tgt_test_x)
    tgt_train_y = torch.from_numpy(tgt_train_y).long()
    tgt_test_y = torch.from_numpy(tgt_test_y).long()

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
                     batch_size: int = 64,
                     generator: Optional[torch.Generator] = None) -> Dict[str, DataLoader]:
    """Convert tensor split dict into PyTorch DataLoaders with GPU pin_memory support."""
    loaders = {}
    use_pin = torch.cuda.is_available()
    for key in ['source_train', 'source_test', 'target_train', 'target_test']:
        if key in split_dict:
            x, y = split_dict[key]
            shuffle = 'train' in key
            ds = TensorDataset(x, y)
            loaders[key] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=generator,
                pin_memory=use_pin
            )
    return loaders


from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_ft_dataloaders_from_subsets(
    fixed_target_subsets: Dict[int, List[Dict[str, Any]]],
    target_train_x: torch.Tensor,
    target_train_y: torch.Tensor,
    sample_size: int,
    ft_sample_size: Optional[int] = None,
    val_ratio: float = 0.2,
    batch_size: int = 64,
    val_batch_size: Optional[int] = None,
    seed: Optional[int] = 42,
    generator: Optional[torch.Generator] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Создаёт train и val DataLoader для дообучения (Fine-Tuning), используя в первую очередь
    все уникальные данные из репликаций фиксированных подвыборок для PC и дополняя
    стратифицированно из остатка target_train при необходимости.

    Итоговая выборка размера ft_sample_size стратифицированно делится на train и val.
    Дубликаты между репликациями исключаются по индексам.

    Параметры
    ---------
    fixed_target_subsets : dict
        Результат select_fixed_subsets: {sample_size: [rep0, rep1, ...]}.
    target_train_x, target_train_y : torch.Tensor
        Полный target_train (все сэмплы для возможного дополнения).
    sample_size : int
        Размер подвыборок для PC (ключ в fixed_target_subsets).
    ft_sample_size : int, optional
        Общий размер выборки для дообучения (train + val). Если None -> sample_size.
    val_ratio : float
        Доля валидационной выборки от итоговых ft_sample_size сэмплов (по умолчанию 0.2).
    batch_size : int
        Размер батча для train DataLoader.
    val_batch_size : int, optional
        Размер батча для val DataLoader. Если None -> batch_size.
    seed : int, optional
        Сид для воспроизводимости стратифицированных разбиений.
    generator : torch.Generator, optional
        Генератор для перемешивания батчей в train DataLoader.

    Возвращает
    --------
    Tuple[DataLoader, DataLoader]
        (train_loader, val_loader) со стратифицированным разбиением.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio должен быть в интервале (0, 1), получено {val_ratio}")

    if ft_sample_size is None:
        ft_sample_size = sample_size
    if val_batch_size is None:
        val_batch_size = batch_size

    rep_list = fixed_target_subsets[sample_size]

    # 1. Собрать все данные и индексы из всех репликаций
    x_all = []
    y_all = []
    indices_all = []
    for item in rep_list:
        x_all.append(item["x"])
        y_all.append(item["y"])
        indices_all.append(item["indices"])

    x_concat = torch.cat(x_all, dim=0)
    y_concat = torch.cat(y_all, dim=0)
    indices_concat = torch.cat(indices_all, dim=0)

    # 2. Дедупликация по индексам (оставляем первое вхождение)
    seen = set()
    unique_mask = torch.zeros(len(indices_concat), dtype=torch.bool)
    for i, idx in enumerate(indices_concat.tolist()):
        if idx not in seen:
            seen.add(idx)
            unique_mask[i] = True

    x_rep_unique = x_concat[unique_mask]
    y_rep_unique = y_concat[unique_mask]
    used_indices = seen

    # 3. Маска оставшихся сэмплов в полном target_train
    n_total = target_train_x.shape[0]
    remaining_mask = torch.ones(n_total, dtype=torch.bool)
    for idx in used_indices:
        remaining_mask[idx] = False

    # 4. Формирование итоговой выборки размера ft_sample_size
    x_rep_np = x_rep_unique.cpu().numpy()
    y_rep_np = y_rep_unique.cpu().numpy()

    if ft_sample_size <= len(x_rep_np):
        if ft_sample_size < len(x_rep_np):
            x_ft, _, y_ft, _ = train_test_split(
                x_rep_np,
                y_rep_np,
                train_size=ft_sample_size,
                stratify=y_rep_np,
                random_state=seed,
            )
        else:
            x_ft, y_ft = x_rep_np, y_rep_np
    else:
        needed = ft_sample_size - len(x_rep_np)
        x_rem = target_train_x[remaining_mask].cpu().numpy()
        y_rem = target_train_y[remaining_mask].cpu().numpy()

        if needed < len(x_rem):
            x_add, _, y_add, _ = train_test_split(
                x_rem,
                y_rem,
                train_size=needed,
                stratify=y_rem,
                random_state=seed,
            )
        elif needed == len(x_rem):
            x_add, y_add = x_rem, y_rem
        else:
            raise ValueError(
                f"Недостаточно данных в target_train для добора {needed} сэмплов "
                f"(доступно в остатке: {len(x_rem)})."
            )

        x_ft = np.concatenate([x_rep_np, x_add], axis=0)
        y_ft = np.concatenate([y_rep_np, y_add], axis=0)

    # 5. Стратифицированное разбиение на train и val
    x_train, x_val, y_train, y_val = train_test_split(
        x_ft,
        y_ft,
        test_size=val_ratio,
        stratify=y_ft,
        random_state=seed,
    )

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(y_val).long(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
