"""
Forward-Adding-Layer-Selection Algorithm from Di et al. (2025).

Implements:
  1. Generality Measure: omega_j = PC(Phi^j(X^t), Y^t) for convolutional layers j=1..J.
  2. Theoretical Phase Transition Location: j* = argmax_{1 <= j <= J} omega_j (Theorem 2).
  3. Algorithm 1: Forward Adding Selection of General Layers with Warm Start (k_0 = j*)
     and early stopping based on target validation loss.
  4. Model-free / Zero-shot layer selection (cut-off estimation without training target models).
"""

from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .projection_correlation import projection_correlation, permutation_test
    from .layer_extractor import extract_layer_activations
    from .models import CIFAR6CNN, create_transferred_model, set_trainable_layers
except ImportError:
    from projection_correlation import projection_correlation, permutation_test
    from layer_extractor import extract_layer_activations
    from models import CIFAR6CNN, create_transferred_model, set_trainable_layers


def compute_pc_profile(model: nn.Module,
                       dataloader: DataLoader,
                       device: Optional[Union[str, torch.device]] = None,
                       layer_names: Optional[List[str]] = None,
                       pooling: str = 'gap',
                       max_samples: Optional[int] = 500,
                       mode: str = 'auto',
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Step 1 of Algorithm 1: Compute generality measure omega_j = PC(Phi^j(X^t), Y^t)
    for all convolutional layers / blocks j = 1..J.

    Parameters
    ----------
    model : nn.Module
        Pre-trained feature extractor model.
    dataloader : DataLoader
        Target dataset loader (X^t, Y^t).
    device : torch.device, optional
    layer_names : list of str, optional
    pooling : {'gap', 'flatten', 'spatial_pool'}
    max_samples : int, default 500
        Number of target samples to use for PC estimation (e.g. 500, 1000, 1500 as in paper).
    mode : {'auto', 'exact', 'approximate'}
    verbose : bool

    Returns
    -------
    dict with:
      - 'omega': dict layer_name -> float (PC generality scores)
      - 'ordered_layers': list of layer names in forward order
      - 'argmax_layer': layer name with highest PC score (warm start k_0)
      - 'argmax_index': 1-based index of argmax layer
      - 'drops': dict layer_name -> float (Delta_j = omega_j^2 - omega_{j+1}^2)
      - 'n_samples': int
    """
    if verbose:
        print("[Di 2025] Extracting activations across layers on target data...")

    activations, labels = extract_layer_activations(
        model, dataloader, device=device,
        layer_names=layer_names, pooling=pooling,
        max_samples=max_samples
    )

    n_samples = labels.shape[0]
    ordered_layers = list(activations.keys())
    omega: Dict[str, float] = {}

    if verbose:
        print(f"[Di 2025] Computing Projection Correlation (n={n_samples}, J={len(ordered_layers)} layers)...")

    for name in ordered_layers:
        act = activations[name]
        pc = projection_correlation(act, labels, mode=mode, device=device)
        omega[name] = float(pc)
        if verbose:
            print(f"  Layer {name}: omega = {pc:.4f} (feature dim: {act.shape[1]})")

    # Compute gaps Delta_j = omega_j^2 - omega_{j+1}^2 (Assumption 1 in paper)
    drops: Dict[str, float] = {}
    for i in range(len(ordered_layers) - 1):
        l_curr = ordered_layers[i]
        l_next = ordered_layers[i + 1]
        delta = (omega[l_curr] ** 2) - (omega[l_next] ** 2)
        drops[l_curr] = float(delta)

    # Argmax layer: j* = argmax omega_j (Theorem 2)
    best_layer = max(omega, key=omega.get)
    best_idx = ordered_layers.index(best_layer) + 1  # 1-based index

    has_phase_transition = (best_idx < len(ordered_layers))

    return {
        'omega': omega,
        'ordered_layers': ordered_layers,
        'argmax_layer': best_layer,
        'argmax_index': best_idx,
        'has_phase_transition': has_phase_transition,
        'drops': drops,
        'n_samples': n_samples
    }


def estimate_phase_transition(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Theoretical cut-off estimation from Section 2 & Theorem 2 (Model-free).

    j* = argmax_{1 <= j <= J} omega_j
    When j* = J, no phase transition occurs (Balance case, all layers general).
    When j* < J, phase transition starts after j* (Imbalance case, layers > j* are specific).
    """
    omega = profile['omega']
    ordered = profile['ordered_layers']
    j_star_name = profile['argmax_layer']
    j_star_idx = profile['argmax_index']
    drops = profile['drops']
    has_pt = profile['has_phase_transition']

    return {
        'j_star_name': j_star_name,
        'j_star_index': j_star_idx,
        'total_layers': len(ordered),
        'has_phase_transition': has_pt,
        'general_layers': ordered[:j_star_idx],
        'specific_layers': ordered[j_star_idx:],
        'omega_values': [omega[l] for l in ordered],
        'drops': drops
    }


def default_train_epoch(model: nn.Module,
                        dataloader: DataLoader,
                        optimizer: torch.optim.Optimizer,
                        criterion: nn.Module,
                        device: torch.device) -> float:
    """Train model for 1 epoch and return average training loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / max(1, total_samples)


def default_eval_loss(model: nn.Module,
                      dataloader: DataLoader,
                      criterion: nn.Module,
                      device: torch.device) -> Tuple[float, float]:
    """Evaluate model and return (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def train_target_candidate(pretrained_model: CIFAR6CNN,
                           k: int,
                           num_target_classes: int,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           epochs: int = 15,
                           lr: float = 0.001,
                           freeze_transferred: bool = True,
                           device: Optional[torch.device] = None,
                           verbose: bool = True) -> Tuple[CIFAR6CNN, float, float]:
    """
    Train candidate model M_k on target dataset for `epochs` epochs.
    Optimizer: RMSProp(lr=0.001, alpha=0.9), following Di et al. (2025) Section 4.1.

    Returns
    -------
    model : CIFAR6CNN
    val_loss : float
    val_acc : float
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_transferred_model(
        pretrained_model=pretrained_model,
        k=k,
        num_target_classes=num_target_classes,
        freeze_transferred=freeze_transferred
    ).to(device)

    # Only optimize parameters that require gradient
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.RMSprop(trainable_params, lr=lr, alpha=0.9)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        tr_loss = default_train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = default_eval_loss(model, val_loader, criterion, device)

        if v_loss < best_loss:
            best_loss = v_loss
            best_acc = v_acc

        if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
            print(f"    [M_{k}] Epoch {ep:02d}/{epochs:02d} - Train Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")

    return model, best_loss, best_acc


def forward_adding_layer_selection(pretrained_model: CIFAR6CNN,
                                   target_train_loader: DataLoader,
                                   target_val_loader: DataLoader,
                                   num_target_classes: int,
                                   pc_sample_size: int = 500,
                                   train_epochs: int = 15,
                                   lr: float = 0.001,
                                   freeze_transferred: bool = True,
                                   device: Optional[torch.device] = None,
                                   mode: str = 'auto',
                                   verbose: bool = True) -> Dict[str, Any]:
    """
    Full implementation of Algorithm 1: Forward Adding Selection of General Layers.

    Input:
      Pre-trained model Phi with J layers,
      target dataset (X^t, Y^t),
      loss function e.

    1: Initialize:
    2:   Compute projection correlation omega_j = PC(Phi^j(X^t), Y^t) for j = 1..J
    3:   Set initial cut-off k_0 = argmax_j omega_j
    4:   Train initial model on target dataset using first k_0 layers, compute loss e_0
    5: Iterative Update:
    6: while True do
    7:   Increment number of layers: k = k_0 + 1
    8:   if k > J: break
    9:   Train updated model on target dataset using first k layers, compute loss e_1
    10:  if e_1 >= e_0 then
    11:    Stop and set final number of layers k = k_0; Break
    12:  else
    13:    Update k_0 = k and e_0 = e_1
    14:  end if
    15: end while
    Output:
      Optimized model for target task with selected layers k transferred.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    J = len(CIFAR6CNN.LAYER_NAMES)

    if verbose:
        print("=" * 65)
        print("Algorithm 1: Forward Adding Selection of General Layers (Di et al. 2025)")
        print("=" * 65)

    # 1-2. Compute PC profile
    start_time = time.time()
    profile = compute_pc_profile(
        pretrained_model, target_train_loader,
        device=device, max_samples=pc_sample_size,
        mode=mode, verbose=verbose
    )

    # 3. Warm start cut-off: k_0 = argmax_j omega_j
    k_0 = profile['argmax_index']
    argmax_layer_name = profile['argmax_layer']

    if verbose:
        print(f"\n[Warm Start] Initial cut-off from PC profile: k_0 = {k_0} ({argmax_layer_name})")
        print(f"[Step 4] Training initial candidate model M_{k_0} on target data...")

    model_0, e_0, acc_0 = train_target_candidate(
        pretrained_model, k=k_0,
        num_target_classes=num_target_classes,
        train_loader=target_train_loader,
        val_loader=target_val_loader,
        epochs=train_epochs, lr=lr,
        freeze_transferred=freeze_transferred,
        device=device, verbose=verbose
    )

    history = [{
        'k': k_0,
        'val_loss': e_0,
        'val_acc': acc_0,
        'status': 'initial (warm start)'
    }]

    best_model = model_0
    final_k = k_0
    final_loss = e_0
    final_acc = acc_0

    # 5-15. Iterative Forward-Adding Update
    while True:
        k = k_0 + 1
        if k > J:
            if verbose:
                print(f"[Stop] Reached maximum layer depth J={J}.")
            break

        if verbose:
            print(f"\n[Step 7-8] Forward-adding layer {k}: training candidate model M_{k}...")

        model_1, e_1, acc_1 = train_target_candidate(
            pretrained_model, k=k,
            num_target_classes=num_target_classes,
            train_loader=target_train_loader,
            val_loader=target_val_loader,
            epochs=train_epochs, lr=lr,
            freeze_transferred=freeze_transferred,
            device=device, verbose=verbose
        )

        history.append({
            'k': k,
            'val_loss': e_1,
            'val_acc': acc_1,
            'status': 'evaluated'
        })

        if e_1 >= e_0:
            if verbose:
                print(f"[Step 9-11] Loss increased or stopped decreasing: e_1 ({e_1:.4f}) >= e_0 ({e_0:.4f}).")
                print(f"  --> Stopping. Selected final cut-off k* = {k_0}.")
            history[-1]['status'] = 'rejected (loss did not decrease)'
            break
        else:
            if verbose:
                print(f"[Step 13] Loss improved: e_1 ({e_1:.4f}) < e_0 ({e_0:.4f}). Updating k_0 = {k}.")
            history[-1]['status'] = 'accepted'
            k_0 = k
            e_0 = e_1
            best_model = model_1
            final_k = k_0
            final_loss = e_0
            final_acc = acc_1

    total_time = time.time() - start_time

    if verbose:
        print("=" * 65)
        print(f"Selection complete in {total_time:.1f}s.")
        print(f"Selected General Layers (1..{final_k}): {CIFAR6CNN.LAYER_NAMES[:final_k]}")
        print(f"Discarded Specific Layers ({final_k + 1}..{J}): {CIFAR6CNN.LAYER_NAMES[final_k:]}")
        print(f"Final Validation Loss: {final_loss:.4f} | Accuracy: {final_acc:.4f}")
        print("=" * 65)

    return {
        'selected_k': final_k,
        'selected_layers': CIFAR6CNN.LAYER_NAMES[:final_k],
        'specific_layers': CIFAR6CNN.LAYER_NAMES[final_k:],
        'best_model': best_model,
        'final_val_loss': final_loss,
        'final_val_acc': final_acc,
        'final_acc': final_acc,
        'pc_profile': profile,
        'history': history,
        'total_time_seconds': total_time
    }
