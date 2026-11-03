"""
Layer Activation Extractor for Neural Networks.

Extracts intermediate representations Phi^j(X) across convolutional layers or blocks
using PyTorch forward hooks, following the formulation in Di et al. (2025), Section 2.
"""

from typing import Dict, List, Optional, Union, Callable, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LayerExtractor:
    """
    Hooks intermediate layers of a PyTorch model and captures activations Phi^j(X).

    Parameters
    ----------
    model : nn.Module
        The neural network.
    pooling : {'gap', 'flatten', 'spatial_pool'}
        Aggregation method for feature maps (Section 2 of Di et al. 2025):
          - 'gap': Global Average Pooling (N, C, H, W) -> (N, C). Default for deep nets.
          - 'flatten': Full vectorization (N, C, H, W) -> (N, C * H * W).
          - 'spatial_pool': Adaptive average pool to (pool_size, pool_size) then flatten.
    pool_size : int, default 2
        Spatial grid size when pooling='spatial_pool'.
    layer_names : list of str, optional
        Explicit module names to hook. If None, automatically hooks Conv2d or ConvBlock modules.
    """

    def __init__(self, model: nn.Module,
                 pooling: str = 'gap',
                 pool_size: int = 2,
                 layer_names: Optional[List[str]] = None):
        self.model = model
        self.pooling = pooling
        self.pool_size = pool_size
        self.layer_names = layer_names
        self.activations: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._gap = nn.AdaptiveAvgPool2d((1, 1))
        self._pool2d = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.register_hooks()

    def _pool_activation(self, act: torch.Tensor) -> torch.Tensor:
        """Apply selected feature map aggregation."""
        if act.dim() != 4:
            return act

        if self.pooling == 'gap':
            return self._gap(act).flatten(1)
        elif self.pooling == 'flatten':
            return act.flatten(1)
        elif self.pooling == 'spatial_pool':
            return self._pool2d(act).flatten(1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def _hook_fn(self, name: str) -> Callable:
        def hook(module, input, output):
            if isinstance(output, (tuple, list)):
                out = output[0]
            else:
                out = output
            pooled = self._pool_activation(out.detach())
            self.activations[name] = pooled
        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on target layers."""
        self.remove_hooks()
        named_modules = dict(self.model.named_modules())

        if self.layer_names is not None:
            for name in self.layer_names:
                if name in named_modules:
                    h = named_modules[name].register_forward_hook(self._hook_fn(name))
                    self._hooks.append(h)
                else:
                    raise KeyError(f"Layer '{name}' not found in model.")
        else:
            # Auto-detect: if model has 'layers' list (like CIFAR6CNN), hook each block
            if hasattr(self.model, 'layers') and isinstance(self.model.layers, (nn.ModuleList, list)):
                names = getattr(self.model, 'LAYER_NAMES', [f"layer_{i}" for i in range(1, len(self.model.layers) + 1)])
                for idx, layer in enumerate(self.model.layers):
                    name = names[idx]
                    h = layer.register_forward_hook(self._hook_fn(name))
                    self._hooks.append(h)
            else:
                # Hook leaf Conv2d modules
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        h = module.register_forward_hook(self._hook_fn(name))
                        self._hooks.append(h)

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


def extract_layer_activations(model: nn.Module,
                             dataloader: DataLoader,
                             device: Optional[Union[str, torch.device]] = None,
                             layer_names: Optional[List[str]] = None,
                             pooling: str = 'gap',
                             max_samples: Optional[int] = None,
                             keep_on_device: Optional[bool] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Run forward pass over dataloader and collect pooled activations for all layers.

    Parameters
    ----------
    model : nn.Module
    dataloader : DataLoader
    device : torch.device, optional
    layer_names : list of str, optional
    pooling : {'gap', 'flatten', 'spatial_pool'}
    max_samples : int, optional
        Maximum total samples to extract (e.g. 500, 1000, 1500 as in paper).
    keep_on_device : bool, optional
        If True (or auto if device is cuda and samples <= 2000), keeps activations in GPU memory
        to avoid PCIe transfer overhead when computing Projection Correlation.

    Returns
    -------
    activations : dict str -> torch.Tensor of shape (n_samples, feature_dim)
    labels : torch.Tensor of shape (n_samples,)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    if keep_on_device is None:
        keep_on_device = (device.type == 'cuda') and (max_samples is None or max_samples <= 2000)

    extractor = LayerExtractor(model, pooling=pooling, layer_names=layer_names)

    collected_acts: Dict[str, List[torch.Tensor]] = {}
    collected_labels: List[torch.Tensor] = []
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None

            inputs = inputs.to(device, non_blocking=True)
            extractor.activations.clear()
            _ = model(inputs)

            batch_n = inputs.shape[0]
            if max_samples is not None and (total_count + batch_n) > max_samples:
                take = max_samples - total_count
            else:
                take = batch_n

            for name, act in extractor.activations.items():
                tensor_to_store = act[:take] if keep_on_device else act[:take].cpu()
                collected_acts.setdefault(name, []).append(tensor_to_store)
            if targets is not None:
                labels_to_store = targets[:take].to(device) if keep_on_device else targets[:take].cpu()
                collected_labels.append(labels_to_store)

            total_count += take
            if max_samples is not None and total_count >= max_samples:
                break

    extractor.remove_hooks()

    out_acts = {name: torch.cat(tensors, dim=0) for name, tensors in collected_acts.items()}
    out_labels = torch.cat(collected_labels, dim=0) if collected_labels else torch.empty(0, device=device if keep_on_device else 'cpu')

    return out_acts, out_labels
