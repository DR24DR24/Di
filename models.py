"""
Neural Network Architectures for Di et al. (2025) Transfer Learning Experiments.

Includes:
  1. CIFAR6CNN: The exact 6-layer CNN architecture illustrated in Figure 3 of Di et al. (2025).
  2. TransferredCIFAR6CNN: Nested transfer model M_j (j=0..6), where the first j layers
     are transferred from a pre-trained model and the remaining 6-j layers are initialized randomly.
  3. ModularTransferModel: Generic wrapper supporting arbitrary backbone CNNs (ResNet, VGG, etc.)
     with prefix transfer, layer freezing, and custom classifiers.
"""

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Single Convolutional block: Conv2d -> ReLU -> optional MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1,
                 has_pool: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.has_pool = has_pool
        if has_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        if self.pool is not None:
            out = self.pool(out)
        return out


class CIFAR6CNN(nn.Module):
    """
    The 6-layer CNN from Figure 3 of Di et al. (2025):
      - Layer 1: 3x3 Conv, 32 filters, ReLU
      - Layer 2: 3x3 Conv, 32 filters, ReLU, MaxPool 2x2
      - Layer 3: 3x3 Conv, 64 filters, ReLU
      - Layer 4: 3x3 Conv, 64 filters, ReLU, MaxPool 2x2
      - Layer 5: 3x3 Conv, 128 filters, ReLU
      - Layer 6: 3x3 Conv, 128 filters, ReLU, MaxPool 2x2
      - Flatten -> Dense (Linear) to num_classes
    """

    LAYER_NAMES = [
        'conv1',
        'conv2',
        'conv3',
        'conv4',
        'conv5',
        'conv6'
    ]

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # Layer 1: 3x3 conv, 32
        self.conv1 = ConvBlock(in_channels, 32, kernel_size=3, padding=1, has_pool=False)
        # Layer 2: 3x3 conv, 32, MaxPool/2
        self.conv2 = ConvBlock(32, 32, kernel_size=3, padding=1, has_pool=True)

        # Layer 3: 3x3 conv, 64
        self.conv3 = ConvBlock(32, 64, kernel_size=3, padding=1, has_pool=False)
        # Layer 4: 3x3 conv, 64, MaxPool/2
        self.conv4 = ConvBlock(64, 64, kernel_size=3, padding=1, has_pool=True)

        # Layer 5: 3x3 conv, 128
        self.conv5 = ConvBlock(64, 128, kernel_size=3, padding=1, has_pool=False)
        # Layer 6: 3x3 conv, 128, MaxPool/2
        self.conv6 = ConvBlock(128, 128, kernel_size=3, padding=1, has_pool=True)

        self.flatten = nn.Flatten()
        # After 3 MaxPools on 32x32: 32 -> 16 -> 8 -> 4, so feature map is 128 x 4 x 4 = 2048
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)

        self.layers = nn.ModuleList([
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.flatten(out)
        logits = self.classifier(out)
        return logits

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning intermediate representations Phi^j(x) for all layers j=1..6.
        """
        feats = []
        out = x
        for layer in self.layers:
            out = layer(out)
            feats.append(out)
        return feats

    def get_layer(self, j: int) -> ConvBlock:
        """1-indexed layer accessor (j in 1..6)."""
        if not (1 <= j <= len(self.layers)):
            raise ValueError(f"Layer index {j} out of range (1..{len(self.layers)})")
        return self.layers[j - 1]


def create_transferred_model(pretrained_model: CIFAR6CNN,
                             k: int,
                             num_target_classes: int,
                             freeze_transferred: bool = False) -> CIFAR6CNN:
    """
    Construct nested transferred model M_k from Section 4.1 of Di et al. (2025).

    Parameters
    ----------
    pretrained_model : CIFAR6CNN
        Pretrained model on source task (e.g. M_source).
    k : int in {0, ..., 6}
        Number of layers to transfer from the pre-trained model:
          k = 0: M_0 (all layers trained from scratch on target dataset).
          k = 6: M_6 (all 6 convolutional layers transferred from source).
          k in 1..5: M_k (first k layers transferred, remaining 6-k initialized randomly).
    num_target_classes : int
        Number of classes in the target task (e.g. 5 in Imbalance case, 10 in Balance case).
    freeze_transferred : bool, default False
        If True, freezes the parameters of the first k transferred layers (requires_grad = False).
        If False, the transferred layers are initialized with source weights but can be fine-tuned.

    Returns
    -------
    model : CIFAR6CNN
        The constructed transfer model M_k ready for target training.
    """
    target_model = CIFAR6CNN(num_classes=num_target_classes)

    # Transfer weights for the first k layers (1-indexed)
    for j in range(1, k + 1):
        source_layer = pretrained_model.get_layer(j)
        target_layer = target_model.get_layer(j)
        target_layer.load_state_dict(source_layer.state_dict())

        if freeze_transferred:
            for param in target_layer.parameters():
                param.requires_grad = False

    return target_model


def set_trainable_layers(model: CIFAR6CNN, k_freeze: int) -> Tuple[List[str], List[str]]:
    """
    Freeze first k_freeze convolutional layers, leave the rest trainable.

    Returns
    -------
    frozen_names : list of str
    trainable_names : list of str
    """
    frozen = []
    trainable = []

    for idx, name in enumerate(CIFAR6CNN.LAYER_NAMES):
        layer = model.layers[idx]
        if idx < k_freeze:
            for param in layer.parameters():
                param.requires_grad = False
            frozen.append(name)
        else:
            for param in layer.parameters():
                param.requires_grad = True
            trainable.append(name)

    # Classifier is always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    trainable.append('classifier')

    return frozen, trainable
