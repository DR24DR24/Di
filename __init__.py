"""
Di et al. (2025) Transfer Learning Layer Selection Package.

Based on:
  Xiao Di, Bingnan Li, Wenzhan Song, Yuan Ke (2025).
  "Computationally Efficient General Convolutional Layer Selection for Transfer Learning",
  Stat, 14:e70037. https://doi.org/10.1002/sta4.70037
  and the theoretical proofs in appendix.pdf.
"""

from .projection_correlation import (
    projection_correlation,
    projection_corr_exact,
    projection_corr_approximate,
    permutation_test,
    nonasymptotic_error_bound
)

from .models import (
    CIFAR6CNN,
    create_transferred_model,
    set_trainable_layers,
    ConvBlock
)

from .layer_extractor import (
    LayerExtractor,
    extract_layer_activations
)

from .di_algorithm import (
    compute_pc_profile,
    estimate_phase_transition,
    forward_adding_layer_selection,
    train_target_candidate
)

from .dataset_splits import (
    split_cifar_imbalance,
    split_cifar_balance,
    generate_synthetic_transfer_dataset,
    make_dataloaders,
    CIFAR_SOURCE_CLASSES,
    CIFAR_TARGET_CLASSES
)

__version__ = "1.0.0"
__all__ = [
    'projection_correlation',
    'projection_corr_exact',
    'projection_corr_approximate',
    'permutation_test',
    'nonasymptotic_error_bound',
    'CIFAR6CNN',
    'create_transferred_model',
    'set_trainable_layers',
    'ConvBlock',
    'LayerExtractor',
    'extract_layer_activations',
    'compute_pc_profile',
    'estimate_phase_transition',
    'forward_adding_layer_selection',
    'train_target_candidate',
    'split_cifar_imbalance',
    'split_cifar_balance',
    'generate_synthetic_transfer_dataset',
    'make_dataloaders',
    'CIFAR_SOURCE_CLASSES',
    'CIFAR_TARGET_CLASSES'
]
