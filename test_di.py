"""
Unit and Integration Test Suite for Di et al. (2025) Algorithm.

Tests:
  1. PC Independence Property: PC(X, Y) ~ 0 for independent variables.
  2. PC Nonlinear Dependence: Detects circular / quadratic relations where Pearson r ~ 0.
  3. Numerical Equivalence: Chunked PyTorch PC matches exact formula.
  4. Model Architecture: CIFAR6CNN layers and feature map dimensions match Figure 3.
  5. Layer Freezing: Transferred layers freeze correctly (requires_grad = False).
  6. Activation Extractor: Forward hooks correctly extract GAP and flattened representations.
  7. Algorithm 1 Forward Adding Selection: Correct warm start and termination.
  8. Theorem 1 Error Bound: Valid non-asymptotic bounds in [0, 1].
"""

import math
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from projection_correlation import (
    projection_correlation,
    projection_corr_exact,
    projection_corr_approximate,
    permutation_test,
    nonasymptotic_error_bound
)
from models import (
    CIFAR6CNN,
    create_transferred_model,
    set_trainable_layers
)
from layer_extractor import LayerExtractor, extract_layer_activations
from dataset_splits import (
    generate_synthetic_transfer_dataset,
    make_dataloaders
)
from di_algorithm import (
    compute_pc_profile,
    estimate_phase_transition,
    forward_adding_layer_selection
)


class TestProjectionCorrelation(unittest.TestCase):
    """Tests for Projection Correlation mathematical properties."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_independence_zero_property(self):
        """Independent vectors should have low PC (close to 0) as sample size grows."""
        n = 300
        X = torch.randn(n, 4)
        Y = torch.randn(n, 2)
        pc = projection_correlation(X, Y, mode='exact')
        self.assertLess(pc, 0.25, f"PC for independent vectors should be small, got {pc}")

    def test_nonlinear_dependence_detection(self):
        """PC should detect nonlinear dependence Y = X^2 where Pearson correlation is zero."""
        n = 250
        # X symmetric around 0 -> Cov(X, X^2) = E[X^3] = 0 (Pearson r ~ 0)
        x = torch.linspace(-3.0, 3.0, n).unsqueeze(1)
        y = (x ** 2) + 0.1 * torch.randn(n, 1)

        # Pearson correlation
        pearson_r = float(torch.corrcoef(torch.cat([x, y], dim=1).T)[0, 1].item())
        self.assertAlmostEqual(pearson_r, 0.0, delta=0.15)

        # Projection correlation
        pc = projection_correlation(x, y, mode='exact')
        self.assertGreater(pc, 0.25, f"PC should detect nonlinear dependency, got {pc}")

    def test_exact_chunked_consistency(self):
        """Chunked computation should give identical results regardless of chunk_size."""
        n = 60
        X = torch.randn(n, 3)
        Y = torch.randint(0, 4, (n,))

        pc_chunk16 = projection_corr_exact(X, Y, chunk_size=16)
        pc_chunk64 = projection_corr_exact(X, Y, chunk_size=64)
        self.assertAlmostEqual(pc_chunk16, pc_chunk64, places=6)

    def test_approximate_vs_exact(self):
        """Approximate Monte Carlo PC should be reasonably close to exact PC."""
        n = 100
        X = torch.randn(n, 3)
        Y = torch.randn(n, 2)
        pc_exact = projection_correlation(X, Y, mode='exact')
        pc_approx = projection_correlation(X, Y, mode='approximate', K=100, seed=42)
        self.assertAlmostEqual(pc_exact, pc_approx, delta=0.15)

    def test_theorem_1_bound(self):
        """Non-asymptotic upper bound should be valid in [0, 1]."""
        bound_res = nonasymptotic_error_bound(n=500, epsilon=0.1)
        b = bound_res['upper_bound']
        self.assertGreaterEqual(b, 0.0)
        self.assertLessEqual(b, 1.0)


class TestModelArchitecture(unittest.TestCase):
    """Tests for the CIFAR6CNN model from Di et al. (2025) Figure 3."""

    def test_forward_pass_and_shapes(self):
        model = CIFAR6CNN(num_classes=5)
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 5))

        feats = model.forward_features(x)
        self.assertEqual(len(feats), 6)
        self.assertEqual(feats[0].shape, (2, 32, 32, 32))  # Layer 1
        self.assertEqual(feats[1].shape, (2, 32, 16, 16))  # Layer 2 (MaxPool)
        self.assertEqual(feats[2].shape, (2, 64, 16, 16))  # Layer 3
        self.assertEqual(feats[3].shape, (2, 64, 8, 8))    # Layer 4 (MaxPool)
        self.assertEqual(feats[4].shape, (2, 128, 8, 8))   # Layer 5
        self.assertEqual(feats[5].shape, (2, 128, 4, 4))   # Layer 6 (MaxPool)

    def test_transfer_and_freezing(self):
        base_model = CIFAR6CNN(num_classes=10)
        # Transfer 3 layers and freeze
        m3 = create_transferred_model(base_model, k=3, num_target_classes=5, freeze_transferred=True)

        # Check weights copied for layers 1..3
        for j in range(1, 4):
            base_w = base_model.get_layer(j).conv.weight
            m3_w = m3.get_layer(j).conv.weight
            self.assertTrue(torch.allclose(base_w, m3_w))
            self.assertFalse(m3_w.requires_grad, f"Layer {j} weight should be frozen")

        # Check layers 4..6 remain trainable
        for j in range(4, 7):
            m3_w = m3.get_layer(j).conv.weight
            self.assertTrue(m3_w.requires_grad, f"Layer {j} weight should be trainable")

        self.assertTrue(m3.classifier.weight.requires_grad)


class TestActivationExtractor(unittest.TestCase):
    """Tests for hook-based activation extraction."""

    def test_gap_and_flatten_extraction(self):
        model = CIFAR6CNN(num_classes=5)
        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 5, (8,))
        ds = torch.utils.data.TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=4)

        # GAP extraction
        acts_gap, labels = extract_layer_activations(model, loader, pooling='gap')
        self.assertEqual(len(acts_gap), 6)
        self.assertEqual(acts_gap['conv1'].shape, (8, 32))
        self.assertEqual(acts_gap['conv6'].shape, (8, 128))

        # Flatten extraction
        acts_flat, _ = extract_layer_activations(model, loader, pooling='flatten')
        self.assertEqual(acts_flat['conv1'].shape, (8, 32 * 32 * 32))
        self.assertEqual(acts_flat['conv6'].shape, (8, 128 * 4 * 4))


class TestDiAlgorithm(unittest.TestCase):
    """Tests for Algorithm 1 and Phase Transition estimation."""

    def test_algorithm1_synthetic(self):
        splits = generate_synthetic_transfer_dataset(
            n_source=200, n_target=150, num_classes=3,
            image_size=32, transition_layer=3, seed=123
        )
        loaders = make_dataloaders(splits, batch_size=32)

        model = CIFAR6CNN(num_classes=3)

        # Profile
        profile = compute_pc_profile(model, loaders['target_train'], max_samples=100, verbose=False)
        self.assertIn('omega', profile)
        self.assertEqual(len(profile['ordered_layers']), 6)

        pt = estimate_phase_transition(profile)
        self.assertIn('j_star_index', pt)
        self.assertIn(pt['j_star_index'], range(1, 7))

        # Algorithm 1
        res = forward_adding_layer_selection(
            model,
            target_train_loader=loaders['target_train'],
            target_val_loader=loaders['target_test'],
            num_target_classes=3,
            pc_sample_size=100,
            train_epochs=2,
            lr=0.01,
            freeze_transferred=True,
            verbose=False
        )
        self.assertIn('selected_k', res)
        self.assertIn(res['selected_k'], range(1, 7))
        self.assertLessEqual(len(res['history']), 6)


if __name__ == '__main__':
    unittest.main()
