"""
pcor_ipcor: Pure mathematical computation of PC and IPC.

No hypothesis tests, no p-values, no result objects, no pandas.
Only the correlation coefficients themselves.

References
----------
- Zhu, Xu, Li, Zhong (2017). Projection correlation between two random vectors.
  Biometrika 104(4):829-843.  (PC)
- Zhang, Zhu (2024). Projective independence tests in high dimensions:
  the curses and the cures. Biometrika 111(3):1013-1027.  (IPC)
"""

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch


def _to_tensor(data: Union[np.ndarray, torch.Tensor],
               device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """Convert array-like to float64 torch.Tensor on the requested device."""
    if isinstance(data, torch.Tensor):
        t = data
    else:
        t = torch.as_tensor(data, dtype=torch.float64)
    if device is not None:
        t = t.to(device)
    return t


def _center_gram_batch(A: torch.Tensor) -> torch.Tensor:
    """Double-center a batch of square matrices (B, n, n) along axes 1 and 2."""
    row_mean = A.mean(dim=2, keepdim=True)
    col_mean = A.mean(dim=1, keepdim=True)
    grand_mean = A.mean(dim=(1, 2), keepdim=True)
    return A - row_mean - col_mean + grand_mean


def _anchor_angles_chunk(Z: torch.Tensor,
                         anchor_idx: torch.Tensor) -> torch.Tensor:
    """
    Angular kernel matrices for a batch of anchors.

    For each anchor r in anchor_idx, computes the matrix of angles
    arccos( <Z_i - Z_r, Z_j - Z_r> / (||Z_i - Z_r|| ||Z_j - Z_r||) ),
    zeroes the anchor row and column, then double-centers.

    Parameters
    ----------
    Z : (n, d) tensor
    anchor_idx : (B,) tensor of row indices in Z

    Returns
    -------
    A_chunk : (B, n, n)
        Double-centered angular matrices for each anchor.
    """
    n, d = Z.shape
    B = anchor_idx.shape[0]
    device = Z.device

    # (B, n, d): Z_i - Z_r for each anchor r
    Z_r = Z.unsqueeze(0) - Z[anchor_idx].unsqueeze(1)

    # Norms (B, n)
    norms = torch.sqrt(torch.sum(Z_r ** 2, dim=2).clamp(min=1e-24))

    # Outer product of norms (B, n, n)
    outer_norms = norms.unsqueeze(2) * norms.unsqueeze(1)
    zero_mask = outer_norms <= 1e-12

    # Dot products (B, n, n)
    dot_prods = torch.bmm(Z_r, Z_r.transpose(1, 2))

    safe_outer = outer_norms.masked_fill(zero_mask, 1.0)
    cos = dot_prods / safe_outer
    cos.masked_fill_(zero_mask, 0.0)
    cos.clamp_(-1.0, 1.0)

    ang = torch.arccos(cos)  # (B, n, n)

    # Zero out anchor row and column explicitly
    rows = torch.arange(n, device=device).unsqueeze(0).expand(B, n)  # (B, n)
    anchor_mask = (rows == anchor_idx.unsqueeze(1))  # (B, n)
    ang = ang.masked_fill(anchor_mask.unsqueeze(2), 0.0)   # zero rows
    ang = ang.masked_fill(anchor_mask.unsqueeze(1), 0.0)   # zero cols

    # Double center along axes 1 and 2
    A_chunk = _center_gram_batch(ang)
    return A_chunk


def pcor(X: Union[np.ndarray, torch.Tensor],
         Y: Union[np.ndarray, torch.Tensor],
         chunk_size: int = 64,
         device: Optional[Union[str, torch.device]] = None) -> float:
    """
    Pure Projection Correlation (PC) between X and Y.

    Zhu et al. (2017). No hypothesis test, no p-value.

    Parameters
    ----------
    X : (n, p) array-like
    Y : (n, q) array-like
    chunk_size : int, number of anchors processed at once (memory O(chunk_size * n^2))
    device : str or torch.device, optional

    Returns
    -------
    float : PC in [0, 1]
    """
    tx = _to_tensor(X, device)
    ty = _to_tensor(Y, device)
    if tx.ndim == 1:
        tx = tx.unsqueeze(1)
    if ty.ndim == 1:
        ty = ty.unsqueeze(1)
    if tx.shape[0] != ty.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")
    n = tx.shape[0]
    if n < 2:
        return 0.0

    device = tx.device
    anchor_chunk_size = min(int(chunk_size), n)

    pcov2 = 0.0
    self_x = 0.0
    self_y = 0.0

    for start in range(0, n, anchor_chunk_size):
        end = min(start + anchor_chunk_size, n)
        anchor_idx = torch.arange(start, end, device=device)

        A_chunk = _anchor_angles_chunk(tx, anchor_idx)  # (C, n, n)
        B_chunk = _anchor_angles_chunk(ty, anchor_idx)  # (C, n, n)

        pcov2 += torch.sum(A_chunk * B_chunk).item()
        self_x += torch.sum(A_chunk * A_chunk).item()
        self_y += torch.sum(B_chunk * B_chunk).item()

    denom = math.sqrt(max(self_x, 0.0) * max(self_y, 0.0))
    if denom <= 0.0:
        return 0.0
    pc = math.sqrt(max(pcov2, 0.0) / denom)
    return float(min(pc, 1.0))


def _median_1d(v: torch.Tensor) -> float:
    """Exact median matching projcorr.py (average middle values when even)."""
    values = torch.sort(v.flatten())[0]
    n = values.numel()
    if n % 2:
        return float(values[n // 2])
    return float(0.5 * (values[n // 2 - 1] + values[n // 2]))


def _angular_matrix(Z: torch.Tensor,
                    sigma2: Optional[float] = None) -> Tuple[torch.Tensor, float]:
    """
    Angular kernel matrix (n, n) for IPC (matches projcorr.py).

    Kernel: acos((sigma2 + z_i^T z_j) / sqrt((sigma2 + ||z_i||^2) * (sigma2 + ||z_j||^2)))
    Diagonal is set to 0.

    Returns
    -------
    angle : (n, n) tensor
    sigma2 : float
    """
    row_norms2 = (Z * Z).sum(dim=1)
    if sigma2 is None:
        sigma2 = _median_1d(row_norms2)
    aug_norm2 = sigma2 + row_norms2
    cosine = (sigma2 + torch.matmul(Z, Z.T)) / torch.sqrt(torch.outer(aug_norm2, aug_norm2))
    cosine = torch.clamp(cosine, -1.0, 1.0)
    angle = torch.acos(cosine)
    angle.fill_diagonal_(0.0)
    return angle, float(sigma2)


def _ipcov2_from_angles(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Distinct-index U-statistic for IPCov^2 in O(n^2) (matches projcorr.py).

    Includes bias-correction terms 1, 2, 3.
    """
    n = float(a.shape[0])
    s1 = (a * b).sum()
    row_a = a.sum(dim=1)
    row_b = b.sum(dim=1)
    row_product = (row_a * row_b).sum()
    total_product = a.sum() * b.sum()

    term1 = s1 / (n * (n - 1.0))
    term2 = -2.0 * (row_product - s1) / (n * (n - 1.0) * (n - 2.0))
    term3 = (total_product - 4.0 * row_product + 2.0 * s1) / (n * (n - 1.0) * (n - 2.0) * (n - 3.0))
    return term1 + term2 + term3


def ipcor(X: Union[np.ndarray, torch.Tensor],
          Y: Union[np.ndarray, torch.Tensor],
          sigma_x2: Optional[float] = None,
          sigma_y2: Optional[float] = None,
          device: Optional[Union[str, torch.device]] = None) -> float:
    """
    Pure Improved Projection Correlation (IPC) between X and Y.

    Zhang & Zhu (2024). No hypothesis test, no p-value.
    Matches projcorr.py `ipcor_test` computational core.

    Parameters
    ----------
    X : (n, p) array-like
    Y : (n, q) array-like
    sigma_x2 : float, optional. Squared kernel scale for X. If None, estimated from row norms.
    sigma_y2 : float, optional. Squared kernel scale for Y. If None, estimated from row norms.
    device : str or torch.device, optional

    Returns
    -------
    float : IPC in [0, 1]
    """
    tx = _to_tensor(X, device)
    ty = _to_tensor(Y, device)
    if tx.ndim == 1:
        tx = tx.unsqueeze(1)
    if ty.ndim == 1:
        ty = ty.unsqueeze(1)
    if tx.shape[0] != ty.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")
    n = tx.shape[0]
    if n < 4:
        return 0.0

    device = tx.device

    A, sx2 = _angular_matrix(tx, sigma_x2)
    B, sy2 = _angular_matrix(ty, sigma_y2)

    raw = _ipcov2_from_angles(A, B)
    self_x = _ipcov2_from_angles(A, A)
    self_y = _ipcov2_from_angles(B, B)

    if self_x.item() <= 0 or self_y.item() <= 0:
        return 0.0

    denom = torch.sqrt(self_x * self_y)
    ipc2 = (raw / denom).item()
    ic = math.sqrt(max(0.0, ipc2)) if ipc2 >= -1e-12 else 0.0
    return float(min(ic, 1.0))
