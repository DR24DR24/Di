"""
Projection Correlation (PC) for Convolutional Layer Selection in Transfer Learning.

References:
- Di, Li, Song, Ke (2025). "Computationally Efficient General Convolutional Layer
  Selection for Transfer Learning", Stat 14:e70037. https://doi.org/10.1002/sta4.70037
- Zhu, Xu, Li, Zhong (2017). "Projection correlation between two random vectors",
  Biometrika 104(4):829-843.
- Appendix to Di et al. (2025): Proofs of Theorem 1, Theorem 2, Lemma 1.1, Lemma 1.2.

Key features of this implementation:
  1. Exact formula via double-centered arccos kernel (Zhu et al. 2017, Di et al. 2025 Eq. 1-4).
  2. Chunked / batched computation along the third index r:
     Reduces memory requirement from O(n^3) to O(chunk_size * n^2), enabling exact
     evaluation for n >= 1000 without out-of-memory errors.
  3. Optimized 1D target path (projection_corr_1dy) for scalar/categorical labels.
  4. Approximate random projections Monte Carlo mode (O(K * n^2)) for very large datasets.
  5. Non-asymptotic exponential bound calculator from Theorem 1 & Lemma 1.2.
  6. Permutation independence test (H0: X indep Y).
"""

import math
import os
from datetime import datetime
from typing import Optional, Tuple, Union, Dict
import numpy as np
import torch


def _center_gram_double(A: torch.Tensor) -> torch.Tensor:
    """
    Double-center a 2-D square matrix along axes 0 and 1.
    A_centered = A - row_mean - col_mean + grand_mean
    """
    row_mean = A.mean(dim=1, keepdim=True)
    col_mean = A.mean(dim=0, keepdim=True)
    grand_mean = A.mean()
    return A - row_mean - col_mean + grand_mean


def _center_gram_double_batch(A: torch.Tensor) -> torch.Tensor:
    """
    Double-center a 3-D batch of matrices (B, n, n) along axes 1 and 2.
    """
    row_mean = A.mean(dim=2, keepdim=True)
    col_mean = A.mean(dim=1, keepdim=True)
    grand_mean = A.mean(dim=(1, 2), keepdim=True)
    return A - row_mean - col_mean + grand_mean


def compute_arccos_kernel_chunk(X: torch.Tensor, r_indices: torch.Tensor) -> torch.Tensor:
    """
    Compute double-centered arccos kernel for a batch of reference points r.

    Parameters
    ----------
    X : torch.Tensor of shape (n, p)
    r_indices : torch.Tensor of shape (B,) with indices in [0, n-1]

    Returns
    -------
    A_chunk : torch.Tensor of shape (B, n, n)
        Double-centered arccos matrices for each r in r_indices.
    """
    n, p = X.shape
    B = r_indices.shape[0]
    device = X.device
    dtype = X.dtype

    # X_r has shape (B, n, p)
    X_r = X.unsqueeze(0) - X[r_indices].unsqueeze(1)  # (B, n, p)

    # Norms of vectors (X_k - X_r) -> (B, n)
    norms = torch.sqrt(torch.sum(X_r ** 2, dim=2).clamp(min=1e-24))  # (B, n)

    # Outer product of norms -> (B, n, n)
    outer_norms = norms.unsqueeze(2) * norms.unsqueeze(1)  # (B, n, n)
    zero_mask = outer_norms <= 1e-12

    # Dot products: (X_k - X_r)^T (X_l - X_r) -> (B, n, n)
    dot_prods = torch.bmm(X_r, X_r.transpose(1, 2))  # (B, n, n)

    safe_outer = outer_norms.masked_fill(zero_mask, 1.0)
    cos_a = dot_prods / safe_outer
    cos_a.masked_fill_(zero_mask, 0.0)
    cos_a.clamp_(-1.0, 1.0)

    a = torch.arccos(cos_a)

    # Double centering along axes 1 and 2
    A_chunk = _center_gram_double_batch(a)
    return A_chunk


def compute_arccos_1d_kernel_chunk(Y: torch.Tensor, r_indices: torch.Tensor) -> torch.Tensor:
    """
    Optimized 1D arccos kernel computation for scalar labels / continuous targets.
    Following Zhu et al. (2017) and PC_Screen get_arccos_1d.

    Parameters
    ----------
    Y : torch.Tensor of shape (n,)
    r_indices : torch.Tensor of shape (B,)

    Returns
    -------
    B_chunk : torch.Tensor of shape (B, n, n)
        Double-centered arccos matrices for each r in r_indices.
    """
    n = Y.shape[0]
    B = r_indices.shape[0]
    device = Y.device
    dtype = torch.float32 if Y.dtype not in (torch.float32, torch.float64) else Y.dtype

    # Y_r has shape (B, n)
    Y_r = Y.unsqueeze(0).to(dtype) - Y[r_indices].unsqueeze(1).to(dtype)

    # Z has shape (B, n, n) where Z[b, k, l] = Y_r[b, k] * Y_r[b, l]
    Z = Y_r.unsqueeze(2) * Y_r.unsqueeze(1)

    pi_val = torch.tensor(math.pi, device=device, dtype=dtype)
    half_pi = torch.tensor(math.pi / 2.0, device=device, dtype=dtype)
    zero_val = torch.tensor(0.0, device=device, dtype=dtype)
    b = torch.where(Z < 0.0, pi_val, torch.where(Z == 0.0, half_pi, zero_val))

    B_chunk = _center_gram_double_batch(b)
    return B_chunk


def projection_corr_exact(X: Union[torch.Tensor, np.ndarray],
                          Y: Union[torch.Tensor, np.ndarray],
                          chunk_size: int = 64,
                          device: Optional[Union[str, torch.device]] = None) -> float:
    """
    Exact Projection Correlation between X and Y using chunked memory-efficient computation.

    Parameters
    ----------
    X : array-like of shape (n, p) or (n,)
    Y : array-like of shape (n, q) or (n,)
    chunk_size : int, default 64
        Number of reference points r to process simultaneously.
        Reduces memory from O(n^3) to O(chunk_size * n^2).
    device : str or torch.device, optional
        Device for tensor computations (e.g. 'cuda' or 'cpu').

    Returns
    -------
    float : Projection Correlation PC(X, Y) in [0, 1].
    """
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32)
    else:
        X = X.float()
    if not isinstance(Y, torch.Tensor):
        Y = torch.as_tensor(Y, dtype=torch.float32)
    else:
        Y = Y.float()

    if device is not None:
        X = X.to(device)
        Y = Y.to(device)

    if X.dim() == 1:
        X = X.unsqueeze(1)
    is_y_1d = (Y.dim() == 1) or (Y.dim() == 2 and Y.shape[1] == 1)
    if is_y_1d and Y.dim() == 2:
        Y = Y.squeeze(1)

    n = X.shape[0]
    ny = Y.shape[0]
    if n != ny:
        raise ValueError(f"Sample sizes do not match: X has {n}, Y has {ny}.")

    if n < 2:
        return 0.0

    # Adaptive chunk size for GPU acceleration
    if X.is_cuda and chunk_size == 64:
        if n <= 600:
            chunk_size = 256
        elif n <= 1200:
            chunk_size = 128

    # Accumulate on device to prevent GPU CPU-synchronization barriers inside loop
    S_xy = torch.zeros(1, device=X.device, dtype=torch.float64)
    S_xx = torch.zeros(1, device=X.device, dtype=torch.float64)
    S_yy = torch.zeros(1, device=X.device, dtype=torch.float64)

    indices = torch.arange(n, device=X.device)

    for start_idx in range(0, n, chunk_size):
        end_idx = min(start_idx + chunk_size, n)
        r_chunk = indices[start_idx:end_idx]

        A_chunk = compute_arccos_kernel_chunk(X, r_chunk)

        if is_y_1d:
            B_chunk = compute_arccos_1d_kernel_chunk(Y, r_chunk)
        else:
            B_chunk = compute_arccos_kernel_chunk(Y, r_chunk)

        S_xy += torch.sum(A_chunk * B_chunk).to(torch.float64)
        S_xx += torch.sum(A_chunk ** 2).to(torch.float64)
        S_yy += torch.sum(B_chunk ** 2).to(torch.float64)

    s_xy_val = float(S_xy.item())
    s_xx_val = float(S_xx.item())
    s_yy_val = float(S_yy.item())

    if s_xx_val * s_yy_val <= 0.0 or s_xy_val <= 0.0:
        return 0.0

    pc_squared = s_xy_val / math.sqrt(s_xx_val * s_yy_val)
    pc = math.sqrt(min(1.0, max(0.0, pc_squared)))
    return float(pc)


def projection_corr_approximate(X: Union[torch.Tensor, np.ndarray],
                                 Y: Union[torch.Tensor, np.ndarray],
                                 K: int = 50,
                                 seed: Optional[int] = None,
                                 device: Optional[Union[str, torch.device]] = None) -> float:
    """
    Monte Carlo Projection Correlation using K random directions on unit spheres.
    Follows Zhu et al. (2017) definition:
      PCov(X, Y)^2 = E_{alpha, beta} [ dCov(alpha^T X, beta^T Y)^2 ]
    Computation complexity: O(K * n^2), memory: O(n^2).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32)
    else:
        X = X.float()
    if not isinstance(Y, torch.Tensor):
        Y = torch.as_tensor(Y, dtype=torch.float32)
    else:
        Y = Y.float()

    if device is not None:
        X = X.to(device)
        Y = Y.to(device)

    if X.dim() == 1:
        X = X.unsqueeze(1)
    if Y.dim() == 1:
        Y = Y.unsqueeze(1)

    n, p = X.shape
    ny, q = Y.shape
    if n != ny:
        raise ValueError(f"Sample sizes do not match: {n} vs {ny}.")

    # Center and scale
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    Xn = Xc / (torch.norm(Xc, dim=0, keepdim=True) + 1e-12)
    Yn = Yc / (torch.norm(Yc, dim=0, keepdim=True) + 1e-12)

    # Sample directions on unit spheres
    Alpha = torch.randn(K, p, device=X.device, dtype=X.dtype)
    Alpha = Alpha / (torch.norm(Alpha, dim=1, keepdim=True) + 1e-12)

    Beta = torch.randn(K, q, device=Y.device, dtype=Y.dtype)
    Beta = Beta / (torch.norm(Beta, dim=1, keepdim=True) + 1e-12)

    X_proj = Alpha @ Xn.T  # (K, n)
    Y_proj = Beta @ Yn.T   # (K, n)

    sum_S_xy = 0.0
    sum_S_xx = 0.0
    sum_S_yy = 0.0

    for k in range(K):
        u = X_proj[k]
        v = Y_proj[k]

        Du = torch.abs(u[:, None] - u[None, :])
        Dv = torch.abs(v[:, None] - v[None, :])

        Au = _center_gram_double(Du)
        Av = _center_gram_double(Dv)

        s_xy = torch.sum(Au * Av).item()
        s_xx = torch.sum(Au ** 2).item()
        s_yy = torch.sum(Av ** 2).item()

        sum_S_xy += s_xy
        sum_S_xx += s_xx
        sum_S_yy += s_yy

    if sum_S_xx * sum_S_yy <= 0.0 or sum_S_xy <= 0.0:
        return 0.0

    corr = math.sqrt(sum_S_xy / math.sqrt(sum_S_xx * sum_S_yy))
    return float(min(1.0, max(0.0, corr)))

def _get_data_dimensions(
    X: torch.Tensor,
    Y: torch.Tensor
) -> Tuple[int, int, int]:
    """
    Вычислить размерности данных для проекционной корреляции.

    Параметры
    ---------
    X : torch.Tensor
        Матрица признаков (уже приведённая к 2D, если была 1D).
    Y : torch.Tensor
        Матрица целевых значений (может быть 1D или 2D).

    Возвращает
    -------
    Tuple[int, int, int]
        (n, p, q) — число образцов, число признаков в X, число целей в Y.
    """
    n = X.shape[0]
    p = X.shape[1] if X.dim() == 2 else 1
    q = Y.shape[1] if Y.dim() == 2 else 1
    return n, p, q

def projection_correlation(X: Union[torch.Tensor, np.ndarray],
                           Y: Union[torch.Tensor, np.ndarray],
                           mode: str = 'auto',
                           chunk_size: int = 64,
                           K: int = 50,
                           seed: Optional[int] = None,
                           device: Optional[Union[str, torch.device]] = None,
                           Debug_Info: int = 0) -> float:
    """
    Main entry point for computing Projection Correlation PC(X, Y).

    Parameters
    ----------
    X : (n, p) or (n,)
    Y : (n, q) or (n,)
    mode : {'auto', 'exact', 'approximate'}
        'exact'       - Chunked exact arccos double-centering kernel (Di et al. 2025 Eq. 4)
        'approximate' - Random projections distance covariance (Zhu et al. 2017)
        'auto'        - Exact if n <= 1000, approximate otherwise
    chunk_size : int, default 64
        Chunk size for exact mode.
    K : int, default 50
        Number of projections for approximate mode.
    seed : int, optional
    device : str or torch.device, optional

    Returns
    -------
    float : PC value in [0, 1]
    """
    n, p, q = _get_data_dimensions(X, Y)
    if mode == 'exact':
        if Debug_Info != 0:
            print(f"Using {mode} with chunk_size={chunk_size} for n={n}.p={p},q={q}")
        return projection_corr_exact(X, Y, chunk_size=chunk_size, device=device)
    elif mode == 'approximate':
        if Debug_Info != 0:
                    print(f"Using {mode} with chunk_size={chunk_size} for n={n}.p={p},q={q}")
        return projection_corr_approximate(X, Y, K=K, seed=seed, device=device)
    elif mode == 'auto':
        if n <= 1000:
            if Debug_Info != 0:
                print(f"Using exact mode for n={n}.p={p},q={q}")    
            return projection_corr_exact(X, Y, chunk_size=chunk_size, device=device)
        else:
            if Debug_Info != 0:
                print(f"Using approximate mode for n={n}.p={p},q={q}")  
            return projection_corr_approximate(X, Y, K=K, seed=seed, device=device)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'auto', 'exact', or 'approximate'.")


def permutation_test(X: Union[torch.Tensor, np.ndarray],
                     Y: Union[torch.Tensor, np.ndarray],
                     n_permutations: int = 500,
                     mode: str = 'auto',
                     seed: Optional[int] = None,
                     device: Optional[Union[str, torch.device]] = None) -> Tuple[float, float]:
    """
    Permutation test of independence: H0: X is independent of Y vs H1: dependent.

    Returns
    -------
    observed_pc : float
    p_value : float
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    observed = projection_correlation(X, Y, mode=mode, seed=seed, device=device)

    if not isinstance(Y, torch.Tensor):
        Y_t = torch.as_tensor(Y)
    else:
        Y_t = Y

    n = Y_t.shape[0]
    count = 0
    for _ in range(n_permutations):
        perm = torch.randperm(n)
        Y_perm = Y_t[perm]
        pc_perm = projection_correlation(X, Y_perm, mode=mode, seed=None, device=device)
        if pc_perm >= observed:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value


def nonasymptotic_error_bound(n: int, epsilon: float,
                              gamma_x: float = 1.0,
                              gamma_y: float = 1.0) -> Dict[str, float]:
    """
    Compute theoretical upper bound on estimation error from Theorem 1 and Lemma 1.2
    in Di et al. (2025) and Appendix:

      Pr(|PC_hat(X, Y)^2 - PC(X, Y)^2| > epsilon) <= 5 * c1 * exp(-c2 * n * epsilon^2 * gamma0)

    where:
      c1 = 6
      c2 = 1 / (320 * pi^4)
      M = (2 * pi)^2 = 4 * pi^2
      gamma0 = min(gamma_x^2 * gamma_y^2 / (4 * M^4),
                   gamma_x^3 * gamma_y^3 / (4 * M^4),
                   gamma_x * gamma_y / 4)
    """
    c1 = 6.0
    c2 = 1.0 / (320.0 * (math.pi ** 4))
    M = (2.0 * math.pi) ** 2  # 4 * pi^2 ~ 39.478

    term1 = (gamma_x ** 2) * (gamma_y ** 2) / (4.0 * (M ** 4))
    term2 = (gamma_x ** 3) * (gamma_y ** 3) / (4.0 * (M ** 4))
    term3 = (gamma_x * gamma_y) / 4.0
    gamma0 = min(term1, term2, term3)

    exponent = -c2 * n * (epsilon ** 2) * gamma0
    bound = min(1.0, 5.0 * c1 * math.exp(max(-500.0, exponent)))

    return {
        'n': n,
        'epsilon': epsilon,
        'gamma0': gamma0,
        'upper_bound': bound,
        'c1': c1,
        'c2': c2,
        'M': M
    }


def save_experiment_results(
    experiment_name: str,
    X,
    Y,
    pc: float,
    output_dir: str = "results",
    pc_filename_suffix: str = "pc",
) -> Dict[str, str]:
    """
    Сохраняет результаты эксперимента (матрицы X, Y и скалярную проекционную корреляцию PC)
    в файлы с меткой времени в имени.

    Параметры
    ----------
    experiment_name : str
        Имя эксперимента, используется в имени файлов.
    X : numpy.ndarray or torch.Tensor
        Матрица признаков/входных данных.
    Y : numpy.ndarray or torch.Tensor
        Матрица целевых значений/меток.
    pc : float
        Скалярное значение проекционной корреляции.
    output_dir : str, default "results"
        Папка для сохранения файлов.
    pc_filename_suffix : str, default "pc"
        Суффикс для имени файла с PC.

    Возвращает
    -------
    Dict[str, str]
        Словарь с путями к сохранённым файлам: {'x': ..., 'y': ..., 'pc': ...}

    Примеры имён файлов (experiment_name="layer_selection", timestamp="20261109_195329"):
        results/layer_selection_20261109_195329_x.csv
        results/layer_selection_20261109_195329_y.csv
        results/layer_selection_20261109_195329_pc.txt
    """
    # Генерация метки времени YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Создание папки для сохранения, если не существует
    os.makedirs(output_dir, exist_ok=True)

    # Вспомогательная функция: тензор -> numpy, иначе как есть
    def _to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    # Преобразование X и Y к numpy
    X_np = _to_numpy(X)
    Y_np = _to_numpy(Y)

    # Если 1D-массив (n,), привести к (n, 1) — сохранить одним столбцом
    if X_np.ndim == 1:
        X_np = X_np.reshape(-1, 1)
    if Y_np.ndim == 1:
        Y_np = Y_np.reshape(-1, 1)

    # Имена файлов
    base = f"{experiment_name}_{timestamp}"
    x_path = os.path.join(output_dir, f"{base}_x.csv")
    y_path = os.path.join(output_dir, f"{base}_y.csv")
    pc_path = os.path.join(output_dir, f"{base}_{pc_filename_suffix}.txt")

    # Сохранение X и Y в CSV (разделитель запятая, кодировка UTF-8)
    with open(x_path, "w", encoding="utf-8") as f:
        np.savetxt(f, X_np, delimiter=",", fmt="%.10f")
    with open(y_path, "w", encoding="utf-8") as f:
        np.savetxt(f, Y_np, delimiter=",", fmt="%.10f")

    # Сохранение PC в отдельный текстовый файл
    with open(pc_path, "w", encoding="utf-8") as f:
        f.write(f"PC = {pc:.10f}\n")

    return {
        "x": x_path,
        "y": y_path,
        "pc": pc_path,
    }
