"""
Projection Correlation (PC) for Convolutional Layer Selection in Transfer Learning.


"""


import math
import os
from datetime import datetime
from typing import Optional, Tuple, Union, Dict
import numpy as np
import torch
import pcor_ipcor


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
                           mode: str = 'ipcor',
                           chunk_size: int = 64,
                           device: Optional[Union[str, torch.device]] = None,
                           Debug_Info: int = 0) -> float:
    """
    Main entry point for computing Projection Correlation PC(X, Y).

    Parameters
    ----------
    X : (n, p) or (n,)
    Y : (n, q) or (n,)
    mode : pcor or ipcor (defoult)
    chunk_size : int, default 64
        Chunk size for pcor mode.
        seed : int, optional
    device : str or torch.device, optional
    Debug_Info: int, default 0
        0: no debug info
        1: print mode and chunk_size    
    Returns
    -------
    float : PC value in [0, 1]
    """
    n, p, q = _get_data_dimensions(X, Y)
    if mode == 'pcor':
        if Debug_Info != 0:
            print(f"Using {mode} with chunk_size={chunk_size} for n={n}.p={p},q={q}")
        return pcor_ipcor.pcor(X, Y, chunk_size=chunk_size, device=device)
    elif mode == 'ipcor':
        if Debug_Info != 0:
            print(f"Using {mode}  for n={n}.p={p},q={q}")
        return pcor_ipcor.ipcor(X, Y, device=device)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'pcor', or 'ipcor'.")


def save_experiment_results(
    experiment_name: str,
    X,
    Y,
    pc: Dict[str, float],
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
        for key, value in pc.items():
            f.write(f"{key} = {value:.10f}\n")
        

    return {
        "x": x_path,
        "y": y_path,
        "pc": pc_path,
    }
