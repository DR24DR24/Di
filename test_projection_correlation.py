"""
Тестирование вычисления проекционной корреляции (Projection Correlation, PC)
на парах данных data_20261109_195329_x.csv / data_20261109_195329_y.csv.

Использует существующий модуль projection_correlation.py из этого же проекта.
Результат записывается в текстовый файл pc_test_result_20261109_195329.txt
и выводится в консоль.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Импорт функции проекционной корреляции из модуля проекта
try:
    from projection_correlation import projection_correlation
except ImportError as exc:
    print(f"Ошибка импорта projection_correlation: {exc}")
    sys.exit(1)


X_FILE = "data_20261109_195329_x.csv"
Y_FILE = "data_20261109_195329_y.csv"
OUT_FILE = "pc_test_result_20261109_195329.txt"

# оличество случайных направлений для приближённого метода онте-арло
APPROX_K = 50
APPROX_SEED = 42


def load_csv(path: str) -> np.ndarray:
    """Загружает CSV-файл в числовой массив numpy."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Файл пуст: {path}")
    # Приводим все столбцы к numeric (ошибки -> NaN)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        bad_cols = [c for c in df.columns if df[c].isna().any()]
        raise ValueError(f"Нечисловые/пропущенные значения в столбцах {path}: {bad_cols}")
    return df.to_numpy(dtype=np.float32)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    x_path = base_dir / X_FILE
    y_path = base_dir / Y_FILE
    out_path = base_dir / OUT_FILE

    # 1. Загрузка данных
    print(f"Загрузка X: {x_path}")
    X = load_csv(str(x_path))
    print(f"Загрузка Y: {y_path}")
    Y = load_csv(str(y_path))

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    # 2. Валидация совпадения количества образцов
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Количество строк не совпадает: X={X.shape[0]}, Y={Y.shape[0]}"
        )

    n_samples = X.shape[0]
    print(f"оличество образцов: n={n_samples}")

    # 3. ычисление проекционной корреляции
    X_t = torch.as_tensor(X, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, dtype=torch.float32)

    # 3a. Точный метод (exact) — используется когда n <= 1000
    pc_exact = projection_correlation(X_t, Y_t, mode="exact")
    print(f"Projection Correlation PC_exact(X, Y) = {pc_exact:.6f}")

    # 3b. риближённый метод онте-арло (approximate)
    pc_approx = projection_correlation(X_t, Y_t, mode="approximate", K=APPROX_K, seed=APPROX_SEED)
    print(f"Projection Correlation PC_approx(X, Y) = {pc_approx:.6f} (K={APPROX_K}, seed={APPROX_SEED})")

    # 4. апись результатов в файл
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "Projection Correlation Test Results",
        "====================================", 
        f"Timestamp: {timestamp}",
        f"X file: {X_FILE}  shape: {X.shape[0]}x{X.shape[1]}",
        f"Y file: {Y_FILE}  shape: {Y.shape[0]}x{Y.shape[1]}",
        f"n_samples: {n_samples}",
        f"Exact PC(X, Y): {pc_exact:.10f}",
        f"Approximate PC(X, Y) (K={APPROX_K}, seed={APPROX_SEED}): {pc_approx:.10f}",
        f"Difference |exact - approx|: {abs(pc_exact - pc_approx):.10f}",
        f"Exact PC in [0, 1]: {0.0 <= pc_exact <= 1.0}",
        f"Approx PC in [0, 1]: {0.0 <= pc_approx <= 1.0}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"езультат записан в: {out_path}")

if __name__ == "__main__":
    main()
