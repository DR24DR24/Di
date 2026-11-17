"""
pc_statistics.py
================
Вычисление статистик PC/IPC согласно pc_stat.txt на основе логов
run_stage1_pretraining (файл pc_ipc_layer_profiles.csv).

Обозначения (согласно pc_stat.txt):
    r — номер запуска (LR × seed); здесь r = 1 (один запуск).
    t — эпоха.
    i — номер слоя,  L — число слоёв.
    b — номер bootstrap-выборки (в данной реализации b = индекс репликации),
        B — количество bootstrap-повторений (num_replications).

Все статистики считаются по каждой эпохе t отдельно из уже сохранённого
CSV-файла pc_ipc_layer_profiles.csv (переобучения модели не требует).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np


def short_hash_from_params(params: Dict[str, Any]) -> str:
    """Тот же короткий хеш (до 8 алfanumerичных символов), что в блокноте:
    sha256(json.dumps(params, sort_keys=True))[:8]."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(param_str.encode("utf-8")).hexdigest()[:8]


def _layer_names_from_csv_fieldnames(fieldnames: List[str]) -> List[str]:
    """Извлекает имена слоёв из заголовков CSV: pc_<layer>, ipc_<layer>."""
    layers: List[str] = []
    for fn in fieldnames:
        if fn.startswith("pc_"):
            layer = fn[len("pc_"):]
            if layer not in layers:
                layers.append(layer)
    return layers


def load_pc_ipc_csv(csv_path: str) -> Dict[str, Any]:
    """Загружает pc_ipc_layer_profiles.csv.

    Возвращает:
        {
            'fieldnames': [...],
            'layer_names': [...],
            'epochs': [int, ...],
            'by_epoch': {
                epoch: {
                    'replications': { layer: {'pc': [...], 'ipc': [...]} },
                    'n_replications': int,
                }
            }
        }
    """
    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    layer_names = _layer_names_from_csv_fieldnames(fieldnames)

    by_epoch: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        try:
            epoch = int(row["epoch"])
            dataset_id = int(row["dataset_id"])
        except (ValueError, KeyError):
            continue
        if dataset_id < 0:
            continue
        if epoch not in by_epoch:
            by_epoch[epoch] = {"replications": {}}
        rep_entries = by_epoch[epoch]["replications"]
        for layer in layer_names:
            pc_key = f"pc_{layer}"
            ipc_key = f"ipc_{layer}"
            try:
                pc_val = float(row[pc_key])
                ipc_val = float(row[ipc_key])
            except (ValueError, KeyError):
                continue
            entry = rep_entries.setdefault(layer, {"pc": [], "ipc": []})
            entry["pc"].append(pc_val)
            entry["ipc"].append(ipc_val)

    epochs = sorted(by_epoch.keys())
    for ep in epochs:
        rep_entries = by_epoch[ep]["replications"]
        n_reps = 0
        for layer in layer_names:
            if layer in rep_entries:
                n_reps = len(rep_entries[layer]["pc"])
                break
        by_epoch[ep]["n_replications"] = n_reps

    return {
        "fieldnames": fieldnames,
        "layer_names": layer_names,
        "epochs": epochs,
        "by_epoch": by_epoch,
    }


def _rankdata(values: List[float]) -> List[float]:
    """Ранги с учётом связей (средний ранг для равных значений)."""
    n = len(values)
    sorted_pairs = sorted((v, idx) for idx, v in enumerate(values))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_pairs[j + 1][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][1]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    """Спирменова корреляция (Pearson на рангах)."""
    n = len(x)
    if n < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[k] - mx) * (ry[k] - my) for k in range(n))
    den_x = math.sqrt(sum((v - mx) ** 2 for v in rx))
    den_y = math.sqrt(sum((v - my) ** 2 for v in ry))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def _empty_epoch_stats(layer_names: List[str]) -> Dict[str, Any]:
    return {
        "mu_pc": None, "se_mu_pc": None,
        "mu_ipc": None, "se_mu_ipc": None,
        "S_pc": None, "se_S_pc": None,
        "S_ipc": None, "se_S_ipc": None,
        "rho_pc_ipc": None,
        "per_layer": {
            l: {"delta_pc": None, "se_delta_pc": None,
                "delta_ipc": None, "se_delta_ipc": None}
            for l in layer_names
        },
        "per_boundary": {},
        "n_replications": 0,
        "n_layers": len(layer_names),
    }


def compute_epoch_statistics(
    pc_by_layer: Dict[str, List[float]],
    ipc_by_layer: Dict[str, List[float]],
    layer_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Статистики pc_stat.txt для одной эпохи t.

    Вход: pc_by_layer[layer] = список PC по репликациям b (аналогично для IPC).

    Возвращает словарь со скалярами (§2, §4, §7) и деталями по слоям / границам
    (§3, §8–§10) для эпохи.
    """
    if layer_names is None:
        layer_names = list(pc_by_layer.keys())
    L = len(layer_names)

    B_pc = len(pc_by_layer.get(layer_names[0], [])) if layer_names else 0
    B_ipc = len(ipc_by_layer.get(layer_names[0], [])) if layer_names else 0
    B = min(B_pc, B_ipc)
    if B == 0 or L == 0:
        return _empty_epoch_stats(layer_names)

    # Матрицы [b][i]
    pc_matrix: List[List[float]] = []
    ipc_matrix: List[List[float]] = []
    for b in range(B):
        pc_row = []
        ipc_row = []
        for layer in layer_names:
            pvals = pc_by_layer.get(layer, [])
            ivals = ipc_by_layer.get(layer, [])
            pc_row.append(pvals[b] if b < len(pvals) else float("nan"))
            ipc_row.append(ivals[b] if b < len(ivals) else float("nan"))
        pc_matrix.append(pc_row)
        ipc_matrix.append(ipc_row)

    # --- §2: Общий уровень по слоям ---
    mu_pc_per_b = [float(np.mean(row)) for row in pc_matrix]
    mu_ipc_per_b = [float(np.mean(row)) for row in ipc_matrix]
    mu_pc = float(np.mean(mu_pc_per_b))
    se_mu_pc = float(np.std(mu_pc_per_b, ddof=0))
    mu_ipc = float(np.mean(mu_ipc_per_b))
    se_mu_ipc = float(np.std(mu_ipc_per_b, ddof=0))

    # --- §3: Центрированный межслойный профиль ---
    delta_pc_matrix: List[List[float]] = []
    delta_ipc_matrix: List[List[float]] = []
    for b in range(B):
        delta_pc_matrix.append([pc_matrix[b][i] - mu_pc_per_b[b] for i in range(L)])
        delta_ipc_matrix.append([ipc_matrix[b][i] - mu_ipc_per_b[b] for i in range(L)])

    per_layer: Dict[str, Dict[str, Optional[float]]] = {}
    for i, layer in enumerate(layer_names):
        dp_vals = [delta_pc_matrix[b][i] for b in range(B)]
        di_vals = [delta_ipc_matrix[b][i] for b in range(B)]
        per_layer[layer] = {
            "delta_pc": float(np.mean(dp_vals)),
            "se_delta_pc": float(np.std(dp_vals, ddof=0)),
            "delta_ipc": float(np.mean(di_vals)),
            "se_delta_ipc": float(np.std(di_vals, ddof=0)),
        }

    # --- §4: Межслойный разброс ---
    S_pc_per_b = [
        float(np.sqrt(np.mean([d ** 2 for d in row]))) for row in delta_pc_matrix
    ]
    S_ipc_per_b = [
        float(np.sqrt(np.mean([d ** 2 for d in row]))) for row in delta_ipc_matrix
    ]
    S_pc = float(np.mean(S_pc_per_b))
    se_S_pc = float(np.std(S_pc_per_b, ddof=0))
    S_ipc = float(np.mean(S_ipc_per_b))
    se_S_ipc = float(np.std(S_ipc_per_b, ddof=0))

    # --- §7: Согласованность ранжирования (Спирмен) ---
    mean_pc_per_layer = [
        float(np.mean([pc_matrix[b][i] for b in range(B)])) for i in range(L)
    ]
    mean_ipc_per_layer = [
        float(np.mean([ipc_matrix[b][i] for b in range(B)])) for i in range(L)
    ]
    rho_pc_ipc = _spearman(mean_pc_per_layer, mean_ipc_per_layer)

    # --- §8–§10: Расстояние между соседними слоями ---
    per_boundary: Dict[str, Dict[str, Optional[float]]] = {}
    for i in range(L - 1):
        boundary_key = f"{layer_names[i]}->{layer_names[i + 1]}"
        D_pc_per_b = [abs(pc_matrix[b][i + 1] - pc_matrix[b][i]) for b in range(B)]
        D_ipc_per_b = [abs(ipc_matrix[b][i + 1] - ipc_matrix[b][i]) for b in range(B)]
        D_pc = float(np.mean(D_pc_per_b))
        D_ipc = float(np.mean(D_ipc_per_b))

        diff_pc_per_b = [pc_matrix[b][i + 1] - pc_matrix[b][i] for b in range(B)]
        diff_ipc_per_b = [ipc_matrix[b][i + 1] - ipc_matrix[b][i] for b in range(B)]
        se_delta_pc = float(np.std(diff_pc_per_b, ddof=0))
        se_delta_ipc = float(np.std(diff_ipc_per_b, ddof=0))

        Q_pc = (D_pc / se_delta_pc) if se_delta_pc > 0 else None
        Q_ipc = (D_ipc / se_delta_ipc) if se_delta_ipc > 0 else None

        per_boundary[boundary_key] = {
            "D_pc": D_pc,
            "se_delta_pc": se_delta_pc,
            "Q_pc": Q_pc,
            "D_ipc": D_ipc,
            "se_delta_ipc": se_delta_ipc,
            "Q_ipc": Q_ipc,
        }

    return {
        "mu_pc": mu_pc,
        "se_mu_pc": se_mu_pc,
        "mu_ipc": mu_ipc,
        "se_mu_ipc": se_mu_ipc,
        "S_pc": S_pc,
        "se_S_pc": se_S_pc,
        "S_ipc": S_ipc,
        "se_S_ipc": se_S_ipc,
        "rho_pc_ipc": rho_pc_ipc,
        "per_layer": per_layer,
        "per_boundary": per_boundary,
        "n_replications": B,
        "n_layers": L,
    }


def compute_statistics_report(
    csv_path: str,
    reference_epochs: Optional[List[int]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Полный отчёт по всем эпохам из CSV."""
    data = load_pc_ipc_csv(csv_path)
    layer_names = data["layer_names"]
    epochs = data["epochs"]
    by_epoch = data["by_epoch"]

    if reference_epochs is None:
        reference_epochs = list(epochs)
    ref_set = set(reference_epochs)

    epoch_stats: Dict[str, Any] = {}
    for ep in epochs:
        rep = by_epoch[ep]["replications"]
        pc_by_layer = {l: rep[l]["pc"] for l in layer_names if l in rep}
        ipc_by_layer = {l: rep[l]["ipc"] for l in layer_names if l in rep}
        stats = compute_epoch_statistics(pc_by_layer, ipc_by_layer, layer_names=layer_names)
        # Статистики считаются для ВСЕХ эпох (не только опорных) —
        # по одной строке на эпоху в итоговом отчёте.
        epoch_stats[str(ep)] = stats

    return {
        "spec": "pc_stat.txt",
        "note": (
            "r = 1 (один запуск); b = индекс репликации (bootstrap-выборка); "
            "B = num_replications; t = эпоха; i = слой; L = число слоёв. "
            "std вычисляется как np.std (ddof=0)."
        ),
        "params": params or {},
        "csv_path": csv_path,
        "layer_names": layer_names,
        "n_layers": len(layer_names),
        "n_replications": by_epoch[epochs[0]]["n_replications"] if epochs else 0,
        "epochs": epochs,
        "reference_epochs": sorted(ref_set),
        "epoch_stats": epoch_stats,
    }


def _fmt(v: Optional[float], fmt: str = ".4f") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "inf" if v == float("inf") else ("-inf" if v == -float("inf") else "nan")
    return format(v, fmt)


def _agg(values: List[Optional[float]]) -> Optional[float]:
    """Среднее по списку, игнорируя None."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def format_report(report: Dict[str, Any]) -> str:
    """Человекочитаемый текстовый отчёт: одна строка на эпоху со всеми метриками."""
    lines: List[str] = []
    layer_names: List[str] = report["layer_names"]
    epochs: List[int] = report["epochs"]

    lines.append("=" * 78)
    lines.append("СТАТИСТИКА PC/IPC согласно pc_stat.txt")
    lines.append("=" * 78)
    lines.append(f"Обозначения: r = 1 (один запуск); t = эпоха; i = слой; L = {report['n_layers']};")
    lines.append(f"             b = индекс репликации (bootstrap-выборка); B = {report['n_replications']}.")
    lines.append(f"Слои: {', '.join(layer_names)}")
    lines.append(f"Эпохи: {epochs}")
    if report.get("params"):
        lines.append("")
        lines.append("Параметры запуска:")
        for k in sorted(report["params"].keys()):
            lines.append(f"  {k}: {report['params'][k]}")
    lines.append("")

    # ---- Одна строка на эпоху -------------------------------------------
    lines.append("-" * 130)
    lines.append("Одна строка на эпоху (все статистики pc_stat.txt)")
    lines.append("-" * 130)
    header = (
        f"{'t':>4} | "
        f"{'mu_PC':>7} {'SE_muPC':>7} | "
        f"{'mu_IPC':>7} {'SE_muIPC':>7} | "
        f"{'S_PC':>6} {'SE_SPc':>6} | "
        f"{'S_IPC':>6} {'SE_SIPC':>6} | "
        f"{'rho':>5} | "
        f"{'mean_dPC':>7} {'SE_dPC':>7} | "
        f"{'mean_dIPC':>7} {'SE_dIPC':>7} | "
        f"{'mean_D_PC':>7} {'mean_Q_PC':>7} | "
        f"{'mean_D_IPC':>7} {'mean_Q_IPC':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for ep in epochs:
        st = report["epoch_stats"][str(ep)]

        # §3: агрегируем центрированные профили по слоям
        delta_pc_vals = [pl.get("delta_pc") for pl in st["per_layer"].values()]
        se_delta_pc_vals = [pl.get("se_delta_pc") for pl in st["per_layer"].values()]
        delta_ipc_vals = [pl.get("delta_ipc") for pl in st["per_layer"].values()]
        se_delta_ipc_vals = [pl.get("se_delta_ipc") for pl in st["per_layer"].values()]

        # §8–§10: агрегируем по границам между слоями
        D_pc_vals = [pb.get("D_pc") for pb in st["per_boundary"].values()]
        Q_pc_vals = [pb.get("Q_pc") for pb in st["per_boundary"].values()]
        D_ipc_vals = [pb.get("D_ipc") for pb in st["per_boundary"].values()]
        Q_ipc_vals = [pb.get("Q_ipc") for pb in st["per_boundary"].values()]

        lines.append(
            f"{ep:>4} | "
            f"{_fmt(st['mu_pc']):>7} {_fmt(st['se_mu_pc']):>7} | "
            f"{_fmt(st['mu_ipc']):>7} {_fmt(st['se_mu_ipc']):>7} | "
            f"{_fmt(st['S_pc']):>6} {_fmt(st['se_S_pc']):>6} | "
            f"{_fmt(st['S_ipc']):>6} {_fmt(st['se_S_ipc']):>6} | "
            f"{_fmt(st['rho_pc_ipc'], '.3f'):>5} | "
            f"{_fmt(_agg(delta_pc_vals)):>7} {_fmt(_agg(se_delta_pc_vals)):>7} | "
            f"{_fmt(_agg(delta_ipc_vals)):>7} {_fmt(_agg(se_delta_ipc_vals)):>7} | "
            f"{_fmt(_agg(D_pc_vals)):>7} {_fmt(_agg(Q_pc_vals)):>7} | "
            f"{_fmt(_agg(D_ipc_vals)):>7} {_fmt(_agg(Q_ipc_vals)):>7}"
        )
    lines.append("")

    # ---- Подробная детализация по эпохам ---------------------------------
    for ep in epochs:
        st = report["epoch_stats"][str(ep)]
        lines.append("=" * 78)
        lines.append(f"Эпоха t = {ep} — подробно")
        lines.append("=" * 78)

        lines.append("")
        lines.append("§2 Общий уровень по слоям (среднее и SE по репликациям):")
        lines.append(
            f"  mu_PC    = {_fmt(st['mu_pc']):>10}   SE_mu_PC = {_fmt(st['se_mu_pc'])}")
        lines.append(
            f"  mu_IPC   = {_fmt(st['mu_ipc']):>10}   SE_mu_IPC = {_fmt(st['se_mu_ipc'])}")

        lines.append("")
        lines.append("§4 Межслойный разброс:")
        lines.append(
            f"  S_PC     = {_fmt(st['S_pc']):>10}   SE_S_PC  = {_fmt(st['se_S_pc'])}")
        lines.append(
            f"  S_IPC    = {_fmt(st['S_ipc']):>10}   SE_S_IPC = {_fmt(st['se_S_ipc'])}")

        lines.append("")
        lines.append("§7 Согласованность ранжирования (Спирмен):")
        lines.append(f"  rho_PC,IPC = {_fmt(st['rho_pc_ipc'], '.4f')}")

        lines.append("")
        lines.append("§3 Центрированный межслойный профиль (по слоям):")
        lines.append(
            f"  {'слой':<12} | {'delta_PC':>10} {'SE_delta_PC':>10} | "
            f"{'delta_IPC':>10} {'SE_delta_IPC':>10}"
        )
        lines.append("  " + "-" * 56)
        for layer in layer_names:
            pl = st["per_layer"].get(layer, {})
            lines.append(
                f"  {layer:<12} | {_fmt(pl.get('delta_pc')):>10} {_fmt(pl.get('se_delta_pc')):>10} | "
                f"{_fmt(pl.get('delta_ipc')):>10} {_fmt(pl.get('se_delta_ipc')):>10}"
            )

        lines.append("")
        lines.append("§8–§10 Расстояние между соседними слоями и Q:")
        lines.append(
            f"  {'граница':<18} | {'D_PC':>8} {'SE_dPC':>8} {'Q_PC':>8} | "
            f"{'D_IPC':>8} {'SE_dIPC':>8} {'Q_IPC':>8}"
        )
        lines.append("  " + "-" * 70)
        for boundary in sorted(st["per_boundary"].keys()):
            pb = st["per_boundary"][boundary]
            lines.append(
                f"  {boundary:<18} | {_fmt(pb.get('D_pc')):>8} {_fmt(pb.get('se_delta_pc')):>8} "
                f"{_fmt(pb.get('Q_pc')):>8} | {_fmt(pb.get('D_ipc')):>8} "
                f"{_fmt(pb.get('se_delta_ipc')):>8} {_fmt(pb.get('Q_ipc')):>8}"
            )
        lines.append("")

    lines.append("=" * 78)
    lines.append("Конец отчёта.")
    lines.append("=" * 78)
    return "\n".join(lines)


def save_report(
    report: Dict[str, Any],
    log_dir: str,
    basename: str = "pc_statistics_report",
) -> Dict[str, str]:
    """Сохраняет отчёт в log_dir: .txt и .json.
    При одинаковых параметрах (одинаковый log_dir) файлы перезаписываются."""
    os.makedirs(log_dir, exist_ok=True)
    txt_path = os.path.join(log_dir, f"{basename}.txt")
    json_path = os.path.join(log_dir, f"{basename}.json")

    text = format_report(report)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    return {"txt_path": txt_path, "json_path": json_path, "text": text}


def compute_and_save(
    log_dir: str,
    csv_path: Optional[str] = None,
    reference_epochs: Optional[List[int]] = None,
    params: Optional[Dict[str, Any]] = None,
    basename: str = "pc_statistics_report",
) -> Dict[str, Any]:
    """Удобная точка входа: загрузить CSV, посчитать статистики, распечатать, сохранить."""
    if csv_path is None:
        csv_path = os.path.join(log_dir, "pc_ipc_layer_profiles.csv")

    report = compute_statistics_report(
        csv_path=csv_path,
        reference_epochs=reference_epochs,
        params=params,
    )

    paths = save_report(report, log_dir=log_dir, basename=basename)

    print(paths["text"])
    print(f"\nPC-статистика сохранена: {paths['txt_path']}")
    print(f"PC-статистика (JSON):    {paths['json_path']}")

    return {"report": report, "txt_path": paths["txt_path"],
            "json_path": paths["json_path"], "text": paths["text"]}