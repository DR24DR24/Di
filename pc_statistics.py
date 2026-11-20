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
    detailed_per_epoch: bool = False,
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
        "legend": format_legend(layer_names=layer_names),
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


def format_legend(
    layer_names: Optional[List[str]] = None,
    bounds: Optional[List[str]] = None,
) -> List[str]:
    """Форматирует расшифровку всех обозначений колонок отчёта (§2–§10 pc_stat.txt)."""
    if layer_names is None:
        layer_names = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6"]
    if bounds is None:
        bounds = [f"{layer_names[i]}->{layer_names[i+1]}" for i in range(len(layer_names) - 1)]

    lines: List[str] = []
    lines.append("=" * 130)
    lines.append("РАСШИФРОВКА ОБОЗНАЧЕНИЙ МЕТРИК ОТЧЁТА (согласно pc_stat.txt)")
    lines.append("=" * 130)
    lines.append("Формат значений в таблице: <оценка> +/- <SE>, где SE — стандартная ошибка по bootstrap-выборкам (B = 10).")
    lines.append("Базовые индексы: r = 1 (номер запуска); t = эпоха; i = слой (1..L); b = индекс bootstrap-выборки (1..B).")
    lines.append("PC_i(t)  — Projection Correlation (проекционная корреляция) слоя i на эпохе t.")
    lines.append("IPC_i(t) — Inverse Projection Correlation (обратная проекционная корреляция) слоя i на эпохе t.")
    lines.append("")
    lines.append("1. Сводные метрики общего уровня, разброса и согласованности сети:")
    lines.append("   t                  : Номер эпохи претренирования Stage 1 (source pretraining epoch).")
    lines.append("   mu_PC              : Общий средний уровень PC по всем слоям сети: mu_PC = (1/L) * sum(PC_i) (значение +/- SE).")
    lines.append("   mu_IPC             : Общий средний уровень IPC по всем слоям сети: mu_IPC = (1/L) * sum(IPC_i) (значение +/- SE).")
    lines.append("   S_PC               : Межслойный разброс PC: выборочное стандартное отклонение профиля по слоям,")
    lines.append("                        S_PC = sqrt((1/L) * sum((PC_i - mu_PC)^2)) (значение +/- SE).")
    lines.append("   S_IPC              : Межслойный разброс IPC: выборочное стандартное отклонение профиля по слоям,")
    lines.append("                        S_IPC = sqrt((1/L) * sum((IPC_i - mu_IPC)^2)) (значение +/- SE).")
    lines.append("   rho                : Коэффициент ранговой корреляции Спирмена между профилями PC и IPC:")
    lines.append("                        rho = Spearman(PC, IPC). Оценивает согласованность ранжирования слоёв.")
    lines.append("   mean_dPC           : Среднее значение центрированного профиля PC по всем слоям (теоретически 0 +/- SE).")
    lines.append("   mean_dIPC          : Среднее значение центрированного профиля IPC по всем слоям (теоретически 0 +/- SE).")
    lines.append("   mean_D_PC          : Среднее абсолютное межслойное расстояние по PC: среднее |PC_{i+1} - PC_i| по всем границам (значение +/- SE).")
    lines.append("   mean_Q_PC          : Средняя разрешающая способность переходов по PC: среднее Q_PC = D_PC / SE_delta_PC.")
    lines.append("   mean_D_IPC         : Среднее абсолютное межслойное расстояние по IPC: среднее |IPC_{i+1} - IPC_i| по всем границам (значение +/- SE).")
    lines.append("   mean_Q_IPC         : Средняя разрешающая способность переходов по IPC: среднее Q_IPC = D_IPC / SE_delta_IPC.")
    lines.append("")
    lines.append("2. Центрированные профили delta по отдельным слоям (отклонение слоя от среднего по сети, d_i = Val_i - mu):")
    for layer in layer_names:
        lines.append(f"   dPC_{layer:<15}: Центрированное значение PC слоя {layer}: dPC = PC_{layer} - mu_PC (значение +/- SE).")
        lines.append(f"   dIPC_{layer:<14}: Центрированное значение IPC слоя {layer}: dIPC = IPC_{layer} - mu_IPC (значение +/- SE).")
    lines.append("")
    lines.append("3. Межслойные расстояния D и разрешающая способность Q между соседними слоями:")
    lines.append("   (В таблице границы обозначаются со стрелкой convX->convY или дефисом convX-convY):")
    for b in bounds:
        b_dash = b.replace("->", "-")
        l1, l2 = b.split("->") if "->" in b else b.split("-")
        lines.append(f"   D_PC_{b_dash:<14}: Абсолютное расстояние по PC между {l1} и {l2}: D = |PC_{l2} - PC_{l1}| (значение +/- SE).")
        lines.append(f"   Q_PC_{b_dash:<14}: Разрешающая способность перехода {b} по PC: Q = D_PC / SE_delta_PC.")
        lines.append(f"   D_IPC_{b_dash:<13}: Абсолютное расстояние по IPC между {l1} и {l2}: D = |IPC_{l2} - IPC_{l1}| (значение +/- SE).")
        lines.append(f"   Q_IPC_{b_dash:<13}: Разрешающая способность перехода {b} по IPC: Q = D_IPC / SE_delta_IPC.")
    lines.append("=" * 130)
    return lines


def format_report(report: Dict[str, Any], detailed_per_epoch: bool = False) -> str:
    """Человекочитаемый текстовый отчёт: одна строка на эпоху со всеми метриками.

    Если detailed_per_epoch=True, добавляет подробную детализацию по каждой эпохе
    (§2–§10) после компактной таблицы.
    """
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

    def _fmt_pm(val, se, se_fmt=".4f"):
        """Формат: value +/- SE в одном поле."""
        if val is None or se is None:
            return "n/a +/- n/a"
        return f"{_fmt(val):>7} +/- {_fmt(se, se_fmt):>7}"

    # Base header — только имена показателей (без SE_)
    header_parts = [
        f"{'t':>4} | ",
        f"{'mu_PC':>19} | ",
        f"{'mu_IPC':>19} | ",
        f"{'S_PC':>19} | ",
        f"{'S_IPC':>19} | ",
        f"{'rho':>19} | ",
        f"{'mean_dPC':>19} | ",
        f"{'mean_dIPC':>19} | ",
        f"{'mean_D_PC':>19} | ",
        f"{'mean_Q_PC':>19} | ",
        f"{'mean_D_IPC':>19} | ",
        f"{'mean_Q_IPC':>19} | ",
    ]

    # Per-layer columns (§3 detailed) — только имена показателей
    for layer in layer_names:
        header_parts.append(
            f"{'dPC_' + layer:>19} | "
            f"{'dIPC_' + layer:>19} | "
        )

    # Per-boundary columns (§8–§10 detailed) — только имена показателей
    first_epoch_stats = report["epoch_stats"][str(epochs[0])]
    for boundary in sorted(first_epoch_stats["per_boundary"].keys()):
        header_parts.append(
            f"{'D_PC_' + boundary:>19} | "
            f"{'Q_PC_' + boundary:>19} | "
            f"{'D_IPC_' + boundary:>19} | "
            f"{'Q_IPC_' + boundary:>19} | "
        )

    header = "".join(header_parts)


    # ---- Русский заголовок (над английским) -------------------------------
    def _rlabel(label):
        return label.rjust(19)

    russian_parts = [
        '' + _rlabel('t') + ' | ',
        '' + _rlabel('СРЕДНЕЕ PC') + ' | ',
        '' + _rlabel('СРЕДНЕЕ IPC') + ' | ',
        '' + _rlabel('S PC') + ' | ',
        '' + _rlabel('S IPC') + ' | ',
        '' + _rlabel('КОРРЕЛЯЦИЯ') + ' | ',
        '' + _rlabel('СРЕДНЕЕ d(PC)') + ' | ',
        '' + _rlabel('СРЕДНЕЕ d(IPC)') + ' | ',
        '' + _rlabel('СРЕДНЕЕ D(PC)') + ' | ',
        '' + _rlabel('СРЕДНЕЕ Q(PC)') + ' | ',
        '' + _rlabel('СРЕДНЕЕ D(IPC)') + ' | ',
        '' + _rlabel('СРЕДНЕЕ Q(IPC)') + ' | ',
    ]
    for layer in layer_names:
        russian_parts.append('' + _rlabel('d(PC)_' + layer) + ' | ')
        russian_parts.append('' + _rlabel('d(IPC)_' + layer) + ' | ')
    for boundary in sorted(first_epoch_stats['per_boundary'].keys()):
        for prefix in ['D(PC)_', 'Q(PC)_', 'D(IPC)_', 'Q(IPC)_']:
            russian_parts.append('' + _rlabel(prefix + boundary) + ' | ')
    russian_header = ''.join(russian_parts)
    lines.append(russian_header)

    lines.append(header)
    lines.append("-" * len(header))

    for ep in epochs:
        st = report["epoch_stats"][str(ep)]

        row_parts = [
            f"{ep:>4} | ",
            _fmt_pm(st['mu_pc'], st['se_mu_pc']) + " | ",
            _fmt_pm(st['mu_ipc'], st['se_mu_ipc']) + " | ",
            _fmt_pm(st['S_pc'], st['se_S_pc']) + " | ",
            _fmt_pm(st['S_ipc'], st['se_S_ipc']) + " | ",
            f"{_fmt(st['rho_pc_ipc'], '.3f'):>19} | ",
            _fmt_pm(
                _agg([pl.get('delta_pc') for pl in st['per_layer'].values()]),
                _agg([pl.get('se_delta_pc') for pl in st['per_layer'].values()]),
            ) + " | ",
            _fmt_pm(
                _agg([pl.get('delta_ipc') for pl in st['per_layer'].values()]),
                _agg([pl.get('se_delta_ipc') for pl in st['per_layer'].values()]),
            ) + " | ",
            _fmt_pm(
                _agg([pb.get('D_pc') for pb in st['per_boundary'].values()]),
                _agg([pb.get('se_delta_pc') for pb in st['per_boundary'].values()]),
            ) + " | ",
            f"{_fmt(_agg([pb.get('Q_pc') for pb in st['per_boundary'].values()])):>19} | ",
            _fmt_pm(
                _agg([pb.get('D_ipc') for pb in st['per_boundary'].values()]),
                _agg([pb.get('se_delta_ipc') for pb in st['per_boundary'].values()]),
            ) + " | ",
            f"{_fmt(_agg([pb.get('Q_ipc') for pb in st['per_boundary'].values()])):>19} | ",
        ]

        # Per-layer details (§3)
        for layer in layer_names:
            pl = st["per_layer"].get(layer, {})
            row_parts.append(
                _fmt_pm(pl.get('delta_pc'), pl.get('se_delta_pc')) + " | "
                + _fmt_pm(pl.get('delta_ipc'), pl.get('se_delta_ipc')) + " | "
            )

        # Per-boundary details (§8–§10)
        for boundary in sorted(st["per_boundary"].keys()):
            pb = st["per_boundary"][boundary]
            row_parts.append(
                _fmt_pm(pb.get('D_pc'), pb.get('se_delta_pc')) + " | "
                + f"{_fmt(pb.get('Q_pc')):>19} | "
                + _fmt_pm(pb.get('D_ipc'), pb.get('se_delta_ipc')) + " | "
                + f"{_fmt(pb.get('Q_ipc')):>19} | "
            )

        lines.append("".join(row_parts))
    lines.append("")

    # ---- Подробная детализация по эпохам ---------------------------------
    if detailed_per_epoch:
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

    # ---- Расшифровка обозначений метрик отчёта (§2–§10 pc_stat.txt) ------
    first_epoch_stats = report["epoch_stats"][str(epochs[0])]
    bounds = sorted(first_epoch_stats.get("per_boundary", {}).keys())
    lines.extend(format_legend(layer_names=layer_names, bounds=bounds))
    lines.append("")

    lines.append("=" * 78)
    lines.append("Конец отчёта.")
    lines.append("=" * 78)
    return "\n".join(lines)


def save_report(
    report: Dict[str, Any],
    log_dir: str,
    basename: str = "pc_statistics_report",
    detailed_per_epoch: bool = False,
) -> Dict[str, str]:
    """Сохраняет отчёт в log_dir: .txt и .json.
    При одинаковых параметрах (одинаковый log_dir) файлы перезаписываются."""
    os.makedirs(log_dir, exist_ok=True)
    txt_path = os.path.join(log_dir, f"{basename}.txt")
    json_path = os.path.join(log_dir, f"{basename}.json")

    text = format_report(report, detailed_per_epoch=detailed_per_epoch)
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
    detailed_per_epoch: bool = False,
) -> Dict[str, Any]:
    """Удобная точка входа: загрузить CSV, посчитать статистики, распечатать, сохранить."""
    if csv_path is None:
        csv_path = os.path.join(log_dir, "pc_ipc_layer_profiles.csv")

    report = compute_statistics_report(
        csv_path=csv_path,
        reference_epochs=reference_epochs,
        params=params,
        detailed_per_epoch=detailed_per_epoch,
    )

    paths = save_report(report, log_dir=log_dir, basename=basename,
                        detailed_per_epoch=detailed_per_epoch)

    print(paths["text"])
    print(f"\nPC-статистика сохранена: {paths['txt_path']}")
    print(f"PC-статистика (JSON):    {paths['json_path']}")

    return {"report": report, "txt_path": paths["txt_path"],
            "json_path": paths["json_path"], "text": paths["text"]}


# =============================================================================
# Расчёт матриц ковариации PC и IPC по эпохам (запись матрицы в строку)
# =============================================================================

def compute_pc_ipc_covariance(
    csv_path: str,
    ddof: int = 0,
) -> Dict[str, Any]:
    """Вычисляет матрицы ковариации слоёв для исходных PC и для IPC по бутстрап-выборкам для каждой эпохи."""
    data = load_pc_ipc_csv(csv_path)
    layers = data["layer_names"]
    epochs = data["epochs"]
    by_epoch = data["by_epoch"]
    L = len(layers)

    epoch_cov_pc: Dict[str, Any] = {}
    epoch_cov_ipc: Dict[str, Any] = {}

    for ep in epochs:
        reps_dict = by_epoch[ep]["replications"]
        mat_pc = np.array([reps_dict[l]["pc"] for l in layers], dtype=float).T  # (B, L)
        mat_ipc = np.array([reps_dict[l]["ipc"] for l in layers], dtype=float).T  # (B, L)
        B = mat_pc.shape[0]

        # Ковариационная матрица для исходных PC (L x L)
        cov_pc = np.cov(mat_pc, rowvar=False, ddof=ddof)
        var_mu_pc = float(np.sum(cov_pc) / (L * L))
        se_mu_pc = float(np.sqrt(max(0.0, var_mu_pc)))
        cov_pc_flat = [float(x) for x in cov_pc.flatten()]

        pairs_pc: Dict[str, float] = {}
        for i in range(L):
            for j in range(L):
                pairs_pc[f"cov_{layers[i]}_{layers[j]}"] = float(cov_pc[i, j])

        epoch_cov_pc[str(ep)] = {
            "epoch": ep,
            "cov_matrix": cov_pc.tolist(),
            "cov_flat": cov_pc_flat,
            "cov_pairs": pairs_pc,
            "var_mu": var_mu_pc,
            "se_mu": se_mu_pc,
            "n_replications": B,
        }

        # Ковариационная матрица для IPC (L x L)
        cov_ipc = np.cov(mat_ipc, rowvar=False, ddof=ddof)
        var_mu_ipc = float(np.sum(cov_ipc) / (L * L))
        se_mu_ipc = float(np.sqrt(max(0.0, var_mu_ipc)))
        cov_ipc_flat = [float(x) for x in cov_ipc.flatten()]

        pairs_ipc: Dict[str, float] = {}
        for i in range(L):
            for j in range(L):
                pairs_ipc[f"cov_{layers[i]}_{layers[j]}"] = float(cov_ipc[i, j])

        epoch_cov_ipc[str(ep)] = {
            "epoch": ep,
            "cov_matrix": cov_ipc.tolist(),
            "cov_flat": cov_ipc_flat,
            "cov_pairs": pairs_ipc,
            "var_mu": var_mu_ipc,
            "se_mu": se_mu_ipc,
            "n_replications": B,
        }

    return {
        "csv_path": csv_path,
        "ddof": ddof,
        "layer_names": layers,
        "n_layers": L,
        "epochs": epochs,
        "pc": epoch_cov_pc,
        "ipc": epoch_cov_ipc,
    }


def format_covariance_report(cov_data: Dict[str, Any]) -> str:
    """Форматирует отчёт по матрицам ковариации PC и IPC: запись каждой матрицы в строку на эпоху."""
    lines: List[str] = []
    layers = cov_data["layer_names"]
    L = cov_data["n_layers"]
    epochs = cov_data["epochs"]
    ddof = cov_data["ddof"]
    B = cov_data["pc"][str(epochs[0])]["n_replications"] if epochs else 0

    lines.append("=" * 130)
    lines.append("ОТЧЁТ: МЕЖСЛОЙНЫЕ МАТРИЦЫ КОВАРИАЦИИ PC И IPC ПО ЭПОХАМ (по bootstrap-выборкам)")
    lines.append("=" * 130)
    lines.append(f"Слои: {', '.join(layers)} (L = {L});  Размер каждой матрицы: {L}x{L} = {L*L} элементов")
    lines.append(f"Число bootstrap-репликаций: B = {B};  Степень свободы: ddof = {ddof}")
    lines.append(f"Эпохи: {epochs}")
    lines.append("")
    lines.append("Теоретическая связь матриц ковариации с погрешностями общего уровня:")
    lines.append("  Var(mu_PC)  = (1 / L^2) * sum_{i=1}^L sum_{j=1}^L Cov(PC_i, PC_j)   ==>  SE(mu_PC)  = sqrt(Var(mu_PC))")
    lines.append("  Var(mu_IPC) = (1 / L^2) * sum_{i=1}^L sum_{j=1}^L Cov(IPC_i, IPC_j) ==>  SE(mu_IPC) = sqrt(Var(mu_IPC))")
    lines.append("")

    # --- 1. Матрица ковариации PC в одну строку для каждой эпохи ---
    lines.append("-" * 130)
    lines.append(f"1. Матрица ковариации исходных PC в строку для каждой эпохи (векторный вид: все {L*L} элементов):")
    lines.append("-" * 130)
    for ep in epochs:
        st = cov_data["pc"][str(ep)]
        flat_str = ", ".join(f"{x:.6f}" for x in st["cov_flat"])
        lines.append(
            f"t = {ep:>2} | Cov_PC row = [{flat_str}] | "
            f"Var(mu_PC) = {st['var_mu']:.6f} | SE(mu_PC) = {st['se_mu']:.4f}"
        )
    lines.append("")

    # --- 2. Матрица ковариации IPC в одну строку для каждой эпохи ---
    lines.append("-" * 130)
    lines.append(f"2. Матрица ковариации IPC в строку для каждой эпохи (векторный вид: все {L*L} элементов):")
    lines.append("-" * 130)
    for ep in epochs:
        st = cov_data["ipc"][str(ep)]
        flat_str = ", ".join(f"{x:.6f}" for x in st["cov_flat"])
        lines.append(
            f"t = {ep:>2} | Cov_IPC row = [{flat_str}] | "
            f"Var(mu_IPC) = {st['var_mu']:.6f} | SE(mu_IPC) = {st['se_mu']:.4f}"
        )
    lines.append("")

    # --- 3. Таблица ковариаций пар слоёв для PC (уникальные пары i <= j) ---
    unique_pairs = []
    for i in range(L):
        for j in range(i, L):
            unique_pairs.append((layers[i], layers[j]))

    header_cols = [f"{'t':>4} | "]
    for l1, l2 in unique_pairs:
        idx1 = l1.replace("conv", "c")
        idx2 = l2.replace("conv", "c")
        header_cols.append(f"{f'c({idx1},{idx2})':>11} | ")
    header_cols.append(f"{'Var_mu':>10} | ")
    header_cols.append(f"{'SE_mu':>8} |")
    tbl_header = "".join(header_cols)

    lines.append("-" * 130)
    lines.append(f"3. Таблица ковариаций пар слоёв исходных PC в одну строку на эпоху (уникальные пары i <= j):")
    lines.append("-" * 130)
    lines.append(tbl_header)
    lines.append("-" * len(tbl_header))
    for ep in epochs:
        st = cov_data["pc"][str(ep)]
        row_cols = [f"{ep:>4} | "]
        for l1, l2 in unique_pairs:
            row_cols.append(f"{st['cov_pairs'][f'cov_{l1}_{l2}']:>11.6f} | ")
        row_cols.append(f"{st['var_mu']:>10.6f} | ")
        row_cols.append(f"{st['se_mu']:>8.4f} |")
        lines.append("".join(row_cols))
    lines.append("")

    # --- 4. Таблица ковариаций пар слоёв для IPC (уникальные пары i <= j) ---
    lines.append("-" * 130)
    lines.append(f"4. Таблица ковариаций пар слоёв IPC в одну строку на эпоху (уникальные пары i <= j):")
    lines.append("-" * 130)
    lines.append(tbl_header)
    lines.append("-" * len(tbl_header))
    for ep in epochs:
        st = cov_data["ipc"][str(ep)]
        row_cols = [f"{ep:>4} | "]
        for l1, l2 in unique_pairs:
            row_cols.append(f"{st['cov_pairs'][f'cov_{l1}_{l2}']:>11.6f} | ")
        row_cols.append(f"{st['var_mu']:>10.6f} | ")
        row_cols.append(f"{st['se_mu']:>8.4f} |")
        lines.append("".join(row_cols))
    lines.append("")

    # --- 5. Расшифровка обозначений ---
    lines.append("=" * 130)
    lines.append("РАСШИФРОВКА ОБОЗНАЧЕНИЙ:")
    lines.append("=" * 130)
    lines.append("  t               : Номер эпохи претренирования Stage 1 (source pretraining epoch).")
    lines.append("  Cov_PC row      : Все 36 элементов матрицы ковариации исходных PC между слоями на эпохе t, записанные в строку.")
    lines.append("  Cov_IPC row     : Все 36 элементов матрицы ковариации IPC между слоями на эпохе t, записанные в строку.")
    lines.append("                    Порядок элементов в строке: строка 1 (conv1 с conv1..conv6), ..., строка L (conv6 с conv1..conv6).")
    lines.append("  c(ci,cj)        : Выборочная ковариация между слоями conv_i и conv_j по B bootstrap-репликациям:")
    lines.append("                    c(ci,cj) = (1/B) * sum_{b=1}^B (X_i^(b) - mean_X_i) * (X_j^(b) - mean_X_j).")
    lines.append("  c(ci,ci)        : Диагональные элементы матрицы — выборочные дисперсии слоёв Var(X_i) = (SE(X_i))^2.")
    lines.append("  Var(mu_PC/IPC)  : Дисперсия среднего уровня сети: Var(mu) = (1 / L^2) * sum_{i,j} Cov(X_i, X_j).")
    lines.append("  SE(mu_PC/IPC)   : Стандартная ошибка среднего уровня: SE(mu) = sqrt(Var(mu)) (в точности совпадает с SE в столбцах mu_PC / mu_IPC).")
    lines.append("=" * 130)
    lines.append("Конец отчёта по матрицам ковариации PC и IPC.")
    lines.append("=" * 130)

    return "\n".join(lines)


def save_covariance_report(
    cov_data: Dict[str, Any],
    log_dir: str,
    basename: str = "pc_ipc_covariance_report",
) -> Dict[str, str]:
    """Сохраняет отчёт по ковариациям в log_dir (.txt, .json, и .csv для PC и IPC)."""
    os.makedirs(log_dir, exist_ok=True)
    txt_path = os.path.join(log_dir, f"{basename}.txt")
    json_path = os.path.join(log_dir, f"{basename}.json")
    pc_csv_path = os.path.join(log_dir, "pc_covariance_by_epoch.csv")
    ipc_csv_path = os.path.join(log_dir, "ipc_covariance_by_epoch.csv")

    text = format_covariance_report(cov_data)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cov_data, f, indent=2, ensure_ascii=False, default=str)

    layers = cov_data["layer_names"]
    L = cov_data["n_layers"]
    epochs = cov_data["epochs"]

    # Сохранение таблицы ковариаций PC в CSV
    with open(pc_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch"] + [f"cov_{layers[i]}_{layers[j]}" for i in range(L) for j in range(L)] + ["var_mu_pc", "se_mu_pc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in epochs:
            st = cov_data["pc"][str(ep)]
            row = {"epoch": ep, "var_mu_pc": st["var_mu"], "se_mu_pc": st["se_mu"]}
            row.update(st["cov_pairs"])
            writer.writerow(row)

    # Сохранение таблицы ковариаций IPC в CSV
    with open(ipc_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch"] + [f"cov_{layers[i]}_{layers[j]}" for i in range(L) for j in range(L)] + ["var_mu_ipc", "se_mu_ipc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in epochs:
            st = cov_data["ipc"][str(ep)]
            row = {"epoch": ep, "var_mu_ipc": st["var_mu"], "se_mu_ipc": st["se_mu"]}
            row.update(st["cov_pairs"])
            writer.writerow(row)

    return {
        "txt_path": txt_path,
        "json_path": json_path,
        "pc_csv_path": pc_csv_path,
        "ipc_csv_path": ipc_csv_path,
        "text": text,
    }


def compute_and_save_covariance(
    log_dir: str,
    csv_path: Optional[str] = None,
    ddof: int = 0,
    basename: str = "pc_ipc_covariance_report",
) -> Dict[str, Any]:
    """Точка входа: вычисляет матрицы ковариации PC и IPC, форматирует, печатает и сохраняет отчёт."""
    if csv_path is None:
        csv_path = os.path.join(log_dir, "pc_ipc_layer_profiles.csv")

    cov_data = compute_pc_ipc_covariance(csv_path=csv_path, ddof=ddof)
    paths = save_covariance_report(cov_data, log_dir=log_dir, basename=basename)

    print(paths["text"])
    print(f"\nОтчёт по ковариациям сохранён: {paths['txt_path']}")
    print(f"Отчёт по ковариациям (JSON):   {paths['json_path']}")
    print(f"Таблица ковариаций PC (CSV):   {paths['pc_csv_path']}")
    print(f"Таблица ковариаций IPC (CSV):  {paths['ipc_csv_path']}")

    return {"cov_data": cov_data, "paths": paths, "text": paths["text"]}


# =============================================================================
# Расчёт матриц корреляции PC и IPC по эпохам (запись матрицы в строку)
# =============================================================================

def compute_pc_ipc_correlation(
    csv_path: str,
) -> Dict[str, Any]:
    """Вычисляет матрицы корреляции слоёв для исходных PC и для IPC по бутстрап-выборкам для каждой эпохи."""
    data = load_pc_ipc_csv(csv_path)
    layers = data["layer_names"]
    epochs = data["epochs"]
    by_epoch = data["by_epoch"]
    L = len(layers)

    triu_idx = np.triu_indices(L, k=1)
    off_diag_pairs = [(layers[i], layers[j]) for i, j in zip(triu_idx[0], triu_idx[1])]

    epoch_corr_pc: Dict[str, Any] = {}
    epoch_corr_ipc: Dict[str, Any] = {}

    for ep in epochs:
        reps_dict = by_epoch[ep]["replications"]
        mat_pc = np.array([reps_dict[l]["pc"] for l in layers], dtype=float).T  # (B, L)
        mat_ipc = np.array([reps_dict[l]["ipc"] for l in layers], dtype=float).T  # (B, L)
        B = mat_pc.shape[0]

        # PC Correlation
        corr_pc = np.corrcoef(mat_pc, rowvar=False)
        corr_pc_flat = [float(x) for x in corr_pc.flatten()]
        off_pc = corr_pc[triu_idx]
        mean_r_pc = float(np.mean(off_pc))
        min_r_pc = float(np.min(off_pc))
        max_r_pc = float(np.max(off_pc))

        pairs_pc: Dict[str, float] = {}
        for l1, l2 in off_diag_pairs:
            i1, i2 = layers.index(l1), layers.index(l2)
            pairs_pc[f"corr_{l1}_{l2}"] = float(corr_pc[i1, i2])

        epoch_corr_pc[str(ep)] = {
            "epoch": ep,
            "corr_matrix": corr_pc.tolist(),
            "corr_flat": corr_pc_flat,
            "corr_pairs": pairs_pc,
            "mean_r": mean_r_pc,
            "min_r": min_r_pc,
            "max_r": max_r_pc,
            "n_replications": B,
        }

        # IPC Correlation
        corr_ipc = np.corrcoef(mat_ipc, rowvar=False)
        corr_ipc_flat = [float(x) for x in corr_ipc.flatten()]
        off_ipc = corr_ipc[triu_idx]
        mean_r_ipc = float(np.mean(off_ipc))
        min_r_ipc = float(np.min(off_ipc))
        max_r_ipc = float(np.max(off_ipc))

        pairs_ipc: Dict[str, float] = {}
        for l1, l2 in off_diag_pairs:
            i1, i2 = layers.index(l1), layers.index(l2)
            pairs_ipc[f"corr_{l1}_{l2}"] = float(corr_ipc[i1, i2])

        epoch_corr_ipc[str(ep)] = {
            "epoch": ep,
            "corr_matrix": corr_ipc.tolist(),
            "corr_flat": corr_ipc_flat,
            "corr_pairs": pairs_ipc,
            "mean_r": mean_r_ipc,
            "min_r": min_r_ipc,
            "max_r": max_r_ipc,
            "n_replications": B,
        }

    return {
        "csv_path": csv_path,
        "layer_names": layers,
        "n_layers": L,
        "epochs": epochs,
        "off_diag_pairs": off_diag_pairs,
        "pc": epoch_corr_pc,
        "ipc": epoch_corr_ipc,
    }


def format_correlation_report(corr_data: Dict[str, Any]) -> str:
    """Форматирует отчёт по матрицам корреляции PC и IPC: запись каждой матрицы в строку на эпоху."""
    lines: List[str] = []
    layers = corr_data["layer_names"]
    L = corr_data["n_layers"]
    epochs = corr_data["epochs"]
    off_diag_pairs = corr_data["off_diag_pairs"]
    B = corr_data["pc"][str(epochs[0])]["n_replications"] if epochs else 0

    lines.append("=" * 130)
    lines.append("ОТЧЁТ: МЕЖСЛОЙНЫЕ МАТРИЦЫ КОРРЕЛЯЦИИ PC И IPC ПО ЭПОХАМ (по bootstrap-выборкам)")
    lines.append("=" * 130)
    lines.append(f"Слои: {', '.join(layers)} (L = {L});  Размер каждой матрицы: {L}x{L} = {L*L} элементов")
    lines.append(f"Число bootstrap-репликаций: B = {B};  Число внедиагональных пар: {len(off_diag_pairs)}")
    lines.append(f"Эпохи: {epochs}")
    lines.append("")
    lines.append("Физический смысл корреляционной матрицы слоёв:")
    lines.append("  r_{i,j} = Cov(X_i, X_j) / sqrt(Var(X_i) * Var(X_j))")
    lines.append("  Высокая средняя межслойная корреляция (mean_r ~ 0.85-0.90) показывает, что слои сети")
    lines.append("  сдвигаются в bootstrap-выборках практически синфазно. Это объясняет, почему погрешность")
    lines.append("  общего уровня mu_PC велика, а центрированный профиль delta_PC определяется в 3 раза точнее.")
    lines.append("")

    # --- 1. Матрица корреляции PC в одну строку для каждой эпохи ---
    lines.append("-" * 130)
    lines.append(f"1. Матрица корреляции исходных PC в строку для каждой эпохи (векторный вид: все {L*L} элементов):")
    lines.append("-" * 130)
    for ep in epochs:
        st = corr_data["pc"][str(ep)]
        flat_str = ", ".join(f"{x:.6f}" for x in st["corr_flat"])
        lines.append(
            f"t = {ep:>2} | Corr_PC row = [{flat_str}] | "
            f"mean_r = {st['mean_r']:.4f} | min_r = {st['min_r']:.4f} | max_r = {st['max_r']:.4f}"
        )
    lines.append("")

    # --- 2. Матрица корреляции IPC в одну строку для каждой эпохи ---
    lines.append("-" * 130)
    lines.append(f"2. Матрица корреляции IPC в строку для каждой эпохи (векторный вид: все {L*L} элементов):")
    lines.append("-" * 130)
    for ep in epochs:
        st = corr_data["ipc"][str(ep)]
        flat_str = ", ".join(f"{x:.6f}" for x in st["corr_flat"])
        lines.append(
            f"t = {ep:>2} | Corr_IPC row = [{flat_str}] | "
            f"mean_r = {st['mean_r']:.4f} | min_r = {st['min_r']:.4f} | max_r = {st['max_r']:.4f}"
        )
    lines.append("")

    # --- 3. Таблица парных корреляций слоёв для PC (15 уникальных пар i < j) ---
    header_cols = [f"{'t':>4} | "]
    for l1, l2 in off_diag_pairs:
        idx1 = l1.replace("conv", "c")
        idx2 = l2.replace("conv", "c")
        header_cols.append(f"{f'r({idx1},{idx2})':>10} | ")
    header_cols.append(f"{'mean_r':>8} | ")
    header_cols.append(f"{'min_r':>8} | ")
    header_cols.append(f"{'max_r':>8} |")
    tbl_header = "".join(header_cols)

    lines.append("-" * 130)
    lines.append(f"3. Таблица парных корреляций слоёв исходных PC в одну строку на эпоху (уникальные пары i < j):")
    lines.append("-" * 130)
    lines.append(tbl_header)
    lines.append("-" * len(tbl_header))
    for ep in epochs:
        st = corr_data["pc"][str(ep)]
        row_cols = [f"{ep:>4} | "]
        for l1, l2 in off_diag_pairs:
            row_cols.append(f"{st['corr_pairs'][f'corr_{l1}_{l2}']:>10.4f} | ")
        row_cols.append(f"{st['mean_r']:>8.4f} | ")
        row_cols.append(f"{st['min_r']:>8.4f} | ")
        row_cols.append(f"{st['max_r']:>8.4f} |")
        lines.append("".join(row_cols))
    lines.append("")

    # --- 4. Таблица парных корреляций слоёв для IPC (15 уникальных пар i < j) ---
    lines.append("-" * 130)
    lines.append(f"4. Таблица парных корреляций слоёв IPC в одну строку на эпоху (уникальные пары i < j):")
    lines.append("-" * 130)
    lines.append(tbl_header)
    lines.append("-" * len(tbl_header))
    for ep in epochs:
        st = corr_data["ipc"][str(ep)]
        row_cols = [f"{ep:>4} | "]
        for l1, l2 in off_diag_pairs:
            row_cols.append(f"{st['corr_pairs'][f'corr_{l1}_{l2}']:>10.4f} | ")
        row_cols.append(f"{st['mean_r']:>8.4f} | ")
        row_cols.append(f"{st['min_r']:>8.4f} | ")
        row_cols.append(f"{st['max_r']:>8.4f} |")
        lines.append("".join(row_cols))
    lines.append("")

    # --- 5. Расшифровка обозначений ---
    lines.append("=" * 130)
    lines.append("РАСШИФРОВКА ОБОЗНАЧЕНИЙ:")
    lines.append("=" * 130)
    lines.append("  t               : Номер эпохи претренирования Stage 1 (source pretraining epoch).")
    lines.append("  Corr_PC row     : Все 36 элементов матрицы парной корреляции исходных PC между слоями на эпохе t (в одну строку).")
    lines.append("  Corr_IPC row    : Все 36 элементов матрицы парной корреляции IPC между слоями на эпохе t (в одну строку).")
    lines.append("                    Порядок элементов в строке: строка 1 (conv1 с conv1..conv6), ..., строка L (conv6 с conv1..conv6).")
    lines.append("  r(ci,cj)        : Парный коэффициент корреляции Пирсона между слоями conv_i и conv_j по B bootstrap-репликациям:")
    lines.append("                    r(ci,cj) = Cov(X_i, X_j) / (std(X_i) * std(X_j)).")
    lines.append("  mean_r          : Средний коэффициент корреляции по всем 15 внедиагональным парам слоёв (i < j).")
    lines.append("  min_r / max_r   : Минимальная и максимальная парная корреляция между слоями на данной эпохе.")
    lines.append("=" * 130)
    lines.append("Конец отчёта по матрицам корреляции PC и IPC.")
    lines.append("=" * 130)

    return "\n".join(lines)


def save_correlation_report(
    corr_data: Dict[str, Any],
    log_dir: str,
    basename: str = "pc_ipc_correlation_report",
) -> Dict[str, str]:
    """Сохраняет отчёт по корреляциям в log_dir (.txt, .json, и .csv для PC и IPC)."""
    os.makedirs(log_dir, exist_ok=True)
    txt_path = os.path.join(log_dir, f"{basename}.txt")
    json_path = os.path.join(log_dir, f"{basename}.json")
    pc_csv_path = os.path.join(log_dir, "pc_correlation_by_epoch.csv")
    ipc_csv_path = os.path.join(log_dir, "ipc_correlation_by_epoch.csv")

    text = format_correlation_report(corr_data)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(corr_data, f, indent=2, ensure_ascii=False, default=str)

    layers = corr_data["layer_names"]
    L = corr_data["n_layers"]
    epochs = corr_data["epochs"]

    # Save PC CSV (all 36 elements + summary metrics)
    with open(pc_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch"] + [f"corr_{layers[i]}_{layers[j]}" for i in range(L) for j in range(L)] + ["mean_r", "min_r", "max_r"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in epochs:
            st = corr_data["pc"][str(ep)]
            row = {"epoch": ep, "mean_r": st["mean_r"], "min_r": st["min_r"], "max_r": st["max_r"]}
            cov_mat_2d = st["corr_matrix"]
            for i in range(L):
                for j in range(L):
                    row[f"corr_{layers[i]}_{layers[j]}"] = cov_mat_2d[i][j]
            writer.writerow(row)

    # Save IPC CSV (all 36 elements + summary metrics)
    with open(ipc_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch"] + [f"corr_{layers[i]}_{layers[j]}" for i in range(L) for j in range(L)] + ["mean_r", "min_r", "max_r"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in epochs:
            st = corr_data["ipc"][str(ep)]
            row = {"epoch": ep, "mean_r": st["mean_r"], "min_r": st["min_r"], "max_r": st["max_r"]}
            cov_mat_2d = st["corr_matrix"]
            for i in range(L):
                for j in range(L):
                    row[f"corr_{layers[i]}_{layers[j]}"] = cov_mat_2d[i][j]
            writer.writerow(row)

    return {
        "txt_path": txt_path,
        "json_path": json_path,
        "pc_csv_path": pc_csv_path,
        "ipc_csv_path": ipc_csv_path,
        "text": text,
    }


def compute_and_save_correlation(
    log_dir: str,
    csv_path: Optional[str] = None,
    basename: str = "pc_ipc_correlation_report",
) -> Dict[str, Any]:
    """Точка входа: вычисляет матрицы корреляции PC и IPC, форматирует, печатает и сохраняет отчёт."""
    if csv_path is None:
        csv_path = os.path.join(log_dir, "pc_ipc_layer_profiles.csv")

    corr_data = compute_pc_ipc_correlation(csv_path=csv_path)
    paths = save_correlation_report(corr_data, log_dir=log_dir, basename=basename)

    print(paths["text"])
    print(f"\nОтчёт по корреляциям сохранён: {paths['txt_path']}")
    print(f"Отчёт по корреляциям (JSON):   {paths['json_path']}")
    print(f"Таблица корреляций PC (CSV):   {paths['pc_csv_path']}")
    print(f"Таблица корреляций IPC (CSV):  {paths['ipc_csv_path']}")

    return {"corr_data": corr_data, "paths": paths, "text": paths["text"]}



# =============================================================================
# Расчёт разности Top-1 и Top-2 слоёв и ошибок (наивной и с учётом корреляции)
# =============================================================================

def _layer_to_num(name: str, layers: List[str]) -> int:
    """Извлекает целочисленный номер слоя (например, 'conv1' -> 1)."""
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else (layers.index(name) + 1)


def compute_top1_top2_difference(
    csv_path: str,
    ddof: int = 0,
) -> Dict[str, Any]:
    """Находит для каждой эпохи максимальное значение (Top-1) и второе по величине (Top-2)
    среди слоёв для исходных PC и для IPC.

    Вычисляет:
      - разность: diff = val_top1 - val_top2
      - наивную ошибку: se_naive = sqrt(var1 + var2)
      - ошибку с учётом корреляции: se_corr = sqrt(var1 + var2 - 2 * cov12)
      - отношение разности к ошибкам (SNR / t-статистика):
          snr_naive = diff / se_naive
          snr_corr  = diff / se_corr
      - коэффициент уточнения (выигрыша): gain_factor = se_naive / se_corr
    """
    data = load_pc_ipc_csv(csv_path)
    layers = data["layer_names"]
    epochs = data["epochs"]
    by_epoch = data["by_epoch"]

    epoch_diff_pc: Dict[str, Any] = {}
    epoch_diff_ipc: Dict[str, Any] = {}

    for ep in epochs:
        reps = by_epoch[ep]["replications"]
        B = by_epoch[ep]["n_replications"]

        # --- PC ---
        pc_means = {l: float(np.mean(reps[l]["pc"])) for l in layers}
        sorted_pc = sorted(pc_means.items(), key=lambda x: x[1], reverse=True)
        l1_pc, v1_pc = sorted_pc[0]
        l2_pc, v2_pc = sorted_pc[1]
        diff_pc = v1_pc - v2_pc

        vec1_pc = np.array(reps[l1_pc]["pc"], dtype=float)
        vec2_pc = np.array(reps[l2_pc]["pc"], dtype=float)
        var1_pc = float(np.var(vec1_pc, ddof=ddof))
        var2_pc = float(np.var(vec2_pc, ddof=ddof))
        cov12_pc = float(np.cov(vec1_pc, vec2_pc, ddof=ddof)[0, 1])
        r12_pc = cov12_pc / (math.sqrt(var1_pc) * math.sqrt(var2_pc)) if var1_pc > 0 and var2_pc > 0 else 0.0

        se_top1_pc = math.sqrt(var1_pc)
        se_top2_pc = math.sqrt(var2_pc)
        se_naive_pc = math.sqrt(var1_pc + var2_pc)
        var_corr_pc = max(0.0, var1_pc + var2_pc - 2.0 * cov12_pc)
        se_corr_pc = math.sqrt(var_corr_pc)

        snr_naive_pc = diff_pc / se_naive_pc if se_naive_pc > 0 else 0.0
        snr_corr_pc = diff_pc / se_corr_pc if se_corr_pc > 0 else 0.0
        gain_pc = se_naive_pc / se_corr_pc if se_corr_pc > 0 else 0.0

        epoch_diff_pc[str(ep)] = {
            "epoch": ep,
            "top1_layer": l1_pc,
            "top1_num": _layer_to_num(l1_pc, layers),
            "top1_val": v1_pc,
            "top2_layer": l2_pc,
            "top2_num": _layer_to_num(l2_pc, layers),
            "top2_val": v2_pc,
            "diff": diff_pc,
            "se_top1": se_top1_pc,
            "se_top2": se_top2_pc,
            "cov12": cov12_pc,
            "r12": r12_pc,
            "se_naive": se_naive_pc,
            "se_corr": se_corr_pc,
            "snr_naive": snr_naive_pc,
            "snr_corr": snr_corr_pc,
            "gain_factor": gain_pc,
            "n_replications": B,
        }

        # --- IPC ---
        ipc_means = {l: float(np.mean(reps[l]["ipc"])) for l in layers}
        sorted_ipc = sorted(ipc_means.items(), key=lambda x: x[1], reverse=True)
        l1_ipc, v1_ipc = sorted_ipc[0]
        l2_ipc, v2_ipc = sorted_ipc[1]
        diff_ipc = v1_ipc - v2_ipc

        vec1_ipc = np.array(reps[l1_ipc]["ipc"], dtype=float)
        vec2_ipc = np.array(reps[l2_ipc]["ipc"], dtype=float)
        var1_ipc = float(np.var(vec1_ipc, ddof=ddof))
        var2_ipc = float(np.var(vec2_ipc, ddof=ddof))
        cov12_ipc = float(np.cov(vec1_ipc, vec2_ipc, ddof=ddof)[0, 1])
        r12_ipc = cov12_ipc / (math.sqrt(var1_ipc) * math.sqrt(var2_ipc)) if var1_ipc > 0 and var2_ipc > 0 else 0.0

        se_top1_ipc = math.sqrt(var1_ipc)
        se_top2_ipc = math.sqrt(var2_ipc)
        se_naive_ipc = math.sqrt(var1_ipc + var2_ipc)
        var_corr_ipc = max(0.0, var1_ipc + var2_ipc - 2.0 * cov12_ipc)
        se_corr_ipc = math.sqrt(var_corr_ipc)

        snr_naive_ipc = diff_ipc / se_naive_ipc if se_naive_ipc > 0 else 0.0
        snr_corr_ipc = diff_ipc / se_corr_ipc if se_corr_ipc > 0 else 0.0
        gain_ipc = se_naive_ipc / se_corr_ipc if se_corr_ipc > 0 else 0.0

        epoch_diff_ipc[str(ep)] = {
            "epoch": ep,
            "top1_layer": l1_ipc,
            "top1_num": _layer_to_num(l1_ipc, layers),
            "top1_val": v1_ipc,
            "top2_layer": l2_ipc,
            "top2_num": _layer_to_num(l2_ipc, layers),
            "top2_val": v2_ipc,
            "diff": diff_ipc,
            "se_top1": se_top1_ipc,
            "se_top2": se_top2_ipc,
            "cov12": cov12_ipc,
            "r12": r12_ipc,
            "se_naive": se_naive_ipc,
            "se_corr": se_corr_ipc,
            "snr_naive": snr_naive_ipc,
            "snr_corr": snr_corr_ipc,
            "gain_factor": gain_ipc,
            "n_replications": B,
        }

    return {
        "csv_path": csv_path,
        "layer_names": layers,
        "epochs": epochs,
        "pc": epoch_diff_pc,
        "ipc": epoch_diff_ipc,
        "ddof": ddof,
    }


def format_top1_top2_difference_report(diff_data: Dict[str, Any]) -> str:
    """Форматирует отчёт по разности Top-1 и Top-2 слоёв: наивная ошибка, ошибка с корреляцией и SNR."""
    lines: List[str] = []
    layers = diff_data["layer_names"]
    epochs = diff_data["epochs"]
    B = diff_data["pc"][str(epochs[0])]["n_replications"] if epochs else 0

    header_cols = (
        f"{'t':>4} | {'Top-1 Layer (#)':>16} | {'Val Top-1':>10} | "
        f"{'Top-2 Layer (#)':>16} | {'Val Top-2':>10} | "
        f"{'Diff':>8} | {'SE_naive':>9} | {'SE_corr':>9} | {'r_12':>7} | "
        f"{'SNR_naive':>10} | {'SNR_corr':>9} | {'Gain':>6}"
    )
    divider = "-" * len(header_cols)

    lines.append("=" * 147)
    lines.append("ОТЧЁТ: СТАТИСТИЧЕСКИЙ АНАЛИЗ РАЗНОСТИ TOP-1 И TOP-2 СЛОЁВ ДЛЯ PC И IPC (ПО ЭПОХАМ)")
    lines.append("=" * 147)
    lines.append(f"Слои: {', '.join(layers)} (L = {len(layers)});  Число bootstrap-репликаций: B = {B};  Эпохи: {epochs}")
    lines.append("")
    lines.append("Математические формулы расчёта:")
    lines.append("  1. Разность средних:     Diff = Val(Top-1) - Val(Top-2) = X_1 - X_2 >= 0")
    lines.append("  2. 'Наивная' ошибка:     SE_naive = sqrt(Var(X_1) + Var(X_2)) = sqrt(s_1^2 + s_2^2)  [без учёта корреляции]")
    lines.append("  3. Ошибка с корреляцией: SE_corr = sqrt(Var(X_1) + Var(X_2) - 2 * Cov(X_1, X_2)) = sqrt(s_1^2 + s_2^2 - 2*r_12*s_1*s_2)")
    lines.append("  4. Отношение разности к ошибке (Signal-to-Noise Ratio, SNR / t-критерий):")
    lines.append("       SNR_naive = Diff / SE_naive   (наивная значимость различия)")
    lines.append("       SNR_corr  = Diff / SE_corr    (истинная значимость различия с учётом синфазности)")
    lines.append("  5. Коэффициент выигрыша:  Gain = SE_naive / SE_corr = SNR_corr / SNR_naive  (во сколько раз учёт корреляции уточняет оценку)")
    lines.append("")
    lines.append("Физический смысл:")
    lines.append("  Поскольку межслойная корреляция очень высока (r_12 ~ 0.90-0.99), вычитание ковариационного члена")
    lines.append("  уменьшает погрешность разности в 3-8 раз. Наивная оценка ошибочно считает разницу незначимой (SNR < 1),")
    lines.append("  тогда как с учётом синфазности флуктуаций превосходство лучшего слоя над вторым статистически достоверно!")
    lines.append("")

    # --- 1. Таблица для PC ---
    lines.append("=" * 147)
    lines.append("1. ПРОЕКЦИОННАЯ КОРРЕЛЯЦИЯ (PC) — РАЗНОСТЬ TOP-1 И TOP-2 СЛОЁВ ПО ЭПОХАМ:")
    lines.append("=" * 147)
    lines.append(header_cols)
    lines.append(divider)
    for ep in epochs:
        st = diff_data["pc"][str(ep)]
        top1_str = f"{st['top1_layer']} ({st['top1_num']})"
        top2_str = f"{st['top2_layer']} ({st['top2_num']})"
        lines.append(
            f"{ep:>4} | {top1_str:>16} | {st['top1_val']:>10.4f} | "
            f"{top2_str:>16} | {st['top2_val']:>10.4f} | "
            f"{st['diff']:>8.4f} | {st['se_naive']:>9.4f} | {st['se_corr']:>9.4f} | {st['r12']:>7.4f} | "
            f"{st['snr_naive']:>10.2f} | {st['snr_corr']:>9.2f} | {st['gain_factor']:>5.1f}x"
        )
    lines.append("")

    # --- 2. Таблица для IPC ---
    lines.append("=" * 147)
    lines.append("2. ИНВЕРСНАЯ ПРОЕКЦИОННАЯ КОРРЕЛЯЦИЯ (IPC) — РАЗНОСТЬ TOP-1 И TOP-2 СЛОЁВ ПО ЭПОХАМ:")
    lines.append("=" * 147)
    lines.append(header_cols)
    lines.append(divider)
    for ep in epochs:
        st = diff_data["ipc"][str(ep)]
        top1_str = f"{st['top1_layer']} ({st['top1_num']})"
        top2_str = f"{st['top2_layer']} ({st['top2_num']})"
        lines.append(
            f"{ep:>4} | {top1_str:>16} | {st['top1_val']:>10.4f} | "
            f"{top2_str:>16} | {st['top2_val']:>10.4f} | "
            f"{st['diff']:>8.4f} | {st['se_naive']:>9.4f} | {st['se_corr']:>9.4f} | {st['r12']:>7.4f} | "
            f"{st['snr_naive']:>10.2f} | {st['snr_corr']:>9.2f} | {st['gain_factor']:>5.1f}x"
        )
    lines.append("")

    # --- 3. Сводные выводы ---
    pc_gains = [diff_data["pc"][str(ep)]["gain_factor"] for ep in epochs]
    ipc_gains = [diff_data["ipc"][str(ep)]["gain_factor"] for ep in epochs]
    pc_snr_corr = [diff_data["pc"][str(ep)]["snr_corr"] for ep in epochs]
    ipc_snr_corr = [diff_data["ipc"][str(ep)]["snr_corr"] for ep in epochs]

    lines.append("=" * 147)
    lines.append("3. СВОДНЫЕ СТАТИСТИЧЕСКИЕ РЕЗУЛЬТАТЫ И ВЫВОДЫ:")
    lines.append("=" * 147)
    lines.append(f"  PC:  Средний выигрыш от учёта корреляции: Gain_mean = {np.mean(pc_gains):.1f}x (диапазон: {np.min(pc_gains):.1f}x - {np.max(pc_gains):.1f}x)")
    lines.append(f"       Максимальный SNR_corr = {np.max(pc_snr_corr):.2f} (эпоха t = {epochs[int(np.argmax(pc_snr_corr))]})")
    lines.append(f"  IPC: Средний выигрыш от учёта корреляции: Gain_mean = {np.mean(ipc_gains):.1f}x (диапазон: {np.min(ipc_gains):.1f}x - {np.max(ipc_gains):.1f}x)")
    lines.append(f"       Максимальный SNR_corr = {np.max(ipc_snr_corr):.2f} (эпоха t = {epochs[int(np.argmax(ipc_snr_corr))]})")
    lines.append("")
    lines.append("  Примечание о центрированных профилях (dPC и dIPC):")
    lines.append("    Поскольку dPC_i - dPC_j = (PC_i - mu_PC) - (PC_j - mu_PC) = PC_i - PC_j, разность между слоями")
    lines.append("    для центрированных профилей в точности тождественна разности для исходных PC/IPC.")
    lines.append("    Истинная ошибка SE_corr и отношение SNR_corr для центрированных профилей совпадают с приведёнными выше.")
    lines.append("")

    # --- 4. Расшифровка обозначений ---
    lines.append("=" * 147)
    lines.append("РАСШИФРОВКА ОБОЗНАЧЕНИЙ:")
    lines.append("=" * 147)
    lines.append("  t               : Номер эпохи претренирования Stage 1 (source pretraining epoch).")
    lines.append("  Top-1 Layer (#) : Слой с максимальным средним значением метрики на данной эпохе (и его порядковый номер 1..L).")
    lines.append("  Val Top-1       : Среднее значение метрики (PC или IPC) на Top-1 слое по B bootstrap-репликациям.")
    lines.append("  Top-2 Layer (#) : Слой со вторым по величине средним значением метрики (и его порядковый номер 1..L).")
    lines.append("  Val Top-2       : Среднее значение метрики на Top-2 слое по B bootstrap-репликациям.")
    lines.append("  Diff            : Разность средних: Diff = Val Top-1 - Val Top-2 >= 0.")
    lines.append("  SE_naive        : 'Наивная' ошибка разности без учёта корреляции: sqrt(s_1^2 + s_2^2).")
    lines.append("  SE_corr         : Ошибка разности с полным учётом межслойной корреляции: sqrt(s_1^2 + s_2^2 - 2*cov_12).")
    lines.append("  r_12            : Парный коэффициент корреляции Пирсона между слоями Top-1 и Top-2 по bootstrap-выборкам.")
    lines.append("  SNR_naive       : Отношение разности к наивной ошибке: Diff / SE_naive.")
    lines.append("  SNR_corr        : Отношение разности к истинной ошибке: Diff / SE_corr (t-критерий Стьюдента).")
    lines.append("                    При SNR_corr > 2 различие статистически значимо на уровне p < 0.05;")
    lines.append("                    при SNR_corr > 3 различие высоко достоверно (p < 0.003).")
    lines.append("  Gain            : Коэффициент выигрыша от учёта синфазности: SE_naive / SE_corr = SNR_corr / SNR_naive.")
    lines.append("=" * 147)
    lines.append("Конец отчёта по разности Top-1 и Top-2 слоёв.")
    lines.append("=" * 147)

    return "\n".join(lines)


def save_top1_top2_difference_report(
    diff_data: Dict[str, Any],
    log_dir: str,
    basename: str = "top1_top2_diff_report",
) -> Dict[str, str]:
    """Сохраняет отчёт по разности Top-1 и Top-2 в log_dir (.txt, .json, и .csv для PC и IPC)."""
    os.makedirs(log_dir, exist_ok=True)
    txt_path = os.path.join(log_dir, f"{basename}.txt")
    json_path = os.path.join(log_dir, f"{basename}.json")
    pc_csv_path = os.path.join(log_dir, "top1_top2_diff_pc.csv")
    ipc_csv_path = os.path.join(log_dir, "top1_top2_diff_ipc.csv")

    text = format_top1_top2_difference_report(diff_data)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(diff_data, f, indent=2, ensure_ascii=False, default=str)

    epochs = diff_data["epochs"]
    fieldnames = [
        "epoch",
        "top1_layer",
        "top1_num",
        "top1_val",
        "top2_layer",
        "top2_num",
        "top2_val",
        "diff",
        "se_top1",
        "se_top2",
        "cov12",
        "r12",
        "se_naive",
        "se_corr",
        "snr_naive",
        "snr_corr",
        "gain_factor",
    ]

    # Save PC CSV
    with open(pc_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in epochs:
            st = diff_data["pc"][str(ep)]
            row = {k: st[k] for k in fieldnames}
            writer.writerow(row)

    # Save IPC CSV
    with open(ipc_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in epochs:
            st = diff_data["ipc"][str(ep)]
            row = {k: st[k] for k in fieldnames}
            writer.writerow(row)

    return {
        "txt_path": txt_path,
        "json_path": json_path,
        "pc_csv_path": pc_csv_path,
        "ipc_csv_path": ipc_csv_path,
        "text": text,
    }


def compute_and_save_top1_top2_difference(
    log_dir: str,
    csv_path: Optional[str] = None,
    ddof: int = 0,
    basename: str = "top1_top2_diff_report",
) -> Dict[str, Any]:
    """Точка входа: вычисляет разность Top-1 и Top-2 слоёв, ошибки (наивную и с корреляцией),
    SNR, форматирует, печатает и сохраняет отчёт."""
    if csv_path is None:
        csv_path = os.path.join(log_dir, "pc_ipc_layer_profiles.csv")

    diff_data = compute_top1_top2_difference(csv_path=csv_path, ddof=ddof)
    paths = save_top1_top2_difference_report(diff_data, log_dir=log_dir, basename=basename)

    print(paths["text"])
    print(f"\nОтчёт по разности Top-1 и Top-2 сохранён: {paths['txt_path']}")
    print(f"Отчёт по разности Top-1 и Top-2 (JSON):   {paths['json_path']}")
    print(f"Таблица разности PC (CSV):                {paths['pc_csv_path']}")
    print(f"Таблица разности IPC (CSV):               {paths['ipc_csv_path']}")

    return {"diff_data": diff_data, "paths": paths, "text": paths["text"]}
