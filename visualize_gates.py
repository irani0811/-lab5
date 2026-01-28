#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.constants import ID2LABEL
from src.utils import ensure_dir


def _get_gate_field(row: Dict[str, str]) -> Optional[str]:
    if "gate_text" in row:
        return row.get("gate_text")
    if "gate" in row:
        return row.get("gate")
    if "gate_mean" in row:
        return row.get("gate_mean")
    return None


def load_gates(path: Path) -> Tuple[List[float], List[Optional[str]], List[Optional[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"未找到 gates CSV: {path}")

    gates: List[float] = []
    true_tags: List[Optional[str]] = []
    pred_tags: List[Optional[str]] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gate_raw = _get_gate_field(row)
            if gate_raw is None:
                continue
            try:
                gate_value = float(gate_raw)
            except ValueError:
                continue
            if math.isnan(gate_value):
                continue

            gates.append(gate_value)
            true = row.get("true_tag")
            pred = row.get("pred_tag")
            true_tags.append(true if true else None)
            pred_tags.append(pred if pred else None)

    return gates, true_tags, pred_tags


def _plot_overall(gates: np.ndarray, output_dir: Path, bins: int) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.hist(gates, bins=bins, range=(0.0, 1.0), color="#1f77b4", alpha=0.85)
    ax.set_title("GMU Gate (Text Weight) Distribution")
    ax.set_xlabel("gate_text (0=image, 1=text)")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "gate_hist_overall.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_by_class(
    gates: np.ndarray,
    true_tags: List[Optional[str]],
    output_dir: Path,
    bins: int,
) -> Optional[Path]:
    if not any(true_tags):
        return None

    label_to_values: Dict[str, List[float]] = {label: [] for label in ID2LABEL}
    for gate_value, tag in zip(gates.tolist(), true_tags):
        if tag in label_to_values:
            label_to_values[tag].append(float(gate_value))

    fig, axes = plt.subplots(1, len(ID2LABEL), figsize=(5.0 * len(ID2LABEL), 4.0), sharey=True)
    if len(ID2LABEL) == 1:
        axes = [axes]

    for ax, label in zip(axes, ID2LABEL):
        values = np.asarray(label_to_values[label], dtype=np.float32)
        ax.hist(values, bins=bins, range=(0.0, 1.0), color="#ff7f0e", alpha=0.85)
        ax.set_title(f"True={label} (n={len(values)})")
        ax.set_xlabel("gate_text")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Count")
    fig.suptitle("GMU Gate by True Class", fontsize=14)
    fig.tight_layout()
    out_path = output_dir / "gate_hist_by_true_class.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_correct_incorrect(
    gates: np.ndarray,
    true_tags: List[Optional[str]],
    pred_tags: List[Optional[str]],
    output_dir: Path,
    bins: int,
) -> Optional[Path]:
    if (not any(true_tags)) or (not any(pred_tags)):
        return None

    correct: List[float] = []
    incorrect: List[float] = []
    for gate_value, t, p in zip(gates.tolist(), true_tags, pred_tags):
        if (t is None) or (p is None):
            continue
        if t == p:
            correct.append(float(gate_value))
        else:
            incorrect.append(float(gate_value))

    if not correct and not incorrect:
        return None

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if correct:
        ax.hist(
            np.asarray(correct, dtype=np.float32),
            bins=bins,
            range=(0.0, 1.0),
            color="#2ca02c",
            alpha=0.65,
            label=f"Correct (n={len(correct)})",
        )
    if incorrect:
        ax.hist(
            np.asarray(incorrect, dtype=np.float32),
            bins=bins,
            range=(0.0, 1.0),
            color="#d62728",
            alpha=0.55,
            label=f"Incorrect (n={len(incorrect)})",
        )
    ax.set_title("GMU Gate: Correct vs Incorrect")
    ax.set_xlabel("gate_text")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / "gate_hist_correct_vs_incorrect.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _export_stats(
    gates: np.ndarray,
    true_tags: List[Optional[str]],
    pred_tags: List[Optional[str]],
    output_dir: Path,
) -> Path:
    stats: Dict[str, object] = {
        "count": int(gates.size),
        "mean": float(gates.mean()) if gates.size else None,
        "std": float(gates.std()) if gates.size else None,
        "min": float(gates.min()) if gates.size else None,
        "max": float(gates.max()) if gates.size else None,
    }

    if any(true_tags):
        per_class: Dict[str, Dict[str, object]] = {}
        for label in ID2LABEL:
            values = [float(g) for g, t in zip(gates.tolist(), true_tags) if t == label]
            arr = np.asarray(values, dtype=np.float32)
            per_class[label] = {
                "count": int(arr.size),
                "mean": float(arr.mean()) if arr.size else None,
                "std": float(arr.std()) if arr.size else None,
            }
        stats["by_true_class"] = per_class

    if any(true_tags) and any(pred_tags):
        correct_values = [
            float(g)
            for g, t, p in zip(gates.tolist(), true_tags, pred_tags)
            if (t is not None) and (p is not None) and t == p
        ]
        incorrect_values = [
            float(g)
            for g, t, p in zip(gates.tolist(), true_tags, pred_tags)
            if (t is not None) and (p is not None) and t != p
        ]
        arr_c = np.asarray(correct_values, dtype=np.float32)
        arr_i = np.asarray(incorrect_values, dtype=np.float32)
        stats["correct"] = {
            "count": int(arr_c.size),
            "mean": float(arr_c.mean()) if arr_c.size else None,
            "std": float(arr_c.std()) if arr_c.size else None,
        }
        stats["incorrect"] = {
            "count": int(arr_i.size),
            "mean": float(arr_i.mean()) if arr_i.size else None,
            "std": float(arr_i.std()) if arr_i.size else None,
        }

    out_path = output_dir / "gate_stats.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GMU gate (text weight) distributions")
    parser.add_argument("--gates-csv", type=Path, default=Path("outputs/val_gates.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate_figures"))
    parser.add_argument("--bins", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    gates_list, true_tags, pred_tags = load_gates(args.gates_csv)
    if not gates_list:
        raise ValueError("未读取到任何 gate 值，请确认 CSV 是否包含 gate_text 列且不为空。")

    gates = np.asarray(gates_list, dtype=np.float32)

    overall_path = _plot_overall(gates, args.output_dir, args.bins)
    by_class_path = _plot_by_class(gates, true_tags, args.output_dir, args.bins)
    ci_path = _plot_correct_incorrect(gates, true_tags, pred_tags, args.output_dir, args.bins)
    stats_path = _export_stats(gates, true_tags, pred_tags, args.output_dir)

    print(f"✅ overall hist: {overall_path}")
    if by_class_path is not None:
        print(f"✅ by-class hist: {by_class_path}")
    if ci_path is not None:
        print(f"✅ correct/incorrect hist: {ci_path}")
    print(f"✅ stats: {stats_path}")


if __name__ == "__main__":
    main()
