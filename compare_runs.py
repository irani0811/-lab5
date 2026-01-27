#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def _read_metrics(metrics_path: Path) -> Dict[str, float]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"未找到指标文件: {metrics_path}")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "best_val_accuracy": float(data.get("best_val_accuracy", 0.0)),
        "best_val_macro_f1": float(data.get("best_val_macro_f1", 0.0)),
    }


def _read_best_epoch_from_history(history_path: Path) -> Tuple[Optional[int], Optional[float]]:
    if not history_path.exists():
        return None, None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        return None, None
    best = max(history, key=lambda x: float(x.get("val_accuracy", 0.0)))
    return int(best.get("epoch", 0)), float(best.get("val_accuracy", 0.0))


def _sanitize_name(path: Path) -> str:
    return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总多个实验输出目录的指标，并生成对比表与柱状图")
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        required=True,
        help="多个输出目录（每个目录内需包含 val_metrics.json）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/compare"),
        help="对比结果输出目录",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Experiment Comparison",
        help="图表标题",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    names: List[str] = []
    accs: List[float] = []
    f1s: List[float] = []

    for run_dir in args.runs:
        run_dir = Path(run_dir)
        metrics = _read_metrics(run_dir / "val_metrics.json")
        best_epoch, best_epoch_acc = _read_best_epoch_from_history(run_dir / "training_history.json")

        name = _sanitize_name(run_dir)
        names.append(name)
        accs.append(metrics["best_val_accuracy"])
        f1s.append(metrics["best_val_macro_f1"])

        rows.append(
            {
                "run": name,
                "best_val_accuracy": f"{metrics['best_val_accuracy']:.6f}",
                "best_val_macro_f1": f"{metrics['best_val_macro_f1']:.6f}",
                "best_epoch": "" if best_epoch is None else str(best_epoch),
                "best_epoch_val_accuracy": "" if best_epoch_acc is None else f"{best_epoch_acc:.6f}",
                "path": str(run_dir),
            }
        )

    # write csv
    csv_path = args.output_dir / "compare_table.csv"
    headers = [
        "run",
        "best_val_accuracy",
        "best_val_macro_f1",
        "best_epoch",
        "best_epoch_val_accuracy",
        "path",
    ]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(row[h] for h in headers) + "\n")

    # plot
    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.6), 4.8))
    ax.bar(x - width / 2, accs, width, label="Best Val Acc", color="#ff7f0e")
    ax.bar(x + width / 2, f1s, width, label="Best Val Macro-F1", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(args.title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig_path = args.output_dir / "compare_bar.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"✅ 对比表已保存: {csv_path}")
    print(f"✅ 对比图已保存: {fig_path}")


if __name__ == "__main__":
    main()
