#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.constants import ID2LABEL


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_neutral_f1(metrics: Dict[str, object]) -> float:
    report = metrics.get("classification_report")
    if not isinstance(report, dict):
        return float("nan")
    neutral = report.get("neutral")
    if not isinstance(neutral, dict):
        return float("nan")
    return _safe_float(neutral.get("f1-score"))


def _best_epoch_from_history(history_path: Path) -> Optional[int]:
    if not history_path.exists():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        return None
    best = max(history, key=lambda x: float(x.get("val_accuracy", 0.0)))
    return int(best.get("epoch", 0))


def _parse_run_name(name: str) -> Dict[str, str]:
    # expected: outputs_352_c{0|1}_{none|weak|strong}_{base|ls|rdrop|md}
    parts = name.split("_")
    # minimal guard
    info = {"run": name, "clean": "", "aug": "", "reg": ""}
    try:
        c_part = next(p for p in parts if p.startswith("c") and len(p) == 2)
        info["clean"] = "on" if c_part == "c1" else "off"
    except StopIteration:
        pass

    # aug/reg are last 2 parts in our naming convention
    if len(parts) >= 2:
        info["reg"] = parts[-1]
    if len(parts) >= 3:
        info["aug"] = parts[-2]
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 3.5.2 ablation runs")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="project root containing outputs_352_* directories",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="outputs_352_*",
        help="glob pattern for run directories",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs_352_summary"),
        help="output directory for summary artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in args.root.glob(args.pattern) if p.is_dir()])
    rows: List[Dict[str, str]] = []

    for run_dir in run_dirs:
        metrics_path = run_dir / "val_metrics.json"
        history_path = run_dir / "training_history.json"

        metrics = _read_json(metrics_path)
        best_acc = _safe_float(metrics.get("best_val_accuracy"))
        best_f1 = _safe_float(metrics.get("best_val_macro_f1"))
        neutral_f1 = _extract_neutral_f1(metrics)
        best_epoch = _best_epoch_from_history(history_path)

        info = _parse_run_name(run_dir.name)
        info.update(
            {
                "best_val_accuracy": f"{best_acc:.6f}" if not np.isnan(best_acc) else "",
                "best_val_macro_f1": f"{best_f1:.6f}" if not np.isnan(best_f1) else "",
                "neutral_f1": f"{neutral_f1:.6f}" if not np.isnan(neutral_f1) else "",
                "best_epoch": "" if best_epoch is None else str(best_epoch),
                "path": str(run_dir),
            }
        )
        rows.append(info)

    headers = [
        "run",
        "clean",
        "aug",
        "reg",
        "best_val_accuracy",
        "best_val_macro_f1",
        "neutral_f1",
        "best_epoch",
        "path",
    ]

    csv_path = args.out / "ablation_3_5_2_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

    # also write a compact markdown table (top-10 by macro-f1)
    def sort_key(row: Dict[str, str]) -> float:
        return _safe_float(row.get("best_val_macro_f1"), default=-1.0)

    top = sorted(rows, key=sort_key, reverse=True)[:10]
    md_path = args.out / "ablation_3_5_2_top10.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| run | clean | aug | reg | acc | macro_f1 | neutral_f1 |\n")
        f.write("|---|---|---|---:|---:|---:|---:|\n")
        for row in top:
            f.write(
                f"| {row.get('run','')} | {row.get('clean','')} | {row.get('aug','')} | {row.get('reg','')} | {row.get('best_val_accuracy','')} | {row.get('best_val_macro_f1','')} | {row.get('neutral_f1','')} |\n"
            )

    print(f"✅ summary csv: {csv_path}")
    print(f"✅ top10 md: {md_path}")


if __name__ == "__main__":
    main()
