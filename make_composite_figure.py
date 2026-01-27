# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _safe_read_json(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _discover_runs(project_root: Path) -> List[Path]:
    candidates: List[Path] = []

    # outputs (no suffix)
    p = project_root / "outputs"
    if p.is_dir():
        candidates.append(p)

    # outputs_* folders
    for item in sorted(project_root.glob("outputs_*")):
        if item.is_dir():
            candidates.append(item)

    # filter runs with metrics
    runs = [p for p in candidates if (p / "val_metrics.json").exists() and (p / "training_history.json").exists()]
    return runs


def _count_parameters_from_checkpoint(checkpoint_path: Path) -> Optional[int]:
    if not checkpoint_path.exists():
        return None

    try:
        import torch

        from src.utils import load_model_from_checkpoint

        device = torch.device("cpu")
        model, _ = load_model_from_checkpoint(checkpoint_path, device)
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


def _extract_metrics(run_dir: Path) -> Dict[str, object]:
    metrics_path = run_dir / "val_metrics.json"
    history_path = run_dir / "training_history.json"

    metrics = _safe_read_json(metrics_path) or {}
    history = _safe_read_json(history_path) or []

    best_acc = float(metrics.get("best_val_accuracy", float("nan")))
    best_macro_f1 = float(metrics.get("best_val_macro_f1", float("nan")))

    report = metrics.get("classification_report", {}) if isinstance(metrics.get("classification_report"), dict) else {}

    def _get_prf(label: str) -> Tuple[float, float, float]:
        item = report.get(label, {}) if isinstance(report.get(label), dict) else {}
        return (
            float(item.get("precision", float("nan"))),
            float(item.get("recall", float("nan"))),
            float(item.get("f1-score", float("nan"))),
        )

    neg_p, neg_r, neg_f1 = _get_prf("negative")
    neu_p, neu_r, neu_f1 = _get_prf("neutral")
    pos_p, pos_r, pos_f1 = _get_prf("positive")

    epochs = len(history) if isinstance(history, list) else 0
    epoch_times = [float(x.get("epoch_time_sec", 0.0)) for x in history if isinstance(x, dict)] if isinstance(history, list) else []
    total_time_sec = float(sum(epoch_times)) if epoch_times else float("nan")
    avg_time_sec = float(np.mean(epoch_times)) if epoch_times else float("nan")

    val_macro_curve = [float(x.get("val_macro_f1", float("nan"))) for x in history if isinstance(x, dict)] if isinstance(history, list) else []

    checkpoint = run_dir / "best_model.pt"
    params = _count_parameters_from_checkpoint(checkpoint)

    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "best_acc": best_acc,
        "best_macro_f1": best_macro_f1,
        "neg_p": neg_p,
        "neg_r": neg_r,
        "neg_f1": neg_f1,
        "neu_p": neu_p,
        "neu_r": neu_r,
        "neu_f1": neu_f1,
        "pos_p": pos_p,
        "pos_r": pos_r,
        "pos_f1": pos_f1,
        "epochs": epochs,
        "total_time_sec": total_time_sec,
        "avg_time_sec": avg_time_sec,
        "val_macro_curve": val_macro_curve,
        "params": params,
    }


def _minmax_norm(values: List[float], invert: bool = False) -> List[float]:
    clean = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    if not clean:
        return [0.0 for _ in values]

    vmin = min(clean)
    vmax = max(clean)
    if abs(vmax - vmin) < 1e-12:
        out = [1.0 if not (math.isnan(v) or math.isinf(v)) else 0.0 for v in values]
        return [1.0 - x if invert else x for x in out]

    out: List[float] = []
    for v in values:
        if math.isnan(v) or math.isinf(v):
            out.append(0.0)
        else:
            out.append((v - vmin) / (vmax - vmin))

    if invert:
        out = [1.0 - x for x in out]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run directories to include (e.g. outputs_baseline outputs_blip). If omitted, auto-discover outputs_*.",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent),
        help="Project root containing outputs_* folders.",
    )
    parser.add_argument(
        "--out",
        default="outputs/composite_figure.png",
        help="Output image path (relative to project root if not absolute).",
    )
    parser.add_argument(
        "--radar-metrics",
        default="acc,macro_f1,neutral_f1,time,params",
        help="Comma-separated metrics for radar: acc,macro_f1,neutral_f1,time,params",
    )
    parser.add_argument(
        "--max-curves",
        type=int,
        default=6,
        help="Max number of runs to plot in convergence subplot.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
        run_dirs = [d if d.is_absolute() else (project_root / d) for d in run_dirs]
        run_dirs = [d for d in run_dirs if d.is_dir()]
    else:
        run_dirs = _discover_runs(project_root)

    if not run_dirs:
        raise SystemExit("No valid runs found. Need val_metrics.json and training_history.json in outputs folders.")

    rows = [_extract_metrics(d) for d in run_dirs]

    radar_keys = [x.strip() for x in str(args.radar_metrics).split(",") if x.strip()]

    # Build radar matrix
    metric_values: Dict[str, List[float]] = {}

    def _as_float_list(key: str) -> List[float]:
        out: List[float] = []
        for r in rows:
            v = r.get(key)
            if v is None:
                out.append(float("nan"))
            elif isinstance(v, (int, float)):
                out.append(float(v))
            elif isinstance(v, str):
                try:
                    out.append(float(v))
                except Exception:
                    out.append(float("nan"))
            else:
                out.append(float("nan"))
        return out

    if "acc" in radar_keys:
        metric_values["Acc"] = _as_float_list("best_acc")
    if "macro_f1" in radar_keys:
        metric_values["Macro-F1"] = _as_float_list("best_macro_f1")
    if "neutral_f1" in radar_keys:
        metric_values["Neutral-F1"] = _as_float_list("neu_f1")
    if "time" in radar_keys:
        metric_values["Time"] = _as_float_list("total_time_sec")
    if "params" in radar_keys:
        # None -> nan
        vals: List[float] = []
        for r in rows:
            p = r.get("params")
            vals.append(float(p) if isinstance(p, int) else float("nan"))
        metric_values["Params"] = vals

    radar_labels = list(metric_values.keys())
    radar_raw = [metric_values[k] for k in radar_labels]

    # Normalize each axis to [0,1]
    radar_norm: List[List[float]] = []
    for label, values in zip(radar_labels, radar_raw):
        invert = label in {"Time", "Params"}
        radar_norm.append(_minmax_norm(values, invert=invert))

    radar_norm = np.array(radar_norm)  # (axes, runs)

    # Build time/params summary
    total_times = [float(r.get("total_time_sec", float("nan"))) for r in rows]
    params = [float(r.get("params")) if isinstance(r.get("params"), int) else float("nan") for r in rows]

    # Convergence curves
    curves = [(r["run"], r.get("val_macro_curve", [])) for r in rows]
    curves = sorted(curves, key=lambda x: len(x[1]), reverse=True)
    curves = curves[: max(1, int(args.max_curves))]

    # Plot
    fig = plt.figure(figsize=(18, 5.2), dpi=160)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25])

    # (a) Radar
    ax0 = fig.add_subplot(gs[0, 0], polar=True)
    n_axes = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    ax0.set_theta_offset(np.pi / 2)
    ax0.set_theta_direction(-1)
    ax0.set_thetagrids(np.degrees(angles[:-1]), radar_labels)
    ax0.set_ylim(0, 1)
    ax0.set_title("(a) Normalized Comprehensive Performance", pad=18)

    for i, r in enumerate(rows):
        vals = radar_norm[:, i].tolist()
        vals += vals[:1]
        ax0.plot(angles, vals, linewidth=1.8, label=r["run"])
        ax0.fill(angles, vals, alpha=0.08)

    ax0.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=8)

    # (b) Params vs Training Time
    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(rows))

    bar_vals = np.array([p if not (math.isnan(p) or math.isinf(p)) else 0.0 for p in params], dtype=float)
    bars = ax1.bar(x, bar_vals / 1e6, color="#4C78A8", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r["run"] for r in rows], rotation=25, ha="right")
    ax1.set_ylabel("Parameters (M)")
    ax1.set_title("(b) Parameters vs Training Time")

    for b, v in zip(bars, params):
        if isinstance(v, float) and not (math.isnan(v) or math.isinf(v)):
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v/1e6:.1f}", ha="center", va="bottom", fontsize=8)

    ax1b = ax1.twinx()
    time_vals = np.array([t if not (math.isnan(t) or math.isinf(t)) else 0.0 for t in total_times], dtype=float)
    ax1b.plot(x, time_vals / 60.0, color="#F58518", marker="o", linewidth=2)
    ax1b.set_ylabel("Training Time (min)")

    for xi, t in zip(x, total_times):
        if isinstance(t, float) and not (math.isnan(t) or math.isinf(t)):
            ax1b.text(xi, (t / 60.0), f"{t/60.0:.1f}", color="#F58518", fontsize=8, ha="left", va="bottom")

    # (c) Convergence
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("(c) Validation Macro-F1 Convergence")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Macro-F1")

    for name, curve in curves:
        if not curve:
            continue
        y = np.array(curve, dtype=float)
        x2 = np.arange(1, len(y) + 1)
        ax2.plot(x2, y, linewidth=2, label=name)
        best_idx = int(np.nanargmax(y)) if np.any(~np.isnan(y)) else None
        if best_idx is not None:
            ax2.scatter([best_idx + 1], [y[best_idx]], s=28)

    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)

    fig.tight_layout()

    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else (project_root / out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
