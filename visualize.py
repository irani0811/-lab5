#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化脚本
---------
1. 绘制训练过程（loss / val acc）曲线；
2. 基于验证集标签与预测绘制混淆矩阵；
3. 输出文本摘要（分类报告等）协助撰写实验报告。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from src.constants import ID2LABEL
from src.utils import ensure_dir


def load_history(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"未找到训练日志: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_training_curves(history: Sequence[dict], output_dir: Path) -> Path:
    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_accs = [item["val_accuracy"] for item in history]
    val_macro_f1 = [item.get("val_macro_f1") for item in history]
    epoch_times = [item.get("epoch_time_sec") for item in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(epochs, train_losses, marker="o", color="#1f77b4")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(epochs, val_accs, marker="o", color="#ff7f0e", label="Accuracy")
    axes[1].plot(epochs, val_macro_f1, marker="s", color="#2ca02c", label="Macro-F1")
    axes[1].set_title("Validation Accuracy & Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    axes[2].bar(epochs, epoch_times, color="#9467bd")
    axes[2].set_title("Epoch Time (s)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Seconds")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.4)

    if any(epoch_times):
        for ax in axes:
            ax2 = ax.twinx()
            ax2.set_yticks([])
        axes[0].legend(["Train Loss"])
        axes[1].legend(["Val Acc"])

    curves_path = output_dir / "training_curves.png"
    fig.suptitle("Training Progress", fontsize=14)
    fig.tight_layout()
    fig.savefig(curves_path, dpi=200)
    plt.close(fig)
    return curves_path


def load_csv_labels(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"未找到 CSV 文件: {path}")
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["tag"])
    return labels


def plot_confusion(y_true: Sequence[str], y_pred: Sequence[str], output_dir: Path) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=ID2LABEL)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ID2LABEL)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Validation Confusion Matrix")
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    return cm_path


def export_text_report(y_true: Sequence[str], y_pred: Sequence[str], output_dir: Path) -> Path:
    report = classification_report(y_true, y_pred, labels=ID2LABEL, digits=4)
    text_path = output_dir / "visualization_summary.txt"
    text_path.write_text(report, encoding="utf-8")
    return text_path


def plot_classwise_bars(y_true: Sequence[str], y_pred: Sequence[str], output_dir: Path) -> Path:
    report = classification_report(y_true, y_pred, labels=ID2LABEL, output_dict=True, zero_division=0)
    metrics = ["precision", "recall", "f1-score"]
    values = {metric: [report[label][metric] for label in ID2LABEL] for metric in metrics}

    x = np.arange(len(ID2LABEL))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, values[metric], width, label=metric.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels([label.title() for label in ID2LABEL])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Class-wise Precision / Recall / F1")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    bar_path = output_dir / "classwise_metrics.png"
    fig.savefig(bar_path, dpi=200)
    plt.close(fig)
    return bar_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成训练曲线与混淆矩阵可视化")
    parser.add_argument("--history", type=Path, default=Path("outputs/training_history.json"))
    parser.add_argument("--val-ground-truth", type=Path, default=Path("outputs/val_ground_truth.csv"))
    parser.add_argument("--val-predictions", type=Path, default=Path("outputs/val_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    history = load_history(args.history)
    curves_path = plot_training_curves(history, args.output_dir)
    print(f"✅ 训练曲线已保存：{curves_path}")

    y_true = load_csv_labels(args.val_ground_truth)
    y_pred = load_csv_labels(args.val_predictions)
    if len(y_true) != len(y_pred):
        raise ValueError("验证集标签与预测数量不一致，确认是否使用相同划分。")

    cm_path = plot_confusion(y_true, y_pred, args.output_dir)
    bars_path = plot_classwise_bars(y_true, y_pred, args.output_dir)
    summary_path = export_text_report(y_true, y_pred, args.output_dir)
    print(f"✅ 混淆矩阵已保存：{cm_path}")
    print(f"✅ 类别柱状图已保存：{bars_path}")
    print(f"✅ 文本报告已保存：{summary_path}")


if __name__ == "__main__":
    main()
