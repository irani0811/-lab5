# -*- coding: utf-8 -*-
"""
通用工具函数：
1. 随机数种子固定，保证实验可复现；
2. 训练过程中的张量搬运、指标计算；
3. 模型权重与预测结果的保存、加载。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from .constants import ID2LABEL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_train_val(
    records: Sequence[Dict[str, str]], val_ratio: float, seed: int
) -> tuple[list[Dict[str, str]], list[Dict[str, str]]]:
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    val_size = max(1, int(len(records) * val_ratio))
    val_records = records[:val_size]
    train_records = records[val_size:]
    return train_records, val_records


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def classification_report_dict(labels: List[int], preds: List[int]) -> Dict[str, Dict[str, float]]:
    report = classification_report(
        labels,
        preds,
        target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))],
        output_dict=True,
        zero_division=0,
    )
    for key, value in report.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, (np.floating, np.integer)):
                    report[key][inner_key] = float(inner_value)
    return report


def save_predictions(guids: Iterable[str], preds: Iterable[int], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("guid,tag\n")
        for guid, pred in zip(guids, preds):
            f.write(f"{guid},{ID2LABEL[pred]}\n")


def save_ground_truth(records: Sequence[Dict[str, str]], output_path: Path) -> None:
    """将验证集 guid-label 对写入 CSV，便于后续可视化/分析。"""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("guid,tag\n")
        for item in records:
            f.write(f"{item['guid']},{item['label']}\n")


def save_json(data: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_checkpoint(model: torch.nn.Module, path: Path, config: Dict[str, object]) -> None:
    package = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    torch.save(package, path)


def load_model_from_checkpoint(path: Path, device: torch.device):
    """加载权重并返回 (model, config)。"""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    from .modeling import MultiModalSentimentModel

    model = MultiModalSentimentModel(
        text_model_name=config["text_model"],
        image_embed_dim=config["image_embed_dim"],
        fusion_hidden_dim=config["fusion_hidden_dim"],
        dropout=config["dropout"],
        freeze_text=False,
        freeze_image=False,
        num_labels=len(ID2LABEL),
        fusion_method=config.get("fusion_method", "concat"),
        image_backbone=config.get("image_backbone", "resnet18"),
        cross_attn_heads=config.get("cross_attn_heads", 4),
        modality_dropout_prob=config.get("modality_dropout_prob", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, config
