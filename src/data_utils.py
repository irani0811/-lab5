# -*- coding: utf-8 -*-
"""
数据读取与数据集定义模块。
封装文本/图像的加载逻辑，方便在训练与推理阶段复用。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import PreTrainedTokenizerBase

from .constants import LABEL2ID


def read_labeled_records(csv_path: Path) -> List[Dict[str, str]]:
    """读取带标签的 guid 列表。"""
    records: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        header_skipped = False
        for line in f:
            if not header_skipped:
                header_skipped = True
                continue
            line = line.strip()
            if not line:
                continue
            guid, label = line.split(",")
            records.append({"guid": guid, "label": label.lower()})
    return records


def read_unlabeled_records(csv_path: Path) -> List[Dict[str, Optional[str]]]:
    """读取测试集 guid 列表。"""
    records: List[Dict[str, Optional[str]]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        header_skipped = False
        for line in f:
            if not header_skipped:
                header_skipped = True
                continue
            line = line.strip()
            if not line:
                continue
            guid, _ = line.split(",")
            records.append({"guid": guid, "label": None})
    return records


def build_image_transform(
    image_size: int, is_train: bool, use_aug: bool = False, use_strong_aug: bool = False
) -> transforms.Compose:
    """构建图像处理流程。

    Args:
        image_size: 最终图像尺寸。
        is_train: 是否用于训练集。
        use_aug: 训练阶段是否启用强化增强。
    """
    if is_train:
        aug_blocks = [transforms.Resize((image_size, image_size))]
        if use_aug:
            aug_blocks = [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
            if use_strong_aug:
                aug_blocks.append(transforms.RandAugment(num_ops=2, magnitude=9))
        extra_aug = []
        if use_aug:
            extra_aug = [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)], p=0.2
                ),
            ]
        pipeline = aug_blocks + extra_aug
    else:
        pipeline = [transforms.Resize((image_size, image_size))]

    pipeline += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(pipeline)


_URL_PATTERN = re.compile(r"http[s]?://\S+")
_TAG_PATTERN = re.compile(r"[@#][\w-]+")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text_content(text: str) -> str:
    """简单文本清洗：去除 URL、@/# 标签与多余空格。"""
    text = _URL_PATTERN.sub(" ", text)
    text = _TAG_PATTERN.sub(" ", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text)
    return text.strip()


def safe_read_text(path: Path) -> str:
    """读取文本并兜底为空字符串情况。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到文本文件: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return text if text else "[PAD]"


def safe_read_image(path: Path) -> Image.Image:
    """打开图像并转为 RGB。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到图像文件: {path}")
    with Image.open(path) as img:
        return img.convert("RGB")


class MultiModalDataset(Dataset):
    """同时返回文本、图像以及标签的 Dataset 实现。"""

    def __init__(
        self,
        records: Sequence[Dict[str, Optional[str]]],
        data_dir: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        image_transform: transforms.Compose,
        caption_map: Optional[Dict[str, str]] = None,
        caption_enabled: bool = False,
        return_guid: bool = False,
        clean_text: bool = False,
    ) -> None:
        self.records = list(records)
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = image_transform
        self.caption_map = caption_map or {}
        self.caption_enabled = caption_enabled
        self.return_guid = return_guid
        self.clean_text = clean_text

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        guid = record["guid"]
        text_path = self.data_dir / f"{guid}.txt"
        img_path = self.data_dir / f"{guid}.jpg"

        text = safe_read_text(text_path)
        if self.clean_text:
            text = clean_text_content(text)
        if self.caption_enabled:
            caption = self.caption_map.get(str(guid)) or self.caption_map.get(guid)
            if caption:
                sep = getattr(self.tokenizer, "sep_token", None) or "[SEP]"
                if self.clean_text:
                    caption = clean_text_content(caption)
                text = f"{text} {sep} {caption}"
        image = safe_read_image(img_path)

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": self.image_transform(image),
        }

        label_name = record.get("label")
        if label_name is not None:
            item["labels"] = torch.tensor(LABEL2ID[label_name], dtype=torch.long)
        if self.return_guid:
            item["guid"] = guid
        return item
