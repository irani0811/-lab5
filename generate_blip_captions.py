#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


def _read_guids_from_txt(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")
    guids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        header_skipped = False
        for line in f:
            if not header_skipped:
                header_skipped = True
                continue
            line = line.strip()
            if not line:
                continue
            guid = line.split(",")[0].strip()
            if guid:
                guids.add(guid)
    return guids


def _safe_open_image(path: Path) -> Optional[Image.Image]:
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 BLIP 为图像生成 caption，并输出 captions.json")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="图像目录（guid.jpg）")
    parser.add_argument("--train-file", type=Path, default=Path("train.txt"), help="训练集 guid 文件")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("test_without_label.txt"),
        help="测试集 guid 文件（可选）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("captions.json"),
        help="输出 caption 映射文件（json），形如 {guid: caption}",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Salesforce/blip-image-captioning-base",
        help="HuggingFace BLIP 模型名称",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="生成 caption 的 batch size")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="caption 最大生成 token 数")
    parser.add_argument("--allow-cpu", action="store_true", help="若无 GPU，可允许使用 CPU（会很慢）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("当前脚本默认要求使用 GPU，请在具备 CUDA 的机器上运行，或指定 --allow-cpu 以继续。")

    data_dir = Path(args.data_dir)
    guid_set: Set[str] = set()
    guid_set |= _read_guids_from_txt(Path(args.train_file))
    if Path(args.test_file).exists():
        guid_set |= _read_guids_from_txt(Path(args.test_file))

    guids = sorted(guid_set, key=lambda x: int(x) if x.isdigit() else x)

    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    captions: Dict[str, str] = {}
    batch_images: List[Image.Image] = []
    batch_guids: List[str] = []

    def flush_batch() -> None:
        if not batch_images:
            return
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        texts = processor.batch_decode(out_ids, skip_special_tokens=True)
        for g, t in zip(batch_guids, texts):
            captions[str(g)] = (t or "").strip()
        batch_images.clear()
        batch_guids.clear()

    missing = 0
    for idx, guid in enumerate(guids, start=1):
        img_path = data_dir / f"{guid}.jpg"
        img = _safe_open_image(img_path)
        if img is None:
            missing += 1
            continue
        batch_images.append(img)
        batch_guids.append(guid)
        if len(batch_images) >= max(1, args.batch_size):
            flush_batch()
        if idx % 200 == 0:
            print(f"已处理 {idx}/{len(guids)}，当前生成 {len(captions)} 条，缺失/损坏 {missing} 张")

    flush_batch()

    args.output.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ captions 已保存：{args.output}（共 {len(captions)} 条，缺失/损坏 {missing} 张）")


if __name__ == "__main__":
    main()
