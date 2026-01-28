# -*- coding: utf-8 -*-
"""
训练与推理管线的封装，统一对外提供函数：
1. run_train：负责训练 + 验证；
2. run_predict：利用 best checkpoint 在测试集上生成标签。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

from .constants import ID2LABEL
from .data_utils import (
    MultiModalDataset,
    build_image_transform,
    read_labeled_records,
    read_unlabeled_records,
)
from .modeling import MultiModalSentimentModel
from .utils import (
    classification_report_dict,
    ensure_dir,
    load_model_from_checkpoint,
    move_batch,
    save_checkpoint,
    save_json,
    save_predictions,
    save_ground_truth,
    set_seed,
    split_train_val,
)

try:
    from torch.amp import autocast as amp_autocast, GradScaler as AmpGradScaler

    def autocast_context(enabled: bool):
        return amp_autocast("cuda", enabled=enabled)

except (ImportError, AttributeError):
    from torch.cuda.amp import autocast as amp_autocast, GradScaler as AmpGradScaler

    def autocast_context(enabled: bool):
        return amp_autocast(enabled=enabled)


def _build_dataloaders(
    args: argparse.Namespace,
    tokenizer,
    train_transform,
    eval_transform,
    data_dir: Path,
    train_records,
    val_records,
    caption_map: Optional[Dict[str, str]] = None,
    train_sampler: Optional[WeightedRandomSampler] = None,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = MultiModalDataset(
        train_records,
        data_dir,
        tokenizer,
        args.max_length,
        train_transform,
        caption_map=caption_map,
        caption_enabled=bool(getattr(args, "use_caption", False)),
        clean_text=args.clean_text,
    )
    val_dataset = MultiModalDataset(
        val_records,
        data_dir,
        tokenizer,
        args.max_length,
        eval_transform,
        caption_map=caption_map,
        caption_enabled=bool(getattr(args, "use_caption", False)),
        return_guid=True,
        clean_text=args.clean_text,
    )

    pin_memory = args.device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], Optional[List[int]], List[str], Dict[str, float], Optional[List[float]]]:
    model.eval()
    preds: List[int] = []
    labels: List[int] = []
    guids: List[str] = []
    gates: Optional[List[float]] = [] if bool(getattr(model, "fusion_method", None) == "gmu") else None
    with torch.no_grad():
        for batch in dataloader:
            guid_batch = batch.get("guid")
            batch = move_batch(batch, device)
            if gates is not None:
                outputs = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["pixel_values"],
                    return_gates=True,
                )
                if isinstance(outputs, tuple):
                    logits, gate_mean = outputs
                else:
                    logits = outputs
                    gate_mean = None
            else:
                logits = model(batch["input_ids"], batch["attention_mask"], batch["pixel_values"])
                gate_mean = None

            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            if "labels" in batch:
                labels.extend(batch["labels"].cpu().tolist())
            if guid_batch is not None:
                guids.extend(list(guid_batch))
            if gates is not None:
                if gate_mean is None:
                    gates.extend([float("nan")] * int(pred.size(0)))
                else:
                    gates.extend(gate_mean.detach().cpu().tolist())

    metrics: Dict[str, float] = {}
    if labels:
        preds_tensor = torch.tensor(preds)
        labels_tensor = torch.tensor(labels)
        metrics["accuracy"] = float((preds_tensor == labels_tensor).float().mean().item())
        metrics["macro_f1"] = float(f1_score(labels, preds, average="macro"))
    return preds, labels or None, guids, metrics, gates


def _compute_class_counts(records) -> torch.Tensor:
    counter = Counter(rec["label"] for rec in records if rec.get("label") is not None)
    counts = [max(counter.get(label, 1), 1) for label in ID2LABEL]
    return torch.tensor(counts, dtype=torch.float32)


def _contrastive_loss(
    text_feat: torch.Tensor,
    image_feat: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if text_feat.size(0) != image_feat.size(0):
        raise ValueError("对比学习的文本与图像特征 batch 大小不一致。")
    text_norm = F.normalize(text_feat, dim=-1)
    image_norm = F.normalize(image_feat, dim=-1)
    logits_text = text_norm @ image_norm.t() / temperature
    logits_image = image_norm @ text_norm.t() / temperature
    targets = torch.arange(text_feat.size(0), device=text_feat.device)
    loss_t = F.cross_entropy(logits_text, targets)
    loss_i = F.cross_entropy(logits_image, targets)
    return 0.5 * (loss_t + loss_i)


def _apply_neutral_oversample(records, factor: int):
    if factor <= 1:
        return records
    neutral_records = [rec for rec in records if rec.get("label") == "neutral"]
    if not neutral_records:
        return records
    augmented = list(records)
    for _ in range(factor - 1):
        for rec in neutral_records:
            augmented.append({**rec})
    return augmented


def _build_balance_sampler(records) -> Optional[WeightedRandomSampler]:
    if not records:
        return None
    label_counts = Counter(rec["label"] for rec in records if rec.get("label") is not None)
    total = sum(label_counts.values())
    if total == 0:
        return None
    weights_per_label = {
        label: total / max(count, 1) for label, count in label_counts.items()
    }
    sample_weights = [
        weights_per_label.get(rec.get("label"), 1.0) for rec in records
    ]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _load_caption_map(args: argparse.Namespace) -> Optional[Dict[str, str]]:
    if not bool(getattr(args, "use_caption", False)):
        return None
    caption_path = Path(getattr(args, "caption_file", "captions.json"))
    if not caption_path.exists():
        raise FileNotFoundError(
            f"未找到 caption 文件: {caption_path}（请先生成 captions.json，或关闭 --use-caption）"
        )
    data = json.loads(caption_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("caption 文件格式错误：需要是形如 {guid: caption} 的 json 对象")
    return {str(k): str(v) for k, v in data.items()}


class WarmupCosineScheduler:
    def __init__(self, optimizer, total_steps: int, warmup_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = max(warmup_steps, 1)
        self.min_lr = min_lr
        self.current_step = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        if self.current_step >= self.total_steps:
            return
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group["lr"] = max(self.min_lr, base_lr * scale)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Optional[Dict[str, torch.Tensor]] = None
        for name, param in model.state_dict().items():
            if torch.is_floating_point(param):
                self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if not torch.is_floating_point(param):
                    continue
                assert name in self.shadow, f"EMA 缺少参数 {name}"
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def store(self, model: nn.Module) -> None:
        self.backup = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if torch.is_floating_point(param)
        }

    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            if torch.is_floating_point(param) and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            return
        for name, param in model.state_dict().items():
            if torch.is_floating_point(param) and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = None


def run_train(args: argparse.Namespace) -> None:
    if args.device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("当前脚本默认要求使用 GPU，请在具备 CUDA 的机器上运行，或指定 --allow-cpu 以继续。")

    set_seed(args.seed)
    total_start = time.perf_counter()
    ensure_dir(Path(args.output_dir))

    if bool(getattr(args, "ablate_text", False)) and bool(getattr(args, "ablate_image", False)):
        raise ValueError("ablation setting conflict: cannot enable both --ablate-text and --ablate-image")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    train_transform = build_image_transform(
        args.image_size, True, args.use_image_aug, args.use_strong_image_aug
    )
    eval_transform = build_image_transform(args.image_size, False)

    data_dir = Path(args.data_dir)
    all_records = read_labeled_records(Path(args.train_file))
    train_records, val_records = split_train_val(all_records, args.val_ratio, args.seed)
    if not train_records or not val_records:
        raise ValueError("划分出的训练/验证集为空，请适当调整 --val-ratio 参数。")

    caption_map = _load_caption_map(args)

    train_records = _apply_neutral_oversample(train_records, args.neutral_oversample)

    train_sampler = _build_balance_sampler(train_records) if args.balance_sampler else None

    train_loader, val_loader = _build_dataloaders(
        args,
        tokenizer,
        train_transform,
        eval_transform,
        data_dir,
        train_records,
        val_records,
        caption_map=caption_map,
        train_sampler=train_sampler,
    )
    save_ground_truth(val_records, Path(args.output_dir) / "val_ground_truth.csv")
    class_counts = _compute_class_counts(train_records).to(args.device)

    model = MultiModalSentimentModel(
        text_model_name=args.text_model,
        image_embed_dim=args.image_embed_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        freeze_text=args.freeze_text,
        freeze_image=args.freeze_image,
        num_labels=len(ID2LABEL),
        fusion_method=args.fusion_method,
        image_backbone=args.image_backbone,
        text_train_layers=args.text_train_layers,
        cross_attn_heads=args.cross_attn_heads,
        modality_dropout_prob=args.modality_dropout_prob,
        ablate_text=bool(getattr(args, "ablate_text", False)),
        ablate_image=bool(getattr(args, "ablate_image", False)),
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    smoothing = args.label_smoothing
    grad_accum_steps = max(1, args.grad_accum_steps)
    rdrop_alpha = args.rdrop_alpha
    steps_per_epoch = max(1, len(train_loader))
    optimizer_steps_per_epoch = max(1, math.ceil(steps_per_epoch / grad_accum_steps))

    if args.loss_type == "weighted_ce":
        class_weights = (class_counts.sum() / class_counts).to(args.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smoothing)
    elif args.loss_type == "focal":
        class_weights = (class_counts.sum() / class_counts).to(args.device)

        def focal_loss(logits, targets, gamma=args.focal_gamma):
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            num_classes = logits.size(-1)
            one_hot = F.one_hot(targets, num_classes=num_classes).float()
            if smoothing > 0:
                one_hot = one_hot * (1 - smoothing) + smoothing / num_classes
            pt = (probs * one_hot).sum(dim=1)
            focal_weights = (1 - pt) ** gamma
            ce = -(one_hot * log_probs).sum(dim=1)
            weights = class_weights.gather(0, targets)
            loss = weights * focal_weights * ce
            return loss.mean()

        criterion = focal_loss
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    use_amp = args.use_amp and args.device.type == "cuda"
    scaler = None
    if use_amp:
        try:
            scaler = AmpGradScaler(device_type="cuda", enabled=True)
        except TypeError:
            scaler = AmpGradScaler(enabled=True)

    scheduler = None
    scheduler_step_per_batch = False
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "warmup_cosine":
        total_steps = args.epochs * optimizer_steps_per_epoch
        warmup_steps = max(1, int(total_steps * args.warmup_ratio))
        scheduler = WarmupCosineScheduler(optimizer, total_steps, warmup_steps)
        scheduler_step_per_batch = True

    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []
    checkpoint_path = Path(args.checkpoint_path)

    def compute_total_loss(primary_logits, labels, secondary_logits=None):
        base_loss = criterion(primary_logits, labels)
        if rdrop_alpha <= 0 or secondary_logits is None:
            return base_loss
        second_loss = criterion(secondary_logits, labels)
        avg_loss = 0.5 * (base_loss + second_loss)
        log_prob_a = F.log_softmax(primary_logits, dim=-1)
        log_prob_b = F.log_softmax(secondary_logits, dim=-1)
        prob_a = log_prob_a.exp()
        prob_b = log_prob_b.exp()
        kl = 0.5 * (
            F.kl_div(log_prob_a, prob_b, reduction="batchmean")
            + F.kl_div(log_prob_b, prob_a, reduction="batchmean")
        )
        return avg_loss + rdrop_alpha * kl

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_items = 0
        optimizer.zero_grad()
        num_batches = len(train_loader)
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, args.device)
            labels = batch["labels"]
            with autocast_context(enabled=use_amp):
                logits = model(batch["input_ids"], batch["attention_mask"], batch["pixel_values"])
                logits_pair = (
                    model(batch["input_ids"], batch["attention_mask"], batch["pixel_values"])
                    if rdrop_alpha > 0
                    else None
                )
                loss = compute_total_loss(logits, labels, logits_pair)
            batch_size = labels.size(0)
            loss_value = loss.detach().item()
            total_loss += loss_value * batch_size
            total_items += batch_size
            loss = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            should_update = (step % grad_accum_steps == 0) or (step == num_batches)
            if should_update:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)
        train_loss = total_loss / max(1, total_items)

        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        preds, labels, guids, metrics, gates = _evaluate(model, val_loader, args.device)
        if ema is not None:
            ema.restore(model)
        val_acc = metrics.get("accuracy", 0.0)
        val_macro_f1 = metrics.get("macro_f1", 0.0)
        epoch_time = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "epoch_time_sec": epoch_time,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"训练Loss={train_loss:.4f}  验证准确率={val_acc:.4f}  "
            f"Macro-F1={val_macro_f1:.4f}  耗时={epoch_time:.1f}s"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                model,
                checkpoint_path,
                config={
                    "text_model": args.text_model,
                    "image_embed_dim": args.image_embed_dim,
                    "fusion_hidden_dim": args.fusion_hidden_dim,
                    "dropout": args.dropout,
                    "fusion_method": args.fusion_method,
                    "image_backbone": args.image_backbone,
                    "text_train_layers": args.text_train_layers,
                    "cross_attn_heads": args.cross_attn_heads,
                    "modality_dropout_prob": args.modality_dropout_prob,
                    "use_caption": bool(getattr(args, "use_caption", False)),
                    "caption_file": str(getattr(args, "caption_file", "captions.json")),
                    "ablate_text": bool(getattr(args, "ablate_text", False)),
                    "ablate_image": bool(getattr(args, "ablate_image", False)),
                },
            )
            save_predictions(guids, preds, Path(args.output_dir) / "val_predictions.csv")
            if gates is not None:
                gate_path = Path(args.output_dir) / "val_gates.csv"
                with gate_path.open("w", encoding="utf-8") as f:
                    f.write("guid,true_tag,pred_tag,gate_text\n")
                    if labels is None:
                        for guid, pred_id, gate_value in zip(guids, preds, gates):
                            f.write(f"{guid},,{ID2LABEL[pred_id]},{gate_value:.6f}\n")
                    else:
                        for guid, true_id, pred_id, gate_value in zip(guids, labels, preds, gates):
                            f.write(
                                f"{guid},{ID2LABEL[true_id]},{ID2LABEL[pred_id]},{gate_value:.6f}\n"
                            )
            if labels is not None:
                metrics_report = classification_report_dict(labels, preds)
                save_json(
                    {
                        "best_val_accuracy": best_acc,
                        "best_val_macro_f1": val_macro_f1,
                        "classification_report": metrics_report,
                    },
                    Path(args.output_dir) / "val_metrics.json",
                )
            if ema is not None:
                ema.store(model)
                ema.copy_to(model)
                save_checkpoint(
                    model,
                    checkpoint_path,
                    config={
                        "text_model": args.text_model,
                        "image_embed_dim": args.image_embed_dim,
                        "fusion_hidden_dim": args.fusion_hidden_dim,
                        "dropout": args.dropout,
                        "fusion_method": args.fusion_method,
                        "image_backbone": args.image_backbone,
                        "text_train_layers": args.text_train_layers,
                        "cross_attn_heads": args.cross_attn_heads,
                        "modality_dropout_prob": args.modality_dropout_prob,
                        "use_caption": bool(getattr(args, "use_caption", False)),
                        "caption_file": str(getattr(args, "caption_file", "captions.json")),
                        "ablate_text": bool(getattr(args, "ablate_text", False)),
                        "ablate_image": bool(getattr(args, "ablate_image", False)),
                    },
                )
                ema.restore(model)
        else:
            epochs_no_improve += 1

        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(
                f"验证集在 {args.patience} 个 epoch 内未提升（当前最佳 {best_acc:.4f} @ epoch {best_epoch}），提前停止训练。"
            )
            break

    save_json(history, Path(args.output_dir) / "training_history.json")
    total_time = time.perf_counter() - total_start
    print(
        f"训练完成，总耗时 {total_time/60:.1f} 分钟，最佳验证准确率：{best_acc:.4f}，模型保存在 {checkpoint_path}"
    )


def run_predict(args: argparse.Namespace) -> None:
    if args.device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("当前脚本默认要求使用 GPU，请在具备 CUDA 的机器上运行，或指定 --allow-cpu 以继续。")

    model, config = load_model_from_checkpoint(Path(args.checkpoint_path), args.device)
    if (not bool(getattr(args, "use_caption", False))) and bool(config.get("use_caption", False)):
        args.use_caption = True
        args.caption_file = Path(config.get("caption_file", getattr(args, "caption_file", "captions.json")))
    tokenizer = AutoTokenizer.from_pretrained(config["text_model"])
    image_transform = build_image_transform(args.image_size, False)

    data_dir = Path(args.data_dir)
    records = read_unlabeled_records(Path(args.test_file))
    caption_map = _load_caption_map(args)
    dataset = MultiModalDataset(
        records,
        data_dir,
        tokenizer,
        args.max_length,
        image_transform,
        caption_map=caption_map,
        caption_enabled=bool(getattr(args, "use_caption", False)),
        return_guid=True,
        clean_text=args.clean_text,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.type == "cuda",
    )

    preds, _, guids, _, gates = _evaluate(model, dataloader, args.device)
    set_seed(args.seed)
    ensure_dir(Path(args.output_dir))

    if bool(getattr(args, "ablate_text", False)) and bool(getattr(args, "ablate_image", False)):
        raise ValueError("ablation setting conflict: cannot enable both --ablate-text and --ablate-image")
    save_predictions(guids, preds, Path(args.output_dir) / "test_predictions.csv")
    if gates is not None:
        gate_path = Path(args.output_dir) / "test_gates.csv"
        with gate_path.open("w", encoding="utf-8") as f:
            f.write("guid,pred_tag,gate_text\n")
            for guid, pred_id, gate_value in zip(guids, preds, gates):
                f.write(f"{guid},{ID2LABEL[pred_id]},{gate_value:.6f}\n")
    print(f"测试集预测已输出到 {Path(args.output_dir) / 'test_predictions.csv'}")
