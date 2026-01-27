#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态情感分类主脚本
--------------------
提供 train / predict 两种模式，均默认要求使用 GPU 运行。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.trainer import run_predict, run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验五：多模态情感分类训练与推理脚本（默认使用 GPU）"
    )
    parser.add_argument("--mode", choices=["train", "predict"], default="train", help="选择 train 或 predict 模式")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="存放图像与文本的 data 目录")
    parser.add_argument("--train-file", type=Path, default=Path("train.txt"), help="训练集标签文件")
    parser.add_argument("--test-file", type=Path, default=Path("test_without_label.txt"), help="测试集 guid 列表文件")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="日志与预测结果输出目录")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("outputs/best_model.pt"), help="模型权重保存/读取路径")
    parser.add_argument(
        "--use-caption",
        action="store_true",
        help="启用图像 Caption 拼接：最终输入文本为 [原始文本] [SEP] [BLIP caption]",
    )
    parser.add_argument(
        "--caption-file",
        type=Path,
        default=Path("captions.json"),
        help="Caption 映射文件（json），形如 {guid: caption}；配合 --use-caption 使用",
    )
    parser.add_argument("--text-model", type=str, default="roberta-base", help="文本编码使用的 HuggingFace 模型名称")
    parser.add_argument("--max-length", type=int, default=128, help="文本序列最大长度")
    parser.add_argument("--image-size", type=int, default=224, help="图像缩放尺寸（边长）")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮次")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW 的 weight decay")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 线程数（Windows 建议保持 0）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="从训练集中划分验证集的比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    parser.add_argument(
        "--image-backbone",
        choices=["resnet18", "resnet50"],
        default="resnet18",
        help="图像编码骨干网络，ResNet50 对细粒度表情更敏感",
    )
    parser.add_argument("--image-embed-dim", type=int, default=256, help="图像特征投影维度")
    parser.add_argument("--fusion-hidden-dim", type=int, default=512, help="多模态融合后的隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.2, help="分类器 dropout 概率")
    parser.add_argument("--freeze-text", action="store_true", help="是否冻结文本编码器参数")
    parser.add_argument(
        "--text-train-layers",
        type=int,
        default=-1,
        help="仅训练文本编码器最后 N 层（-1 表示全部训练），可降低算力消耗",
    )
    parser.add_argument("--freeze-image", action="store_true", help="是否冻结图像编码器参数")
    parser.add_argument("--use-amp", action="store_true", help="是否使用混合精度（AMP）训练")
    parser.add_argument("--use-image-aug", action="store_true", help="训练阶段开启基础图像增强")
    parser.add_argument(
        "--use-strong-image-aug",
        action="store_true",
        help="在基础图像增强上叠加 RandAugment，进一步提升鲁棒性",
    )
    parser.add_argument("--clean-text", action="store_true", help="加载文本时执行 URL/标签清洗")
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine", "warmup_cosine"],
        default="warmup_cosine",
        help="学习率调度器类型（warmup_cosine 为分段 Warmup + 余弦）",
    )
    parser.add_argument(
        "--fusion-method",
        choices=["concat", "gmu", "cross_attn", "co_attn"],
        default="concat",
        help="多模态融合策略：拼接/GMU/跨模态注意力",
    )
    parser.add_argument("--cross-attn-heads", type=int, default=4, help="cross_attn 融合时 MultiheadAttention 的头数")
    parser.add_argument(
        "--modality-dropout-prob",
        type=float,
        default=0.1,
        help="训练阶段以该概率随机屏蔽单一模态特征，提升鲁棒性",
    )
    parser.add_argument(
        "--ablate-text",
        action="store_true",
        help="消融：训练/验证时屏蔽文本模态，仅使用图像模态（image-only）",
    )
    parser.add_argument(
        "--ablate-image",
        action="store_true",
        help="消融：训练/验证时屏蔽图像模态，仅使用文本模态（text-only）",
    )
    parser.add_argument("--balance-sampler", action="store_true", help="训练 DataLoader 使用类别均衡采样")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="梯度累积步数，可在显存受限时提升等效 batch size（>=1）",
    )
    parser.add_argument(
        "--loss-type",
        choices=["ce", "weighted_ce", "focal"],
        default="ce",
        help="损失函数：普通交叉熵、类别重加权交叉熵或 Focal Loss",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal Loss 的 γ 参数")
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.0,
        help=">0 时启用跨模态对比学习损失，建议 0.1~0.5",
    )
    parser.add_argument(
        "--contrastive-temp",
        type=float,
        default=0.07,
        help="对比学习温度系数（越小越强调困难负样本）",
    )
    parser.add_argument(
        "--neutral-oversample",
        type=int,
        default=1,
        help="对 neutral 样本进行过采样的倍数（>1 时复制样本）",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="warmup_cosine 中 Warmup 步数占总步数的比例")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--patience", type=int, default=0, help="验证集 Accuracy 在若干个 epoch 无提升则提前停止训练（0 表示关闭）")
    parser.add_argument("--ema-decay", type=float, default=0.0, help=">0 时启用模型参数 EMA（如 0.999），提升鲁棒性")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="交叉熵标签平滑系数（0 表示关闭，为 0.05~0.1 可有效抑制过拟合）",
    )
    parser.add_argument(
        "--rdrop-alpha",
        type=float,
        default=0.0,
        help=">0 时启用 R-Drop 正则项，加强多模态表示一致性（建议取 0.5~1.0）",
    )
    parser.add_argument("--allow-cpu", action="store_true", help="万一没有 GPU，可加此参数在 CPU 上强行运行（不推荐）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    if device.type != "cuda":
        print("⚠️ 当前未检测到 GPU。如果你确认要在 CPU 上运行，请添加 --allow-cpu 参数。")
    if args.mode == "train":
        run_train(args)
    else:
        run_predict(args)


if __name__ == "__main__":
    main()
