# 实验五：多模态情感分类

本项目实现文本 + 图像情感识别流程。
核心思想：

- 文本模态：复用 HuggingFace 预训练 Transformer（默认 `roberta-base`）。
- 图像模态：使用 ImageNet 预训练的 `ResNet18/ResNet50`，并通过线性层投影到统一嵌入维度。
- 融合策略：支持 `concat / gmu / cross_attn / co_attn` 多种融合结构，便于做公平对比实验。
- Caption 增强：可选启用 BLIP caption 拼接（输入变为 `[原始文本] [SEP] [caption]`）。
- 消融实验：支持 `text-only / image-only`（通过屏蔽某一模态特征）。
- 训练/推理脚本默认要求 **GPU**，如确实没有 GPU，可额外加 `--allow-cpu` 参数强行运行（速度较慢）。

## 目录结构

```
project5/
├── data/                  # 提供的图像与文本对（guid.jpg / guid.txt）
├── train.txt              # 训练集 guid,tag
├── test_without_label.txt # 测试集 guid,null
├── src/
│   ├── __init__.py
│   ├── constants.py       # 标签映射常量
│   ├── data_utils.py      # 数据读取 & Dataset
│   ├── modeling.py        # 模型结构
│   └── trainer.py         # 训练/推理流程
├── train_multimodal.py    # 主入口脚本（命令行）
├── requirements.txt       # 环境依赖
└── README.md              # 使用说明
```

## 环境准备

1. **Python**：建议 3.9+
2. **依赖安装**：
   ```bash
   pip install -r requirements.txt
   ```
3. **显卡驱动 / CUDA**：需确保本地 PyTorch 可识别 GPU（`torch.cuda.is_available()` 为 True）。

## 训练流程

```bash
python train_multimodal.py \
  --mode train \
  --data-dir data \
  --train-file train.txt \
  --output-dir outputs \
  --checkpoint-path outputs/best_model.pt \
  --text-model roberta-base \
  --batch-size 16 \
  --epochs 6 \
  --use-amp \
  --use-image-aug \
  --scheduler warmup_cosine \
  --fusion-method co_attn
```

关键参数：
- `--val-ratio`：自动从 train.txt 划分验证集（默认 0.2）。
- `--freeze-text/--freeze-image`：可按需冻结某一模态的编码器。
- `--image-embed-dim` / `--fusion-hidden-dim` / `--dropout`：控制融合策略容量。
- `--use-image-aug`：训练阶段启用随机裁剪 / 翻转 / ColorJitter / 模糊，缓解过拟合。
- `--use-strong-image-aug`：在基础增强上叠加 RandAugment。
- `--scheduler`：`none / cosine / warmup_cosine`（默认 warmup_cosine）。
- `--fusion-method`：`concat / gmu / cross_attn / co_attn`。
- `--image-backbone`：`resnet18 / resnet50`。
- `--use-caption` + `--caption-file`：启用 caption 拼接（caption 映射为 `{guid: caption}` 的 json 文件）。
- `--ablate-image`：消融（text-only），屏蔽图像模态。
- `--ablate-text`：消融（image-only），屏蔽文本模态。

Caption 文件格式说明：

```json
{
  "<guid>": "a short caption about the image",
  "<guid2>": "..."
}
```

训练结束后会在 `outputs/` 下生成：
- `best_model.pt`：最佳验证准确率对应的模型权重。
- `val_predictions.csv`、`val_metrics.json`、`training_history.json`：用于报告撰写。

建议固定对比设置以保证公平性：

```bash
python train_multimodal.py --mode train --data-dir data --train-file train.txt --epochs 6 --seed 42 --val-ratio 0.2 --batch-size 16 --use-amp
```

在此基础上仅切换一个因素（如 `--fusion-method` 或 `--use-caption`）完成对照实验。

Windows 提示：如你在 PowerShell/CMD 中运行，建议将命令写成一行，并加 `--num-workers 0`。

## 测试集推理

```bash
python train_multimodal.py \
  --mode predict \
  --data-dir data \
  --test-file test_without_label.txt \
  --checkpoint-path outputs/best_model.pt \
  --output-dir outputs
```

结果文件：`outputs/test_predictions.csv`。该文件会包含每条测试样本的 `guid` 与预测标签（`negative/neutral/positive`），可直接用于提交。

## 可视化分析

训练完成后会自动在 `outputs/` 目录下生成：
- `training_history.json`：每个 epoch 的 loss、验证准确率、耗时；
- `val_ground_truth.csv`：验证集 guid-label；
- `val_predictions.csv`：模型在验证集上的预测。

执行下列命令即可生成训练曲线与混淆矩阵（默认输出至 `outputs/figures/`）：

```bash
python visualize.py \
  --history outputs/training_history.json \
  --val-ground-truth outputs/val_ground_truth.csv \
  --val-predictions outputs/val_predictions.csv \
  --output-dir outputs/figures
```

脚本会输出：`training_curves.png`、`confusion_matrix.png` 以及 `visualization_summary.txt`，可直接放入报告。

## 消融实验（Text-only / Image-only）

```bash
python train_multimodal.py --mode train --data-dir data --train-file train.txt --output-dir outputs_text_only --checkpoint-path outputs_text_only/best_model.pt --epochs 6 --seed 42 --val-ratio 0.2 --fusion-method concat --ablate-image
python train_multimodal.py --mode train --data-dir data --train-file train.txt --output-dir outputs_image_only --checkpoint-path outputs_image_only/best_model.pt --epochs 6 --seed 42 --val-ratio 0.2 --fusion-method concat --ablate-text
```

## 正则化/预处理消融（批量运行 + 汇总）

批量运行（Windows）：

```bat
run_ablation_3_5_2.bat
```

运行结束后可汇总结果（会扫描 `outputs_352_*` 目录并生成汇总表）：

```bash
python summarize_ablation_3_5_2.py --root . --out-dir outputs_352_summary
```

产物：
- `outputs_352_summary/ablation_3_5_2_summary.csv`
- `outputs_352_summary/ablation_3_5_2_top10.md`

## GMU 门控权重导出与可视化

当 `--fusion-method gmu` 时，验证阶段会在输出目录保存：
- `val_gates.csv`

可视化门控分布：

```bash
python visualize_gates.py --gate-csv outputs_gmu_gate/val_gates.csv --out-dir outputs_gmu_gate/gate_figures
```

产物：
- `outputs_gmu_gate/gate_figures/gate_hist_overall.png`
- `outputs_gmu_gate/gate_figures/gate_hist_by_true_class.png`
- `outputs_gmu_gate/gate_figures/gate_hist_correct_vs_incorrect.png`
- `outputs_gmu_gate/gate_figures/gate_stats.json`

## 复现建议

1. **随机种子**：训练时建议固定 `--seed 42`，并保持同一 `--val-ratio` 以便公平对比。
2. **best 的口径**：脚本默认按验证集 Acc 保存 `best_model.pt`；若你更关注 Macro-F1，可结合 `training_history.json` 的 `val_macro_f1` 曲线选择峰值 epoch。
3. **Windows DataLoader**：Windows 建议保持 `--num-workers 0`，避免多进程带来的卡死/句柄问题。

如需进一步扩展（更大模型、CLIP/BLIP 等），可在 `src/modeling.py` 替换对应编码器，同时复用现有训练脚本。祝实验顺利！

## 一图多信息综合对比图（雷达图 + 参数/耗时 + 收敛曲线）

运行：
- 自动扫描项目根目录下的 `outputs/` 与所有 `outputs_*` 目录（需存在 `val_metrics.json` 与 `training_history.json`）：
  - `python make_composite_figure.py --out outputs/composite_figure.png`

可选：只对比指定实验目录（示例）：
- `python make_composite_figure.py --runs outputs_baseline outputs_blip outputs_gmu outputs_cross_attn outputs_co_attn --out outputs/composite_figure.png`

输出：
- `outputs/composite_figure.png`

## 参考与致谢

本项目参考/使用了以下论文与开源库（报告中也给出了对应引用）：

- Arevalo et al. **GMU: Gated Multimodal Units** (2017)
- Liu et al. **RoBERTa** (2019)
- He et al. **ResNet** (2016)
- Li et al. **BLIP** (2022)
- Liang et al. **R-Drop** (2021)
- Loshchilov & Hutter **AdamW** (2019)
- HuggingFace **Transformers**
- PyTorch / TorchVision
- scikit-learn、matplotlib
