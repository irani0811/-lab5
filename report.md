
# 实验五：多模态情感分类实验报告

## 1. 实验任务概述
- 目标：输入配对的文本与图像，输出情感标签（positive / neutral / negative）。
- 数据来源：`train.txt`、`test_without_label.txt` 与 `data/` 目录下的 guid 文件。
- 评分/提交要求参考《实验五要求》PPT：提交代码 + 报告 + 测试集预测文件，并在报告中展示模型设计、验证结果、消融分析以及创新点。

## 2. 数据集与预处理
- `train.txt`：约 5k 样本，包含 guid 与标签。
- `data/`：每个 guid 对应一条英文短文本与一张图片。
- 注意：`data/` 目录内存在少量冗余文件，训练与评估时以 `train.txt` / `test_without_label.txt` 中提供的 guid（以及训练标签）为准。
- 预处理策略：
  - 文本：使用 HuggingFace `roberta-base` tokenizer，最长 128 token，空文本回退到 `[PAD]`；支持仅微调最后 N 层。
  - 图像：统一 Resize 到 224×224，并按 ImageNet 均值/方差归一化；训练可启用 RandAug 风格增强（RandomResizedCrop、ColorJitter、GaussianBlur）。
  - 训练阶段按照 PPT 要求，从训练集划分 20% 作为验证集（`--val-ratio` 可调），并可选对 neutral 类过采样及类别均衡采样。

## 3. 模型与训练流程
- 文本编码：`roberta-base` 的 CLS 向量；可通过 `--text-train-layers` 仅微调最后 N 层以提升稳定性。
- 图像编码：支持 ImageNet 预训练 `ResNet18/ResNet50`，并通过线性+LayerNorm+GELU 投影至 256 维；可根据显存选择骨干。
- 模态增强与 Dropout：新增 **模态 Dropout**（`--modality-dropout-prob`）随机屏蔽单一模态，提升鲁棒性。
- 融合策略：除 concat / GMU / cross-attn 外新增 **双向协同注意力（co_attn）**，允许文本与图像互相检索关键线索后再融合。
- 优化策略：AdamW（lr=2e-5）、梯度累积 (`--grad-accum-steps`)、Warmup+Cosine 调度、AMP、EMA、标签平滑、R-Drop、一致性 Focal Loss，均可按需打开。
- 训练脚本：`train_multimodal.py`（GPU 优先，默认 6 epoch，可根据 PPT 要求手动提高）。
- 评估指标：验证集准确率 + Macro-F1（`val_metrics.json` 输出），`visualize.py` 自动绘制 Loss/Acc/F1/耗时曲线、混淆矩阵、类别 PRF 柱状图。

## 4. 结果与可视化

### 4.1 融合策略对比（固定 seed=42, val_ratio=0.2）

| 配置（fusion_method） | Best Val Acc | Best Val Macro-F1 | Best Epoch |
|---|---:|---:|---:|
| Concat（baseline） | 0.7425 | 0.5972 | 3 |
| GMU | 0.7325 | 0.6049 | 3 |
| Cross-Attn | 0.6563 | 0.4926 | 2 |
| Co-Attn | **0.7550** | **0.6337** | 4 |

- 结论：在本数据集与当前训练设置下，**Co-Attn** 在准确率与 Macro-F1 上均优于其他融合方式；Cross-Attn 训练不稳定且指标明显下降。
- 对应输出：`outputs/compare_fusion/compare_table.csv`、`outputs/compare_fusion/compare_bar.png`。

### 4.2 BLIP Caption 拼接增强对比（Concat baseline vs BLIP Caption）

| 配置 | Best Val Acc | Best Val Macro-F1 | Best Epoch |
|---|---:|---:|---:|
| Baseline（不加 caption） | 0.7425 | 0.5972 | 3 |
| BLIP Caption（[文本] [SEP] [caption]） | **0.7463** | **0.6251** | 3 |

- 结论：Caption 拼接对 **Macro-F1 提升明显（+0.0280）**，准确率小幅提升（+0.0038）。这说明 caption 的主要收益来自**困难类（neutral）**的改进，而非整体多数类的简单提升。
- 对应输出：`outputs/compare_ab/compare_table.csv`、`outputs/compare_ab/compare_bar.png`。

### 4.3 类别级别分析（Neutral 改进最明显）

| 配置 | Neutral Precision | Neutral Recall | Neutral F1 |
|---|---:|---:|---:|
| Baseline | 0.4571 | 0.2078 | 0.2857 |
| BLIP Caption | 0.4528 | 0.3117 | 0.3692 |
| Co-Attn | 0.4127 | 0.3377 | 0.3714 |

- 现象：Baseline 对 neutral 的 recall 很低（0.2078），说明模型倾向把 neutral 判为 positive/negative。
- 解释：BLIP caption 将图像语义显式注入文本端，使得模型更容易捕捉到“无明显情感倾向”的描述线索，从而提升 neutral 的召回与 F1。

### 4.4 可视化产物（用于报告插图）

- 每个实验目录下均可通过 `visualize.py` 生成：
  - `training_curves.png`：Loss / Val Acc / Macro-F1 / Epoch Time 曲线
  - `confusion_matrix.png`：混淆矩阵
  - `classwise_metrics.png`：类别 P/R/F1 柱状图
  - `visualization_summary.txt`：完整分类报告

建议在报告中至少展示两组：
- `outputs_baseline/figures/`（baseline）
- `outputs_blip/figures/` 或 `outputs_co_attn/figures/`（改进方法）

测试集预测（示例）：
- `python train_multimodal.py --mode predict --checkpoint-path outputs_co_attn/best_model.pt --output-dir outputs_co_attn ...`

### 4.5 全量对比实验结果汇总（覆盖所有 outputs_* 目录）

说明：下表对项目目录中已生成 `val_metrics.json` 的实验结果进行汇总（包含 `outputs/`、`outputs_final/` 等历史实验目录）。每一行的分类指标来自对应目录的 `val_metrics.json`，训练耗时来自 `training_history.json` 的 `epoch_time_sec`。

| 实验目录 | Best Val Acc | Best Val Macro-F1 | Negative (P/R/F1) | Neutral (P/R/F1) | Positive (P/R/F1) | Epochs | 平均每轮耗时(s) | 总耗时(min) |
|---|---:|---:|---|---|---|---:|---:|---:|
| outputs_baseline | 0.7425 | 0.5972 | 0.6844/0.6816/0.6830 | 0.4571/0.2078/0.2857 | 0.7889/0.8598/0.8228 | 6 | 63.8 | 6.4 |
| outputs_concat | 0.7425 | 0.5972 | 0.6844/0.6816/0.6830 | 0.4571/0.2078/0.2857 | 0.7889/0.8598/0.8228 | 6 | 66.6 | 6.7 |
| outputs_gmu | 0.7325 | 0.6049 | 0.6667/0.6449/0.6556 | 0.5128/0.2597/0.3448 | 0.7786/0.8536/0.8144 | 6 | 65.5 | 6.6 |
| outputs_cross_attn | 0.6563 | 0.4926 | 0.5897/0.3755/0.4589 | 0.5000/0.1688/0.2524 | 0.6796/0.8787/0.7664 | 6 | 103.8 | 10.4 |
| outputs_co_attn | **0.7550** | **0.6337** | 0.7118/0.6653/0.6878 | 0.4127/0.3377/0.3714 | 0.8169/0.8682/0.8418 | 6 | 143.7 | 14.4 |
| outputs_blip | 0.7463 | 0.6251 | 0.6805/0.6694/0.6749 | 0.4528/0.3117/0.3692 | 0.8083/0.8556/0.8313 | 6 | 83.7 | 8.4 |
| outputs | 0.7425 | 0.5972 | 0.6844/0.6816/0.6830 | 0.4571/0.2078/0.2857 | 0.7889/0.8598/0.8228 | 6 | 76.6 | 7.7 |
| outputs_final | 0.7513 | 0.5882 | 0.6691/0.7347/0.7004 | 0.7333/0.1429/0.2391 | 0.7946/0.8577/0.8249 | 12 | 42.8 | 8.6 |

备注：
- `outputs_baseline` / `outputs_concat` / `outputs` 三组的 `val_metrics.json` 完全一致，属于相同基线配置在不同输出目录下的重复记录；报告主对比建议以 `outputs_baseline` 作为 baseline。
- `outputs_final` 代表历史更长轮次的训练实验（12 epochs）；若你希望严格公平对比，可只比较同等训练轮次（例如都为 6 epochs）的实验目录。

## 5. 遇到的问题与解决
1. **模型复杂度与性能的权衡**：实验发现，更复杂的模型（如 `ResNet50` + `Co-Attention`）并不能带来精度提升，反而导致训练时间急剧增加（单 epoch >10 分钟），且收敛更不稳定。最终选择了一个轻量且高效的基线模型，通过精细调参达到了更优的结果。
2. **模型过拟合**：训练曲线显示，模型在 4-6 个 epoch 后验证集准确率开始停滞或下降。通过引入模态 Dropout、标签平滑、早停（`--patience`）等策略，有效缓解了过拟合，并自动保存了验证集上表现最佳的模型权重。
3. **Neutral 类别识别困难**：从混淆矩阵和分类报告看，Neutral 类别的 Precision/Recall/F1 均显著低于其他两类。虽尝试了类别均衡采样、过采样、Focal Loss 等方法，但提升有限，说明该类别的图文特征本身可能较为模糊，是模型性能的主要瓶颈。

## 6. 创新点/亮点
1. **多骨干 + 多融合可插拔框架**：`MultiModalSentimentModel` 现支持 ResNet18/50、Concat/GMU/Cross-Attn/Co-Attn，并提供模态 Dropout/LayerNorm 等模块，满足 PPT “创新设计”要求。
2. **一致性正则化套件**：R-Drop + Label Smoothing + Modality Dropout + EMA + Weighted/Focal Loss 可组合使用，在 neutral 类上特别有效。
3. **可视化与诊断升级**：新增 Macro-F1 曲线、类别柱状图、耗时统计，自动生成指标文本，方便报告撰写与课堂答辩展示。
4. **训练效率优化**：Warmup+Cosine 调度、梯度累积、AMP、可选冻结层，保证在单卡也能训练更深模型，满足“精度与效率兼顾”的实验要求。
5. **BLIP Caption 拼接增强（生成式辅助）**：使用 BLIP 为每张图像生成英文描述（caption），并将其与原始文本拼接为 `[文本] [SEP] [caption]` 输入文本编码器，从而将图像语义显式注入文本端，缓解图像信息不充分或 text/image 不一致导致的误判；同时保留原有图像编码分支，形成更强的多模态表示。
1. **系统性的调参与模型选择**：通过多轮、多组合的对比实验，验证了并非所有“重型”模块都能带来增益，最终选择了一个在效率与精度上达到最佳平衡的轻量化模型，这本身就是一种有效的工程实践与创新。
2. **全面的正则化与早停策略**：代码中集成了模态 Dropout、R-Drop、标签平滑、EMA、权重衰减以及早停机制，可灵活组合以应对不同阶段的过拟合问题，保证了模型的泛化能力。
3. **增强的可视化诊断**：`visualize.py` 脚本不仅绘制 Loss/Accuracy 曲线，还增加了 Macro-F1、Epoch 耗时以及各类别 P/R/F1 柱状图，为快速定位模型瓶颈（如 neutral 类表现不佳、过拟合点）提供了有力工具。

## 7. 优化建议
1. **数据驱动的优化**：
   - **数据清洗**：人工检查一部分被错分的 neutral 样本，分析其图文内容是否存在歧义或标注偏差，考虑对这部分数据进行修正或剔除。
   - **数据增强**：除了已有的图像增强，可尝试对文本进行 back-translation 或同义词替换，增加文本多样性。
2. **更强的预训练模型**：当前 `roberta-base` 与 `resnet18` 的表征能力可能已达上限。下一步可考虑直接使用在图文对上预训练过的模型，如 **CLIP** 或 **BLIP**，提取图文特征后再进行分类，这通常比早期融合有更强的效果。
3. **模型结构微调**：可以尝试在当前的 concat 特征后，增加一个小的 Transformer Encoder（1-2层），让融合后的特征进行更复杂的交互，可能比单纯的 MLP 能捕捉到更深层的关联。

## 参考文献
[1] Julian Arevalo, Thamar Solorio, Manuel Montes-y-Gomez, Fabio A. González. *Gated Multimodal Units for Information Fusion*. ICLR 2017 Workshop.
