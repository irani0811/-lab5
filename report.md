# 实验五：多模态情感分类实验报告
仓库地址：https://github.com/irani0811/-lab5
## 1. 摘要

本实验研究图文配对的三分类情感分析任务，输入为同一 `guid` 对应的一段文本与一张图片，输出情感标签（`negative / neutral / positive`）。我们以 `roberta-base` 与 ImageNet 预训练 `ResNet18` 为骨干，构建了统一训练框架，并在固定的数据划分（`seed=42`、`val_ratio=0.2`）与超参数设置下，对多种后期融合策略进行对比，包括简单拼接（Concat）、门控融合（GMU）、跨注意力（Cross-Attn）与双向协同注意力（Co-Attn）。此外，我们验证了使用 BLIP 生成图像 caption 并与原始文本拼接的增强策略，并通过 text-only / image-only 消融实验量化两种模态的贡献。

主要结论如下：

- **融合结构对比**：Co-Attn 在融合结构对比中取得最优验证集性能（Acc=0.7550，Macro-F1=0.6337），相较 Concat baseline（Acc=0.7425，Macro-F1=0.5972）有稳定提升。
- **Caption 增强**：在不改变融合方式（Concat）的前提下，引入 BLIP caption 可提升整体 Macro-F1（0.5972 → 0.6251），并改善 neutral 类召回率与 F1。
- **类别瓶颈**：neutral 类仍是主要瓶颈，baseline 的 neutral Recall 为 0.2078，说明模型倾向将中性样本误判为正/负向；更强的跨模态交互（Co-Attn）与 caption 增强均能缓解该问题。
- **正则化与预处理消融**：在 24 组对照中，弱图像增强 + R-Drop 达到最高 Macro-F1=0.6522，Neutral F1 最高 0.4463。
- **GMU 门控分析**：`gate_text` 分布集中在 0.5 附近（均值 0.5017，std 0.0049），类别间差异很小，整体更接近等权融合。

## 2. 方法

### 2.1. 模型架构与动机

整体框架采用“文本编码 + 图像编码 + 融合分类”的后期融合范式：

- **文本侧**：采用 HuggingFace `roberta-base`（`AutoModel`）输出的 `last_hidden_state`，取首位 token（RoBERTa 等价于 CLS 位）的表示作为文本全局特征。
- **图像侧**：采用 ImageNet 预训练 `ResNet18/ResNet50`，提取最后卷积特征图，经 `AdaptiveAvgPool2d` 池化后做线性投影得到视觉全局特征。

在融合模块上，我们实现并对比了四类后期融合方式：

- **Concat**：直接拼接文本特征与图像特征，经 LayerNorm 后送入两层 MLP 分类头。
- **GMU**：分别将两模态映射到同一隐藏空间，并用门控网络产生融合权重，实现样本级的动态加权融合。
- **Cross-Attn**：将 ResNet 特征图展平为“视觉 token”，以文本 CLS 为 query 对视觉 token 做多头注意力汇聚，再与图像全局特征拼接进行分类。
- **Co-Attn**：同时进行“文本从视觉聚合上下文”和“视觉从文本聚合线索”的双向注意力交互，再将两路上下文与图像全局特征拼接归一化后分类。该结构更强调跨模态的双向对齐与互补，适合处理图文不一致或任一模态信息不足的样本。

此外，为提高鲁棒性，模型在训练阶段支持 Modality Dropout（特征级随机屏蔽单一模态），并支持 text-only / image-only 的消融开关，用于分析两模态的相对贡献。

### 2.2. 文本数据清洗

Twitter 文本中普遍存在 URL、@ 用户、# 话题等噪声信息。为降低模型对无关模式的拟合，本项目提供 `--clean-text` 开关，对原始文本做轻量清洗。其实现规则为：

- **URL 清理**：移除形如 `http(s)://...` 的 URL 片段。
- **标签清理**：移除以 `@` 或 `#` 开头的标签串（如 `@user`、`#topic`）。
- **空白规范化**：合并连续空格，并做首尾去空。

同时，若 `data/{guid}.txt` 为空文本，读取逻辑会将其回退为 `"[PAD]"`，以避免空输入导致的异常，并保证训练过程稳定。

### 2.3. 图像数据增强

图像侧默认做统一的尺度与归一化处理：

- **Resize**：统一缩放到 `224×224`。
- **Normalize**：采用 ImageNet 均值/方差进行归一化。

训练阶段可通过参数启用增强策略：

- **基础增强（`--use-image-aug`）**：`RandomResizedCrop(scale=0.7~1.0)` + `RandomHorizontalFlip`，并额外加入 `ColorJitter` 与 `GaussianBlur` 的随机扰动。
- **强化增强（`--use-strong-image-aug`）**：在基础增强上叠加 `RandAugment(num_ops=2, magnitude=9)`，用于提升在小数据集上的泛化能力。

### 2.4. 损失函数与优化策略

本项目训练默认采用 AdamW 优化器与 Warmup+Cosine 学习率调度，核心配置为：学习率 `2e-5`，权重衰减 `0.01`，训练轮数 `6`。同时，训练脚本提供多种可选的损失与正则化策略，便于后续扩展：

- **损失函数**：交叉熵（`ce`）/ 类别重加权交叉熵（`weighted_ce`）/ Focal Loss（`focal`）。
- **Label Smoothing**：通过 `--label-smoothing` 抑制过拟合与过度置信。
- **R-Drop**：通过 `--rdrop-alpha` 在双前向输出之间加入对称 KL 约束，提升分布一致性。
- **梯度累积与裁剪**：支持 `--grad-accum-steps` 与 `--max-grad-norm`，用于显存受限场景。



### 2.5. Modality Dropout

为缓解多模态学习中“过度依赖单一强模态”的问题，我们在训练阶段引入样本级的模态丢弃（`--modality-dropout-prob`）。其具体做法是：对同一 batch 内的样本，分别以给定概率随机置零文本特征或图像特征，并显式避免同时丢弃两种模态。该策略能够迫使模型学习到更稳健的跨模态判别机制，从而在推理阶段更好地处理模态冲突与噪声输入。

## 3. 实验与分析

### 3.1. 实验设置

本实验使用提供的图文配对数据集，训练集共 4000 条样本，类别分布为：positive 2388、negative 1193、neutral 419。按照 `val_ratio=0.2` 从训练集中划分验证集（固定 `seed=42`），得到验证集 800 条样本，类别分布为：positive 478、negative 245、neutral 77。测试集 `test_without_label.txt` 共 511 条样本，仅提供 `guid` 用于最终推理与提交。

本实验的统一超参数设置如表1 所示。除对比变量外，各实验尽量保持一致，以确保结论具备可比性。

表1. 关键实验设置与超参数

| 模块 | 设置 |
|---|---|
| 数据划分 | `val_ratio=0.2`，`seed=42` |
| 编码器 | Text: `roberta-base`；Image: `ResNet18`（ImageNet 预训练） |
| 输入 | `max_len=128`；image=224×224 |
| 优化与训练 | AdamW（lr=2×10^-5, wd=0.01），epochs=6，batch=16 |
| 学习率调度 | Warmup+Cosine（`warmup_ratio=0.1`） |
| 正则化与融合 | dropout=0.2；modality_dropout=0.1；fusion_hidden=512；image_proj=256；attn_heads=4 |

评价指标以验证集 Accuracy 与 Macro-F1 为主，同时给出类别级 Precision/Recall/F1。训练脚本默认以验证集 Accuracy 保存 `best_model.pt`，因此 `val_metrics.json` 对应的是“按 Acc 最优”的分类报告；而逐 epoch 的收敛情况记录在 `training_history.json` 中。

### 3.2. 不同模型架构性能对比

#### 3.2.1. 融合结构对比

在保持骨干网络与训练设置一致的前提下，我们仅切换 `fusion_method`，对比 Concat / GMU / Cross-Attn / Co-Attn 四种后期融合结构在验证集上的表现。统计结果如表2 所示，其中参数量为模型可训练参数总量（M），平均耗时为每个 epoch 的平均训练耗时（秒）。

表2. 不同融合结构验证集性能与代价对比

| 架构名称 | Acc最优Epoch | Acc / Macro-F1 / W-F1（按Acc最优） | Neutral F1 | 参数量(M) | 平均每轮耗时(s) |
|---|---:|---|---:|---:|---:|
| Concat（baseline） | 3 | 0.7425 / 0.5972 / 0.7283 | 0.2857 | 136.76 | 63.8 |
| GMU | 3 | 0.7325 / 0.6049 / 0.7206 | 0.3448 | 137.54 | 65.5 |
| Cross-Attn | 2 | 0.6562 / 0.4926 / 0.6228 | 0.2524 | 139.51 | 103.8 |
| Co-Attn | 4 | **0.7550 / 0.6337 / 0.7493** | **0.3714** | 142.27 | 143.7 |

从表2 可以看出：

- **Co-Attn 效果最优但代价最高**：其 Acc 与 Macro-F1 均为最高，同时 neutral F1 达到 0.3714，但由于引入双向注意力交互，训练耗时显著增加。
- **GMU 在 neutral 上更均衡**：相较 Concat，GMU 的 neutral F1 有提升（0.2857 → 0.3448），但整体 Acc 略有下降，体现了“均衡性提升”与“总体正确率”之间的权衡。
- **Cross-Attn 收敛与稳定性较弱**：该结构在本设置下出现明显性能退化，且训练成本较高，说明仅靠单向跨注意力并不一定能带来收益。

为便于从“效果—效率”角度综合观察不同方法的取舍，我们进一步给出多维对比可视化（雷达图、参数量与训练时间、Macro-F1 收敛曲线）如下。

![composite_figure](project5/outputs/composite_figure.png)

图1. 多方法综合对比可视化（归一化雷达图、参数量与训练时间、Macro-F1 收敛曲线）

#### 3.2.2. BLIP Caption 增强对比

本节保持融合方式为 Concat 不变，仅引入 BLIP 生成的图像 caption，并将输入文本拼接为 `[原始文本] [SEP] [caption]`，以检验“生成式语义注入”对验证集性能的影响。对比结果如表3。

表3. Caption 增强对比（Concat baseline vs BLIP+Concat）

| 配置 | Acc最优Epoch | Acc / Macro-F1 / W-F1（按Acc最优） | Neutral P / R / F1 | 平均每轮耗时(s) |
|---|---:|---|---|---:|
| Baseline（不加 caption） | 3 | 0.7425 / 0.5972 / 0.7283 | 0.4571 / 0.2078 / 0.2857 | 63.8 |
| BLIP Caption（拼接 caption） | 3 | 0.7462 / 0.6251 / 0.7389 | 0.4528 / 0.3117 / 0.3692 | 83.7 |

可以观察到，引入 caption 后 Macro-F1 提升更明显（+0.028），且 neutral Recall 从 0.2078 提升到 0.3117。与此同时，由于文本序列变长与输入信息增加，训练耗时也相应上升。整体上，caption 将图像语义显式转写为文本线索，有助于提升模型对困难样本的判别能力。

#### 3.2.3. 训练过程分析

为进一步观察模型的收敛速度与训练成本，我们对 Concat baseline 与 Co-Attn 的训练过程可视化结果进行对比，如图2 所示。

| Concat（baseline） | Co-Attn |
|---|---|
| ![baseline_training](project5/outputs_baseline/figures/training_curves.png) | ![coattn_training](project5/outputs_co_attn/figures/training_curves.png) |

图2. Baseline 与 Co-Attn 的训练曲线与每轮耗时对比

从图2 可以看出：

- **收敛趋势**：两种方法在前 2~3 个 epoch 内 Macro-F1 上升较快，随后趋于平稳。
- **训练成本**：Co-Attn 的每轮耗时显著高于 baseline（表2 中约 143.7s vs 63.8s），主要源于引入双向注意力交互计算。

### 3.3. 混淆矩阵分析

为了定位性能差异的来源，我们进一步对比不同方法在验证集上的混淆矩阵与类别级 Precision/Recall/F1（图3、图4）。

| Baseline（Concat） | Co-Attn | BLIP+Concat |
|---|---|---|
| ![baseline_cm](project5/outputs_baseline/figures/confusion_matrix.png) | ![coattn_cm](project5/outputs_co_attn/figures/confusion_matrix.png) | ![blip_cm](project5/outputs_blip/figures/confusion_matrix.png) |

图3. 验证集混淆矩阵对比（从左到右：baseline、Co-Attn、BLIP+Concat）

| Baseline（Concat） | Co-Attn | BLIP+Concat |
|---|---|---|
| ![baseline_cls](project5/outputs_baseline/figures/classwise_metrics.png) | ![coattn_cls](project5/outputs_co_attn/figures/classwise_metrics.png) | ![blip_cls](project5/outputs_blip/figures/classwise_metrics.png) |

图4. 类别级 Precision / Recall / F1 对比（从左到右：baseline、Co-Attn、BLIP+Concat）

结合图3、图4 与 `val_metrics.json`，可以得到以下结论：

- **neutral 仍是主要误差来源**：baseline 中 neutral 仅 16/77 被正确识别（Recall=0.2078），其中 40/77 被误判为 positive，表明模型倾向于将“中性”样本推向情绪两端。
- **Co-Attn 改善 neutral 识别**：Co-Attn 将 neutral 正确数提升至 26/77（Recall=0.3377，F1=0.3714），同时减少 neutral→positive 的误判（40→31）。
- **BLIP 对 neutral 也有增益**：引入 caption 后 neutral 正确数提升至 24/77（Recall=0.3117，F1=0.3692），说明将图像语义转写为文本有助于缓解“图像信息难以直接利用”的问题。


### 3.4. 消融实验

#### 3.4.1. 模态消融

为验证两种模态对最终性能的贡献，我们进行了 text-only 与 image-only 消融实验，并与 Co-Attn 的多模态结果对比，如表5。

表4. 单模态消融与多模态对比

| 模型 | Acc | Macro-F1 | Neutral R / F1 | 平均每轮耗时(s) |
|---|---:|---:|---|---:|
| Text-only | 0.7275 | 0.5548 | 0.1429 / 0.2245 | 57.3 |
| Image-only | 0.6700 | 0.4721 | 0.1169 / 0.1935 | 33.6 |
| Co-Attn（Text+Image） | **0.7550** | **0.6337** | **0.3377 / 0.3714** | 143.7 |

进一步地，我们统计了 text-only 与 image-only 在验证集上的预测一致性，并考察多模态模型在“分歧样本”上的纠错能力：

- **一致性**：两种单模态预测一致的样本占 72.6%（581/800），其中两者同时正确 462 条、同时错误 119 条。
- **分歧纠错**：在 27.4%（219/800）的分歧样本上，多模态 Co-Attn 能将其中 64.8% 的样本纠正为正确预测（142/219）。
- **模态贡献差异**：当 text-only 正确而 image-only 错误时（120 条），Co-Attn 有 89.2%（107/120）给出正确预测；当 image-only 正确而 text-only 错误时（74 条），Co-Attn 的纠错率为 43.2%（32/74）。该现象说明当前任务中文本仍是更强信号，但图像在部分样本上提供了有效补充。

#### 3.4.2. 正则化与数据预处理消融

本项目在训练侧实现了多种可选项（如 `--clean-text`、图像增强强度、label smoothing、R-Drop、Modality Dropout 概率等），这些策略通常影响“泛化稳定性”而非单一指标的极值。

我们系统性地完成了该组消融（共 24 组配置），并按维度汇总最优设置与总体趋势如下：

- **文本清洗（clean-text）**：对比开启/关闭 URL 与 @/# 标签清理。
- **图像增强强度**：对比无增强 / 基础增强 / 强增强（RandAugment）。
- **正则化项**：对比（label smoothing、R-Drop、modality dropout）不同组合的增益与训练开销。

表5. 正则化与预处理消融（各维度最优设置）

| 维度 | 设置 | 对应实验 | Acc | Macro-F1 | Neutral F1 |
|---|---|---|---:|---:|---:|
| clean-text | off | outputs_352_c0_weak_rdrop | **0.7538** | **0.6522** | 0.4414 |
| clean-text | on | outputs_352_c1_none_md | 0.7525 | 0.6518 | 0.4463 |
| image aug | none | outputs_352_c1_none_md | 0.7525 | 0.6518 | 0.4463 |
| image aug | weak | outputs_352_c0_weak_rdrop | 0.7538 | 0.6522 | 0.4414 |
| image aug | strong | outputs_352_c0_strong_ls | 0.7438 | 0.6428 | 0.4286 |
| regularization | base | outputs_352_c0_none_base | 0.7500 | 0.6475 | 0.4317 |
| regularization | +LS | outputs_352_c1_none_ls | 0.7513 | 0.6495 | 0.4394 |
| regularization | +RDrop | outputs_352_c0_weak_rdrop | 0.7538 | 0.6522 | 0.4414 |
| regularization | +MD | outputs_352_c1_none_md | 0.7525 | 0.6518 | 0.4463 |

从结果可以观察到：

- **clean-text**：开启/关闭的差异较小（最优 Macro-F1 0.6522 vs 0.6518），说明该数据集文本噪声并非主要瓶颈。
- **图像增强**：弱增强整体更稳健；强增强在本任务上未带来收益，Macro-F1 反而下降，可能是 RandAugment 引入了与情感相关的细节噪声。
- **正则化**：R-Drop 与 Modality Dropout 均带来小幅提升（Macro-F1≈0.652），且对 Neutral F1 更友好（最高 0.4463）。

### 3.6. 对后期融合模型的进一步分析

#### 3.6.1. 门控权重分布与模态偏好（GMU）

GMU 通过门控网络为不同样本分配“更依赖文本/更依赖图像”的融合权重。我们在验证阶段导出每个样本的门控均值 `gate_text`（值越接近 1 表示越依赖文本，越接近 0 表示越依赖图像），并对 800 条验证样本进行统计分析。

整体上，`gate_text` 的均值为 0.5017、标准差 0.0049（min=0.4923, max=0.5134），门控几乎集中在 0.5 附近，说明在当前训练设置下 GMU 更接近“近似等权融合”，并未呈现强烈的样本级模态切换。

- **类别偏好**：negative/neutral/positive 的 `gate_text` 均值分别为 0.5047 / 0.5015 / 0.5002，差异很小，类别间未呈现明显的模态偏好分化。
- **正确/错误对比**：正确与错误样本的 `gate_text` 均值分别为 0.5015 与 0.5022，同样差异不显著。

<table align="center">
  <tr>
    <td align="center">
      <img src="project5/outputs_gmu_gate/gate_figures/gate_hist_overall.png" style="width:320px; max-width:100%;" />
    </td>
    <td align="center">
      <img src="project5/outputs_gmu_gate/gate_figures/gate_hist_correct_vs_incorrect.png" style="width:320px; max-width:100%;" />
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="project5/outputs_gmu_gate/gate_figures/gate_hist_by_true_class.png" style="width:520px; max-width:100%;" />
    </td>
  </tr>
</table>

## 4. 总结

本实验围绕图文多模态情感分类任务，构建了统一的训练与评测框架，并在固定数据划分与超参设置下对多种融合策略进行了对比。实验表明：
- **融合策略方面**：Co-Attn 在融合结构对比中表现最佳（Acc=0.7550，Macro-F1=0.6337），但训练成本明显更高；在进一步的 3.5.2 消融中，加入弱图像增强 + R-Drop 可将 Macro-F1 提升到 0.6522。
- **类别难点方面**：neutral 是主要短板，baseline 的 neutral Recall 为 0.2078；引入更强跨模态交互或 caption 增强后，neutral 识别得到改善。
- **模态互补方面**：text-only 整体优于 image-only，但在分歧样本上，多模态模型能较高比例纠正单模态错误，说明融合确实利用到了互补信息。
- **可解释性方面**：GMU 的门控均值 `gate_text` 基本集中在 0.5 附近，类别间差异很小，表明该设置下 GMU 更接近等权融合。

## 5. 参考文献

[1] Julian Arevalo, Thamar Solorio, Manuel Montes-y-Gomez, Fabio A. González. *Gated Multimodal Units for Information Fusion*. ICLR 2017 Workshop.

[2] Yinhan Liu, Myle Ott, Naman Goyal, et al. *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. arXiv:1907.11692, 2019.

[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016.

[4] Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi. *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*. ICML 2022.

[5] Xiaobo Liang, Hao He, Zhiqiang Shen, et al. *R-Drop: Regularized Dropout for Neural Networks*. NeurIPS 2021.

[6] Ilya Loshchilov, Frank Hutter. *Decoupled Weight Decay Regularization (AdamW)*. ICLR 2019.
