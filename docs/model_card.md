# 明鉴模型卡 v0.3（v1.1 证据方向卡版）

## 模型名称

明鉴图文虚假新闻辅助研判 lite baseline。

## 当前版本

- 产品版本：**v1.1 证据方向卡版**
- 模型卡版本：0.3.0
- 展示模型状态：沿用 v1.0 的同一个检查点，未替换权重。本地复现入口另行训练新模型，不覆盖该检查点或下列历史指标。
- 训练提交：`5c11c3e`。
- 输入：新闻文本 + 单张可选新闻图像；缺图时执行降级。
- 输出：`P(fake)`、`P(real)`、风险等级、文本梯度证据、图像 Grad-CAM、模态贡献、图文证据方向卡、人工复核提示和结构化 JSON。

> 证据卡比较的是模型内部梯度证据方向，不是语义一致性、事实核验、篡改定位或因果证明。

## 当前结构

1. 文本编码器：字符级多核 TextCNN，词表由训练集字符统计构建。
2. 图像编码器：轻量 strided CNN，保留最后特征图用于 Grad-CAM。
3. 融合方式：文本/图像特征分别投影后，通过门控和 LayerNorm 融合。
4. 分类头：两层 MLP + GELU，输出 fake logit，经 sigmoid 得到概率。
5. 单模态解释：文本使用梯度 × embedding 归因；图像使用 Grad-CAM。
6. 图文证据方向卡：对 fake logit 反传，比较文本/图像投影特征的梯度方向、证据占比和方向稳定性。

第一版不下载 BERT/ViT/CLIP 预训练权重，不依赖联网 API，便于在 8 GB 显存环境中离线复现。

## 训练数据

- 数据集：Weibo17 / EANN Weibo；第一版只使用该数据集。
- 官方评测划分：train / validate / test = `5,415 / 843 / 1,465`，合计 `7,723`。
- 标签契约：`0 = real/non-rumor`，`1 = fake/rumor`。
- 原始文本、完整 ID 列表、图片和归档不随仓库提交，不重新分发。
- 数据许可与引用风险记录在数据卡和合规说明中；只表述为“公开可获取研究数据”。

## 主要超参数

| 项目 | 值 |
|---|---:|
| 参数量 | 1,488,161 |
| 词表大小 | 3,706 |
| 最大文本长度 | 196 |
| 图像尺寸 | 128 × 128 |
| Batch size | 64 |
| 初始学习率 | 2e-3 |
| Epochs | 8 |
| Dropout | 0.3 |
| Seed | 42 |
| GPU | RTX 5060 Laptop，8 GB |
| 训练耗时 | 71.66 秒 |

## 真实分类指标

### 测试集（默认决策阈值 0.5）

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.8785 |
| Precision | 0.9082 |
| Recall | 0.8505 |
| F1 | 0.8784 |
| AUC | 0.9437 |
| ECE | 0.0918 |

### 验证集（决策阈值 0.5）

Accuracy `0.8529`，Precision `0.9125`，Recall `0.8040`，F1 `0.8548`，AUC `0.9436`，ECE `0.1194`。

### 阈值调优说明

验证集 F1 最优网格阈值约为 `0.05`。该阈值在测试集上的 F1 为 `0.8770`，低于阈值 `0.5` 的 `0.8784`。这说明阈值对划分敏感，当前不声明概率已校准，默认仍展示模型概率和 `0.5` 决策阈值。

## 图文证据方向卡

对 fake logit 反传后计算：

```text
text_signed    = sum(text_feature_grad  * text_feature)
image_signed   = sum(image_feature_grad * image_feature)
text_strength  = |text_signed|  / (|text_signed| + |image_signed|)
image_strength = |image_signed| / (|text_signed| + |image_signed|)
coherence      = |sum(grad * feature)| / sum(|grad * feature|)
```

`text_strength` 是“证据占比”而非绝对置信度：两路占比合计为 1，只表示本次净有向梯度贡献在两种模态间的相对分配。

默认判定顺序：

1. 缺图：`insufficient / missing_image`；
2. 任一方向为 neutral：`insufficient / neutral_direction`；
3. 任一模态证据占比低于 10%：`insufficient / weak_modality`；
4. 任一方向稳定性低于 10%：`insufficient / unstable_direction`；
5. 两路方向相同：`support / same_direction`；
6. 两路方向相反：`conflict / opposite_direction`。

10% 是工程启发式阈值，不是经过真实分布校准的最优值。模型卡不预设三态比例；真实比例必须由 `scripts/audit_evidence_cards.py` 在给定检查点和数据划分上审计。

## 解释能力边界

- “证据同向”只表示两路模型梯度方向相同，不代表图文语义一致；
- “方向冲突”只表示两路梯度方向相反，不代表事实矛盾；
- Grad-CAM 是粗粒度相对关注区域，不是图像篡改定位；
- 门控贡献是模型内部幅度估计，不代表因果关系；
- 证据卡没有改变分类指标，不能把 F1/AUC 的提升归因给证据卡。

## 已知限制

- 验证损失从 epoch 1 的 `0.3189` 上升到 epoch 8 的 `0.4887`，存在明显过拟合迹象。
- 测试 ECE `0.0918`，概率校准仍不足，阈值附近结果必须人工复核。
- 只在 Weibo17 上训练，跨域、跨平台、跨事件泛化尚未验证。
- 缺图降级通过零化图像特征实现，尚未完成系统的缺模态重训练。
- 三态阈值尚未形成经真实数据分布验证的固定策略。
- 不能判断绝对真假，不能作为法律、行政或人身处置的唯一依据。

## 明确不适用的用途

- 对个人、组织进行无人工复核的自动定性；
- 高风险领域（医疗、选举、司法）的自动决策；
- 用热图或归因结果证明某一区域“确实被篡改”；
- 把“证据同向 / 方向冲突”解释为事实一致性或事实矛盾；
- 作为事实认定、法律判断或行政处置的唯一证据。

## 复现与验证

评委没有模型权重时，按[真实模型运行指南](reviewer_quickstart.md)从发布方准备完整归档，执行`prepare_local_model.py`，在仓库外生成新模型。2026-10-30起训练入口在模型初始化前设置种子；不能承诺新模型逐位等同历史检查点，新指标以该次输出为准。

训练产物保存在本地 `artifacts/lite_v1/`，包含 `model.pt`、`tokenizer.json`、`metrics.json` 和预测 CSV。权重与 artifacts 默认不进入 Git。

```powershell
.\.venv\Scripts\python.exe scripts\train_lite_baseline.py --data-dir <processed> --out artifacts/new_local_run --epochs 8 --batch-size 64 --lr 2e-3 --device cuda
.\.venv\Scripts\python.exe scripts\explain_lite.py --checkpoint artifacts/lite_v1/model.pt --text "示例文本"
.\.venv\Scripts\python.exe scripts\run_web_demo.py --checkpoint artifacts/lite_v1/model.pt --device auto
.\.venv\Scripts\python.exe scripts\audit_evidence_cards.py --checkpoint artifacts/lite_v1/model.pt --data <processed>/test.jsonl --limit 200 --out artifacts/evidence_card_audit.json
```

## 伦理与合规

- 所有高风险结论必须人工复核；
- 明确展示不确定性与已知限制；
- 不收集或公开个人信息；
- 第三方模型、数据、论文和库必须声明来源；
- 本模型卡中的指标与训练提交 `5c11c3e` 关联，证据卡改造不改变该分类结果。
