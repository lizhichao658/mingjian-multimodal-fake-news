# 明鉴模型卡 v0.2（lite baseline 已训练）

## 模型名称

明鉴图文虚假新闻辅助研判 lite baseline。

## 当前版本

- 版本：0.2.0
- 状态：已在 Weibo17 上完成首轮端到端训练、测试和解释器接入。
- 训练提交：`5c11c3e`。
- 输入：新闻文本 + 单张新闻图像；图片可选，缺图时执行降级。
- 输出：`P(fake)`、风险等级、文本证据、图像 Grad-CAM、模态贡献、限制说明和结构化 JSON。

## 当前结构

1. 文本编码器：字符级多核 TextCNN，词表由训练集字符统计构建。
2. 图像编码器：轻量 strided CNN，保留最后特征图用于 Grad-CAM。
3. 融合方式：文本/图像特征分别投影后，通过门控和 LayerNorm 融合。
4. 分类头：两层 MLP + GELU，输出 fake logit，经 sigmoid 得到概率。
5. 解释器：文本使用梯度 x embedding 归因；图像使用 Grad-CAM；跨模态使用门控后的贡献幅度估计。

第一版不下载 BERT/ViT/CLIP 权重，不依赖联网 API，便于在 8 GB 显存环境中离线复现。

## 训练数据

- 数据集：Weibo17 / EANN Weibo；第一版只使用该数据集。
- 本地划分：train / val / test = `5,415 / 843 / 1,465`。
- 标签契约：`0 = real/non-rumor`，`1 = fake/rumor`。
- 原始文本、完整 ID 列表和图片不随仓库提交，不重新分发。
- 数据许可与引用风险已记录在数据卡和合规说明中。

## 主要超参数

| 项目 | 值 |
|---|---:|
| 参数量 | 1,488,161 |
| 词表大小 | 3,706 |
| 最大文本长度 | 196 |
| 图像尺寸 | 128 x 128 |
| Batch size | 64 |
| 初始学习率 | 2e-3 |
| Epochs | 8 |
| Dropout | 0.3 |
| Seed | 42 |
| GPU | RTX 5060 Laptop，8 GB |
| 训练耗时 | 71.66 秒 |

## 真实评测指标

### 测试集（决策阈值 0.5）

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

验证集 F1 最优网格阈值约为 `0.05`。该阈值会把更多样本判为 fake，在验证集得到更高 F1，但测试集 F1 为 `0.8770`，低于阈值 0.5 的 `0.8784`。这说明阈值对划分敏感，当前不做“概率已校准”的声明。工作台展示模型原始概率，并同时报告检查点阈值。

## 已知限制

- 验证损失从 epoch 1 的 `0.3189` 上升到 epoch 8 的 `0.4887`，存在明显过拟合迹象。
- 测试 ECE `0.0918`，概率校准仍不足，阈值附近结果必须人工复核。
- 只在 Weibo17 上训练，跨域、跨平台、跨事件泛化尚未验证。
- Grad-CAM 是粗粒度区域证据，不等同于图像篡改定位。
- 门控贡献是模型内部幅度估计，不代表因果关系。
- 缺图降级通过零化图像特征实现，尚未完成系统的缺模态重训练。
- 不能判断绝对真假，不能作为法律、行政或人身处置的唯一依据。

## 明确不适用的用途

- 对个人、组织进行无人工复核的自动定性；
- 高风险领域（医疗、选举、司法）的自动决策；
- 用热图或归因结果证明某一区域“确实被篡改”；
- 作为事实认定、法律判断或行政处置的唯一证据。

## 复现与验证

训练产物保存在本地 `artifacts/lite_v1/`，包含 `model.pt`、`tokenizer.json`、`metrics.json` 和预测 CSV。权重与 artifacts 默认不进入 Git。

```powershell
python scripts\train_lite_baseline.py --data-dir <processed> --out artifacts/lite_v1 --device cuda
python scripts\explain_lite.py --checkpoint artifacts/lite_v1/model.pt --text "示例文本"
python scripts\run_web_demo.py --checkpoint artifacts/lite_v1/model.pt --device auto
```

## 伦理与合规

- 所有高风险结论必须人工复核；
- 明确展示不确定性与已知限制；
- 不收集或公开个人信息；
- 第三方模型、数据、论文和库必须声明来源；
- 本模型卡中的指标必须与对应运行日志和提交关联。
