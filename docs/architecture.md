# 明鉴架构说明 v0.2

## 1. 第一版范围

输入新闻文本 + 单张可选图像，输出：

- `P(fake)`、`P(real)` 与检查点阈值判定；
- 风险等级、置信度和不确定性提示；
- 字符级文本证据；
- 图像 Grad-CAM 粗粒度区域热图；
- 文本/图像门控贡献；
- 缺图降级与限制说明；
- 结构化 JSON，供工作台展示。

第一版不包含视频、音频、OCR 全链路、深度伪造定位和大规模爬虫。

## 2. 当前已实现的代码分层

```text
data/schema.py            数据契约
data/dataset.py           JSONL 读写与数据集
data/tokenization.py      字符分词器
data/torch_dataset.py     PyTorch 数据映射与图像预处理
data/weibo17.py           Weibo17 本地适配器与审计
models/lite.py            TextCNN + ImageCNN + 门控融合
training/lite_runner.py   训练、阈值调优、早停、预测收集
evaluation/metrics.py     分类指标与 ECE
explain/lite_explainer.py 文本归因、Grad-CAM、模态贡献
inference/lite_predictor.py 检查点加载与单条推理
web/app.py                无 API Key 本地工作台和 JSON API
```

## 3. lite baseline 模型结构

1. **文本编码器**：字符 Embedding → 多核一维卷积 → padding-aware max pooling。
2. **图像编码器**：4 层 strided CNN → 自适应平均池化；保留最后特征图用于 Grad-CAM。
3. **模态投影**：文本/图像特征分别映射到同一隐藏维度。
4. **门控融合**：根据拼接特征学习逐维 gate，再与 LayerNorm 和分类头连接。
5. **分类头**：两层 MLP + GELU + Dropout + 单 logit。
6. **解释输出**：梯度 x embedding 归因、Grad-CAM、门控贡献幅度和缺图标志。

当前模型不需要下载预训练权重，能在 8 GB 显存环境中离线完成训练与推理。

## 4. 训练流程

- 读取本地 Weibo17 JSONL；
- 只在训练划分上构建字符词表；
- `pos_weight` 缓解类别不平衡；
- AMP 混合精度、梯度裁剪、验证集早停；
- 按验证 AUC 保存最佳权重，并在验证集网格搜索 F1 阈值；
- 输出 `model.pt`、`tokenizer.json`、`metrics.json` 和逐样本预测 CSV；
- 原始数据、权重和 artifacts 默认不进入 Git。

## 5. 解释与降级流程

1. 输入文本编码为字符 ID 和 attention mask；
2. 有图时读取并归一化为 128x128 CHW 张量；无图时构造零图并令 `has_image=0`；
3. 前向计算概率、gate、文本/图像特征和图像特征图；
4. 对 fake logit 反传，提取文本 embedding 梯度和图像特征图梯度；
5. 生成文本证据、Grad-CAM 和模态贡献；
6. 附加“辅助研判、非因果证明、数据域限制、校准不足”等限制说明；
7. 工作台展示结构化 JSON，并记录本次会话历史。

## 6. 后续计划模块

- 显式图文一致性和矛盾性通道；
- 可靠性路由器与缺模态重训练；
- 温度缩放或等价概率校准；
- 批量检测、导出和审核动作闭环；
- 更严格的切分和跨事件鲁棒性评测；
- 可选的预训练文本/图像编码器替换实验。

## 7. 待办

- [x] 接入真实 JSONL 数据管线；
- [x] 实现字符 TextCNN 图像 CNN 融合 baseline；
- [x] 完成 Weibo17 真实训练与测试；
- [x] 实现文本归因、Grad-CAM 和缺图降级；
- [x] 实现无 API Key 的本地工作台；
- [ ] 实现显式一致性/矛盾性双通道；
- [ ] 完成概率校准与校准后指标；
- [ ] 完成消融、跨域和缺模态系统实验；
- [ ] 锁定依赖文件并完成干净环境 10 分钟复现；
- [ ] 输出最终 PDF 技术文档和 5–8 分钟演示视频。
