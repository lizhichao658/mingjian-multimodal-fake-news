# 明鉴（MingJian）

> 面向图文新闻的可解释多模态辅助研判工作台。
> 当前版本：**v1.1 证据方向卡版（文本 + 单图）**。分类模型、Weibo17 数据、检查点和原始分类指标均沿用 v1.0；新增内容集中在解释层与工作台。

- 公开仓库：https://github.com/lizhichao658/mingjian-multimodal-fake-news
- 第一版范围：新闻文本 + 单张可选配图
- 产品边界：辅助研判，不自动定罪，不构成事实认定、法律意见或行政处置依据

## 1. 项目定位

内容审核人员面对一条可疑图文消息时，真正需要的不是一个孤立的真假标签，而是：

1. 模型给出的风险方向和概率；
2. 文本中哪些片段影响了判断；
3. 图像中哪些区域受到关注；
4. 图文两路模型证据是否同向、冲突或不足；
5. 哪些情况必须转人工复核。

明鉴第一版输出：

- `P(fake)`、`P(real)`、决策标签与检查点阈值；
- `low / medium / high / uncertain` 风险等级；
- 字符级与句级文本梯度证据；
- 粗粒度 Grad-CAM 相对关注热图与 Top-K 网格；
- 文本/图像门控贡献的幅度估计；
- **图文证据方向卡**：`证据同向 / 方向冲突 / 证据不足`；
- 缺图降级、限制说明和本地会话历史；
- 无 API Key 的标准库 HTTP 工作台与结构化 JSON。

> “证据同向 / 方向冲突”比较的是模型梯度证据方向，**不是语义一致性、事实核验、篡改定位或因果证明**。

## 2. 证据方向卡

证据卡复用同一次前向/反向传播，不引入第二个解释模型，也不新增可训练参数。对 fake logit 反传后，分别计算：

```text
text_signed    = sum(text_feature_grad  * text_feature)
image_signed   = sum(image_feature_grad * image_feature)
text_strength  = |text_signed|  / (|text_signed| + |image_signed|)
image_strength = |image_signed| / (|text_signed| + |image_signed|)
coherence      = |sum(grad * feature)| / sum(|grad * feature|)
```

`text_strength` 是“证据占比”而非绝对置信度：两路占比合计为 1，只表示本次净有向梯度贡献在两种模态间的相对分配。

判定顺序：

1. 缺图：`证据不足`；
2. 任一方向为中性：`证据不足`；
3. 任一模态证据占比小于 10%：`证据不足`；
4. 任一方向稳定性低于 10%：`证据不足`；
5. 两路方向相同：`证据同向`；
6. 两路方向相反：`方向冲突`。

10% 是工程启发式阈值，必须用真实检查点审计分布后再解释。审计命令：

```powershell
.\.venv\Scripts\python.exe scripts\audit_evidence_cards.py `
  --checkpoint artifacts\lite_v1\model.pt `
  --data <local-weibo17-processed>\test.jsonl `
  --limit 200 `
  --out artifacts\evidence_card_audit.json
```

## 3. 原创与参考边界

- 本项目为 clean-room 独立实现，不复制 ComP 或其他开源项目源码。
- MIAN、IMOL 等论文只用于研究问题与设计思想参考；代码、命名、目录结构、训练流程、输出字段和界面均独立完成。
- 第一版不下载预训练文本/图像权重，字符级 TextCNN + 小 CNN + 门控融合可在本机离线训练和推理。
- 第三方组件、论文和数据集的来源与边界记录在 `docs/`。
- 不使用任何现有开源项目包装参赛。

## 4. 模型与真实基线

数据：Weibo17 / EANN Weibo，本地处理后 train / val / test = `5,415 / 843 / 1,465`，标签契约为 `0 = real/non-rumor`、`1 = fake/rumor`。原始数据不进入仓库。

模型：约 `1,488,161` 个参数，字符词表 `3,706`，训练提交 `5c11c3e`。测试集阈值 `0.5`：

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.8785 |
| Precision | 0.9082 |
| Recall | 0.8505 |
| F1 | 0.8784 |
| AUC | 0.9437 |
| ECE | 0.0918 |

这些是 v1.0 分类器的真实指标。证据方向卡没有改变模型、权重、损失函数或决策阈值，因此：

- 不把这些指标归因给证据卡；
- 不声称证据卡提升 F1 或 AUC；
- 验证损失后期上升、ECE 偏高和阈值敏感仍然是当前限制。

完整数字与训练历史见本地 `artifacts/lite_v1/metrics.json`；权重与 artifacts 不入库。

## 5. 选择复现路径

| 路径 | 前提 | 可验证范围 |
|---|---|---|
| A 固定效果示例 | Python标准库与源码 | 明确标注的固定55.8%界面示例，不执行模型分析 |
| B 无torch基础验证 | 新venv与轻量依赖 | 基础测试及可选依赖机制，模型测试允许跳过 |
| C 已有检查点真实推理 | 完整环境、model.pt与tokenizer.json | 真实模型HTTP/CLI分析、图文解释与缺图降级 |
| D 完整训练 | 正式研究数据与完整依赖 | 数据适配、训练与分类评估 |

详细命令、实测范围和外部评审准备问题见 [四路径复现指南](docs/reproduction.md)。基础测试与固定示例不能替代真实模型推理。

### 5.1 固定效果示例

```powershell
cd <repo>
python scripts/sync_workbench.py --output work/static_demo/明鉴_工作台.html
python -m http.server 8123 --bind 127.0.0.1 --directory work/static_demo
```

打开`http://127.0.0.1:8123/明鉴_工作台.html?example=review`。配图为AI生成、热区为示意；静态服务器不提供`/api/analyze`。结束后按Ctrl+C。

### 5.2 无torch基础验证

```powershell
python -m venv .venv-light
.\.venv-light\Scripts\python.exe -m pip install numpy==2.5.2 pandas==3.0.6 pillow==12.3.0 pytest==9.1.1 ruff==0.16.10 "setuptools>=69" wheel
.\.venv-light\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
.\.venv-light\Scripts\python.exe -m pytest tests -p no:cacheprovider -o addopts='' -q --basetemp work/pytest_light_local
.\.venv-light\Scripts\ruff.exe check src tests scripts
.\.venv-light\Scripts\python.exe scripts/smoke_test.py
```

模型测试的`skipped`不等于通过。每次复测用新的可写临时目录；Windows直接使用虚拟环境Python，不依赖Activate.ps1。Node不在PATH时可通过`MINGJIAN_NODE`指定现有Node供可选前端回归使用。

## 6. 已有检查点真实推理

需要完整环境及匹配的`model.pt`和`tokenizer.json`。模型文件不随公开仓库或提交包分发；评委可从数据发布方准备完整归档，再通过[评委真实模型运行指南](docs/reviewer_quickstart.md)在本地校验数据、训练新模型并验证CPU推理。依赖安装、数据下载和训练分别计时，不承诺任意电脑冷安装十分钟完成。

```powershell
.\.venv\Scripts\python.exe scripts/explain_lite.py --checkpoint artifacts/lite_v1/model.pt --device cpu --text "这是一条自行编写的流程验收文本，请核查原始来源。" --out work/self_authored_text.json
.\.venv\Scripts\python.exe scripts/run_web_demo.py --checkpoint artifacts/lite_v1/model.pt --device cpu --port 8002
```

打开`http://127.0.0.1:8002/`输入并执行真实分析。服务默认仅监听本机，不需要FastAPI、uvicorn或付费API。`?example=review`入口展示固定示例，页面明确区分示例与新推理结果。

## 7. 完整训练

按数据发布方渠道准备本地Weibo17研究数据，可用自动入口完成完整划分的适配、训练与CPU解释验证：

```powershell
& .\scripts\install_cpu.ps1
.\.venv-cpu\Scripts\python.exe scripts/prepare_local_model.py --archive <local-weibo.zip> --work-dir ..\mingjian-review-run --device cpu --threads 8
.\.venv-cpu\Scripts\python.exe scripts/run_web_demo.py --checkpoint ..\mingjian-review-run\model\model.pt --device cpu --port 8002
```

`--work-dir`必须是仓库外的新目录，已有模型不覆盖。归档获取方式、前提和失败处理见[评委运行指南](docs/reviewer_quickstart.md)。本地复现生成的新模型与既有展示检查点分开保存，新指标以该次`metrics.json`为准，不替换上表历史基线。

## 8. 测试与验证范围

当前测试覆盖：

- 数据契约、标签方向、JSONL 读写和 Weibo17 适配器；
- 字符分词器、模型形状、前向/反向和训练入口；
- 指标与配置；
- 文本证据、Grad-CAM、模态贡献和缺图降级；
- 证据卡三态、低强度、中性方向、方向稳定性、缺图；
- Web 页面、JSON API、会话历史与证据卡界面结构；
- 原创性守卫。

提交冻结前必须现场执行：

```text
pytest tests
ruff check src tests scripts
python scripts/smoke_test.py
python scripts/check_originality_guard.py
python scripts/audit_evidence_cards.py ...
```

所有对外数字以执行当次输出为准，不沿用历史计数。

## 9. 目录结构

```text
mingjian/
├─ configs/                 # 实验配置
├─ data/                    # 数据说明与本地数据目录
├─ docs/                    # 项目章程、模型卡、数据卡、复现说明
├─ scripts/                 # 训练、解释、工作台、审计与数据适配入口
├─ src/mingjian/
│  ├─ data/                 # 数据契约、分词与数据集
│  ├─ evaluation/           # 指标与校准
│  ├─ explain/              # 文本归因、Grad-CAM 与证据方向卡
│  ├─ inference/            # 检查点加载与单条推理
│  ├─ models/               # lite 模型与基础模型
│  ├─ training/             # 训练循环
│  └─ web/                  # 标准库 HTTP 本地工作台
└─ tests/
```

## 10. 当前限制

- 测试 ECE 为 `0.0918`，概率校准尚未闭环；
- 验证损失后期上升，存在过拟合迹象；
- 只在 Weibo17 上训练，跨域、跨平台、跨事件泛化尚未验证；
- 缺图降级通过零化图像特征实现，尚未完成缺模态重训练；
- 10% 强度/稳定性阈值是工程启发式，真实分布仍需审计；
- 语义级图文一致性、事实冲突对和外部事实检索属于第二阶段，不是当前能力；
- Grad-CAM 只是粗粒度相对关注区域，不是因果证明或篡改定位；
- 会话历史只保存在内存，服务重启即丢失。

## 11. 参考论文

- Zhang et al. *Multimodal Inverse Attention Network with Intrinsic Discriminant Feature Exploitation for Fake News Detection*.
- Zeng et al. *IMOL: Incomplete-Modality-Tolerant Learning for Multi-Domain Fake News Video Detection*.

论文只用于研究背景与概念参考，不提供也不替代本项目代码。

## 12. 赛事要求映射

- **过程可复现**：细粒度 Git 提交、环境说明、模型卡、数据卡、真实指标和复现步骤。
- **实用价值**：面向内容审核初筛，输出风险、证据方向、人工复核建议和缺图降级。
- **创新与意图控制**：在同一反向传播中增加可审计的图文证据方向卡；明确拒绝语义一致性和事实核验的过度表述。
- **工程质量**：分层目录、类型标注、测试、异常处理、无 API Key 本地运行。
- **用户体验**：结论、概率、证据方向、句级/区域证据、限制说明和历史记录一体化展示。

## 13. 免责声明

明鉴是辅助研判工具，不构成事实认定或法律意见。所有高风险结论必须经过人工复核。
