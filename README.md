# 明鉴（MingJian）

> 多模态虚假新闻检测与可解释审核工作台。
> 当前阶段：**M2 文本 + 单图 MVP 已端到端跑通**（真实 Weibo17 训练、解释器、无 API Key 本地工作台）。

## 1. 项目定位

明鉴面向内容审核与事实核查场景。第一版输入一条新闻文本和一张可选配图，输出：

- 假/真概率与风险等级；
- 文本证据与图像 Grad-CAM 区域证据；
- 模态贡献与缺图降级说明；
- 结构化解释与人工复核建议；
- 可追溯的本地会话历史。

第一版只做 **文本 + 单图**。视频、音频、OCR 全链路放到后续阶段。系统只做辅助研判，不构成事实认定或法律意见。

## 2. 原创与参考边界

- 本项目为 clean-room 自研实现，不复制 ComP 或其他开源项目源码。
- MIAN、IMOL 等论文只作为研究思想参考，所有代码、模块命名、训练流程和实验设计均独立完成。
- 第一版模型不依赖预训练权重下载：字符级 TextCNN + 小 CNN + 门控融合，可在本机离线训练和推理。
- 不使用任何现有开源项目包装参赛；第三方组件与数据集的使用边界见 `docs/`。

## 3. 当前状态

已完成：

- Python 包骨架、配置、JSONL 数据契约和评测指标；
- Weibo17 官方归档字段、ID 划分、标签方向、图文配对和本地适配器；
- 字符级 `CharTokenizer` 与 PyTorch 数据集；
- 自包含 `TextCNN + ImageCNN + gated fusion` lite 模型；
- 真实 Weibo17 训练入口、阈值调优、早停和产物保存；
- 梯度文本证据、图像 Grad-CAM、模态贡献和缺图降级解释器；
- 标准库 HTTP 本地工作台：单条分析、历史记录、无 API Key 演示；
- 60 项测试与 ruff 静态检查；
- CUDA 12.8 / RTX 5060 环境验证。

尚未完成或需要继续改进：

- 概率校准：验证/测试 ECE 仍偏高，阈值附近结果必须人工复核；
- 训练曲线显示后期验证损失上升，需要正则化、早停策略和更多真实实验；
- 图文一致性/矛盾性显式评分仍处于后续迭代；
- 独立干净环境、10 分钟极速复现包尚未最终锁定；
- 演示视频、PDF 技术文档和远程仓库推送尚未完成。

## 4. 真实基线结果

数据：Weibo17 / EANN Weibo，本地处理后 train/val/test = `5,415 / 843 / 1,465`，标签契约为 `0 = real/non-rumor`、`1 = fake/rumor`。原始数据不进入仓库。

模型：约 `1,488,161` 参数，字符词表 `3,706`，训练提交 `5c11c3e`，GPU 训练耗时约 `71.66s`。

测试集（阈值 `0.5`）：

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.8785 |
| Precision | 0.9082 |
| Recall | 0.8505 |
| F1 | 0.8784 |
| AUC | 0.9437 |
| ECE | 0.0918 |

验证集调优阈值 `0.05` 时测试 F1 为 `0.8770`。阈值过低且概率校准未闭环，因此解释器默认仍以 `0.5` 作为决策阈值显示，并在结构化结果中报告检查点阈值。完整数字和训练历史见本地 `artifacts/lite_v1/metrics.json`；权重不入库。

## 5. 快速开始

基础测试（不要求下载模型）：

```powershell
cd <repo>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest tests
python scripts\smoke_test.py
```

已锁定版本的复现安装（推荐用于评审复现，见 `requirements.lock`）：

```powershell
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

`requirements.lock` 记录的是本机验证过的精确版本（含 CUDA 12.8 的 torch/torchvision）。纯 CPU 机器请把 `torch==2.11.0+cu128`、`torchvision==0.26.0+cu128` 换成不带 `+cu128` 的对应版本。

GPU 环境（本机 RTX 5060 Laptop 建议 CUDA 12.8）：

```powershell
.\scripts\install_gpu_cu128.ps1 -BypassProxy
```

体验解释器（给定本地检查点和可选图片）：

```powershell
python scripts\explain_lite.py `
  --checkpoint artifacts\lite_v1\model.pt `
  --text "某条新闻文本" `
  --image D:\local\news.jpg `
  --out artifacts\one_explanation.json
```

启动无 API Key 本地工作台：

```powershell
python scripts\run_web_demo.py `
  --checkpoint artifacts\lite_v1\model.pt `
  --device auto
```

浏览器打开终端输出的本地地址即可。默认只监听 `127.0.0.1`。

## 6. 复现训练

假设 Weibo17 已经按 `docs/data_card.md` 的合规边界处理到本地目录：

```powershell
python scripts\train_lite_baseline.py `
  --data-dir <weibo17_processed> `
  --out artifacts/lite_v1 `
  --epochs 8 --batch-size 64 --lr 2e-3 `
  --num-workers 8 --device cuda
```

训练会输出 `model.pt`、`tokenizer.json`、`metrics.json`、验证/测试预测 CSV。`artifacts/` 和权重文件默认被 `.gitignore` 忽略。

## 7. 目录结构

```text
mingjian/
├─ configs/                 # 实验配置
├─ data/                    # 数据说明与本地数据目录
├─ docs/                    # 项目章程、模型卡、数据卡、留痕规范
├─ scripts/                 # 训练、解释、工作台、数据适配入口
├─ src/mingjian/            # 项目源码
│  ├─ data/                 # 数据契约、分词与数据集
│  ├─ evaluation/           # 指标与校准
│  ├─ explain/              # 文本归因与 Grad-CAM
│  ├─ inference/            # 检查点加载与单条推理
│  ├─ models/               # lite 模型与基础模型
│  ├─ training/             # 训练循环
│  └─ web/                  # 无外部依赖本地工作台
└─ tests/                   # 测试
```

## 8. 参考论文

- Zhang et al. *Multimodal Inverse Attention Network with Intrinsic Discriminant Feature Exploitation for Fake News Detection*.
- Zeng et al. *IMOL: Incomplete-Modality-Tolerant Learning for Multi-Domain Fake News Video Detection*.

论文只用于研究背景与概念参考，不提供也不替代本项目代码。

## 9. 比赛要求映射

- **过程可复现**：细粒度 Git 提交、环境说明、模型卡、数据卡、真实指标和复现步骤。
- **解题有效性**：真实 Weibo17 端到端训练，测试集 F1/AUC 可核验。
- **方法创新**：门控融合、梯度文本证据、Grad-CAM、缺图降级和可靠性提示的组合。
- **工程质量**：分层目录、类型标注、测试、异常处理、无 API Key 本地运行。
- **用户体验**：结论 + 概率 + 证据 + 限制说明 + 历史记录一体化展示。

## 10. 免责声明

明鉴是辅助研判工具，不构成事实认定或法律意见。所有高风险结论必须经过人工复核。
