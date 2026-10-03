# 明鉴（MingJian）

> 多模态虚假新闻检测与可解释审核工作台。
> 当前阶段：M1 仓库骨架完成，CUDA 12.8 环境与 baseline 冒烟测试已验证。

## 1. 项目定位

明鉴面向内容审核与事实核查场景，输入一条新闻文本和一张配图，输出：

- 假/真概率与风险等级；
- 图文一致性与矛盾性分数；
- 文本证据、图像区域证据、跨模态冲突对；
- 结构化解释与人工复核建议；
- 模态缺失或质量不足时的降级结果与不确定性提示。

第一版只做 **文本 + 单图**。视频、音频、OCR 全链路放到后续阶段。

## 2. 原创与参考边界

- 本项目为 clean-room 自研实现，不复制 ComP 或其他开源项目源码。
- MIAN、IMOL 等论文只作为研究思想参考，所有代码、模块命名、训练流程、实验设计均独立完成。
- 使用 BERT/RoBERTa、ViT/CLIP 等预训练模型时，会在模型卡中声明版本、来源和许可证。
- 不使用任何现有开源项目包装参赛；第三方组件与数据集的使用边界见 `docs/`。

## 3. 当前状态

已完成：

- Python 包骨架与配置；
- 新闻样本数据契约；
- JSONL 数据读写工具；
- 二分类与校准评测指标；
- PyTorch 特征融合 baseline；
- 可跳过大依赖的冒烟测试；`r`n- CUDA 12.8 / RTX 5060 环境验证；
- 项目章程、数据卡/模型卡模板、Prompt 留痕模板。

尚未完成：

- 跨平台依赖锁文件与离线复现包；
- 真实数据集接入；
- 文本/图像编码器接入；
- 双通道跨模态交互模型；
- 可靠性路由器与解释器；
- Web 审核工作台；
- 训练、消融与复现实验。

## 4. 快速开始（开发环境尚未安装 PyTorch 时）

```powershell
cd C:\Users\LX\Documents\Codex\2026-10-03\new-chat\outputs\chuanzhibei\01_源码与仓库\mingjian
python -m pytest tests
python scripts\smoke_test.py
```

`scripts\smoke_test.py` 会在没有 PyTorch 时跳过模型前向测试，仍会验证数据契约与评测指标。

安装完整依赖（需要联网，CPU 版示例）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

GPU 环境（本机 RTX 5060 Laptop 建议 CUDA 12.8）：

```powershell
.\scripts\install_gpu_cu128.ps1 -BypassProxy
```

其他 GPU 型号请按显卡架构和驱动选择对应 CUDA 版本的 PyTorch，再执行 `pip install -e ".[dev]"`。

## 5. 目录结构

```text
mingjian/
├─ configs/                 # 实验配置
├─ data/                    # 数据说明与本地数据目录
├─ docs/                    # 项目章程、模型卡、数据卡、留痕规范
├─ scripts/                 # 冒烟测试、数据生成、训练入口
├─ src/mingjian/            # 项目源码
│  ├─ data/                 # 数据契约与数据集
│  ├─ evaluation/           # 指标与校准
│  ├─ models/               # 模型与编码器
│  ├─ training/             # 训练循环
│  └─ utils/                # 通用工具
└─ tests/                   # 测试
```

## 6. 参考论文

- Zhang et al. *Multimodal Inverse Attention Network with Intrinsic Discriminant Feature Exploitation for Fake News Detection*.
- Zeng et al. *IMOL: Incomplete-Modality-Tolerant Learning for Multi-Domain Fake News Video Detection*.

论文只用于研究背景与概念参考，不提供也不替代本项目代码。

## 7. 比赛要求映射

- **过程可复现**：细粒度 Git 提交、环境锁定、模型卡、数据卡、一键复现。
- **实用价值**：审核工作台、风险分级、批量检测、人工复核闭环。
- **方法创新与意图控制**：一致性/矛盾性双通道、可靠性路由、置信度校准、审核模式切换。
- **工程质量**：分层目录、类型标注、测试、日志、异常处理。
- **用户体验**：结论 + 证据 + 限制说明一体化展示。

## 8. 免责声明

明鉴是辅助研判工具，不构成事实认定或法律意见。所有高风险结论必须经过人工复核。