# 明鉴复现说明 v0.2

> 目标：评委先在干净环境验证项目结构与基础测试；若已有本地数据和训练产物，再完成端到端训练、解释和工作台复现。
> 本机验证环境（2026-10-03）：Python 3.13.15、torch 2.11.0+cu128、CUDA 12.8、RTX 5060 Laptop GPU 8 GB。

## 1. 10 分钟极速复现（基础路径）

这条路径不下载模型权重，也不需要 Weibo17 原始数据。为避免 Windows PowerShell 的 `Activate.ps1` 执行策略问题，所有命令都直接使用虚拟环境中的 `python.exe`。

### 1.1 轻量路径（不安装 PyTorch，适合快速核验）

```powershell
cd <repo>
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy==2.5.2 pandas==3.0.6 pillow==12.3.0 pytest==9.1.1 ruff==0.16.10
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

当前验证结果（无 PyTorch 干净环境）：

- `pytest tests`：`25 passed, 8 skipped`；
- `ruff check src tests scripts`：`All checks passed!`；
- `scripts\smoke_test.py`：数据契约、配置和指标检查通过；模型前向/反向步骤显示 `SKIP: PyTorch is not installed`；
- 8 条跳过项均为 PyTorch 相关测试，不是失败。

### 1.2 完整路径（安装锁定依赖和 PyTorch）

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

开发机完整环境（torch 2.11.0+cu128）验证结果：`pytest tests` 为 `60 passed`，`ruff` 为 `All checks passed!`，`smoke_test.py` 的模型前向/反向检查通过。完整路径需要下载 PyTorch，耗时会随网络和缓存变化。
## 2. 本地数据准备（不随仓库分发）

Weibo17 原始归档只保存在本地，不允许把正文、完整 ID 列表或图片提交到 Git。准备完成后运行：

```powershell
python scripts\build_weibo17_local.py `
  --archive <local-weibo.zip> `
  --out <local-weibo17-processed>
```

输出至少包含：

- `train.jsonl`、`val.jsonl`、`test.jsonl`；
- 本地图片目录；
- `manifest.json` 审计信息。

核验口径：官方 ID 划分 `7,723` 条，train/val/test = `5,415 / 843 / 1,465`；标签契约为 `0 = real/non-rumor`、`1 = fake/rumor`。详细来源、校验和与风险披露见 `docs/data_card.md` 和 `docs/open_source_checklist.md`。

## 3. 训练 lite baseline

```powershell
python scripts\train_lite_baseline.py `
  --data-dir <local-weibo17-processed> `
  --out artifacts\lite_v1 `
  --epochs 8 `
  --batch-size 64 `
  --lr 2e-3 `
  --num-workers 8 `
  --device cuda
```

产物：

- `artifacts\lite_v1\model.pt`：检查点（git-ignored）；
- `artifacts\lite_v1\tokenizer.json`：字符分词器；
- `artifacts\lite_v1\metrics.json`：环境、数据划分、超参数、训练历史和评测指标；
- `predictions_val.csv`、`predictions_test.csv`：逐样本预测，方便核对。

本机真实结果（提交 `5c11c3e`）：

| 测试集指标（阈值 0.5） | 数值 |
|---|---:|
| Accuracy | 0.8785 |
| Precision | 0.9082 |
| Recall | 0.8505 |
| F1 | 0.8784 |
| AUC | 0.9437 |
| ECE | 0.0918 |

已知风险：验证损失后期上升，存在过拟合；阈值 0.05 虽在验证集 F1 较高，但测试集 F1 为 0.8770，低于阈值 0.5。当前不声明概率已校准。

## 4. 解释器复现

```powershell
python scripts\explain_lite.py `
  --checkpoint artifacts\lite_v1\model.pt `
  --text "某条新闻文本" `
  --image D:\local\news.jpg `
  --out artifacts\one_explanation.json
```

不提供 `--image` 时会执行缺图降级，结构化结果中的 `modalities.has_image` 为 false，图像贡献为 0。文本证据来自梯度 x embedding 归因，图像证据来自 Grad-CAM 粗粒度热图。

## 5. 无 API Key 工作台复现

```powershell
python scripts\run_web_demo.py `
  --checkpoint artifacts\lite_v1\model.pt `
  --device auto
```

浏览器打开终端打印的本地地址。当前 MVP 使用 Python 标准库 HTTP 服务，不要求 FastAPI、uvicorn 或任何付费 API；默认仅监听 `127.0.0.1`。工作台支持单条检测、缺图输入、文本证据、Grad-CAM 热图和本次会话历史。

## 6. 环境变量

复制 `.env.example` 为本地 `.env` 后再修改，提交时不要包含真实密钥。lite MVP 推理不要求 API Key；环境变量只用于后续扩展外部服务。

## 7. 复现边界与注意事项

- 原始数据、图片和模型权重不得提交到仓库；
- 所有指标必须来自真实运行，不得把计划值写成已完成值；
- 使用 GPU 时按显卡架构选择匹配的 CUDA/PyTorch 版本；
- 若系统配置了失效代理，可使用 `.\scripts\install_gpu_cu128.ps1 -BypassProxy`；
- 10 分钟极速复现只保证基础测试与已有检查点推理；完整训练时间取决于数据和 GPU；
- 比赛最终提交前需要补充独立干净环境实测记录、依赖锁文件和演示视频。
