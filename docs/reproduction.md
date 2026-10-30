# 明鉴四路径复现指南

更新日期：2026-10-30。适用范围：文本与单张可选配图的本地辅助研判工作台。2026-10-29验证和历史基线保留各自日期；新增本地模型准备流程见[评委真实模型运行指南](reviewer_quickstart.md)。

明鉴的界面示例、基础测试、真实推理和完整训练有不同前提。先选择对应路径，再运行命令；固定示例不能代替真实模型推理，基础测试通过也不能代替完整应用复现。

## 1 选择路径

| 路径 | 需要什么 | 能验证什么 | 本轮状态 |
|---|---|---|---|
| A 固定效果示例 | Python标准库与当前源码 | 导出单一界面源并查看明确标注的固定展示内容 | 已实测导出和HTTP读取；不是模型推理 |
| B 无torch基础验证 | 新venv、轻量依赖与源码 | 数据契约、非模型逻辑及可选跳过机制 | 见本轮复现记录；模型测试会跳过 |
| C 已有检查点真实推理 | 完整依赖、model.pt、tokenizer.json | 真实CPU推理、HTTP工作台、图文输入及缺图降级 | 已在预装完整环境实测；不含依赖和模型下载 |
| D 完整训练 | 发布方完整研究归档与完整依赖 | 自动完成校验、数据适配、训练、评估及CPU解释验证 | 新模型单独生成；既有展示模型和历史指标不覆盖 |

所有命令在仓库根目录执行。Windows使用PowerShell，不运行Activate.ps1。以下的`<repo>`和`<local-...>`是使用者选择的本地位置，不是下载链接。

## 2 路径A 固定效果示例

本路径只使用Python标准库，不安装PyTorch、不读取研究数据、不加载模型权重。

```powershell
cd <repo>
python scripts/sync_workbench.py --output work/static_demo/明鉴_工作台.html
python scripts/sync_workbench.py --output work/static_demo/明鉴_工作台.html --check
python -m http.server 8123 --bind 127.0.0.1 --directory work/static_demo
```

浏览器打开`http://127.0.0.1:8123/明鉴_工作台.html?example=review`。该页面使用固定55.8%示例，配图由AI生成，热区为示意；界面会明确说明。静态HTTP服务不提供`/api/analyze`，不能执行真实研判。结束后在终端按Ctrl+C。

也可以打开导出的HTML离线查看，但真实分析必须进入路径C的模型服务。核验副本时使用`--check`，核验命令不会改写已有副本。

## 3 路径B 无torch基础验证

建议使用与本轮相同的Python 3.13；项目声明支持Python 3.11至3.13，其他版本未在本轮验证。先用`python --version`确认解释器可用；Windows的`python3`有时是无法启动的WindowsApps占位程序。

```powershell
cd <repo>
python -m venv .venv-light
.\.venv-light\Scripts\python.exe -m pip install numpy==2.5.2 pandas==3.0.6 pillow==12.3.0 pytest==9.1.1 ruff==0.16.10 "setuptools>=69" wheel
.\.venv-light\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
.\.venv-light\Scripts\python.exe -c "import importlib.util; print(importlib.util.find_spec('torch'))"
.\.venv-light\Scripts\python.exe -m pytest tests -p no:cacheprovider -o addopts='' -q --basetemp work/pytest_light_local
.\.venv-light\Scripts\ruff.exe check src tests scripts
.\.venv-light\Scripts\python.exe scripts/smoke_test.py
.\.venv-light\Scripts\python.exe scripts/check_originality_guard.py
```

`find_spec('torch')`应为`None`。模型相关测试允许显示`skipped`，跳过不等于通过。安装项目时保留`--no-deps`，否则会按项目依赖自动安装PyTorch；先安装构建工具，再使用`--no-build-isolation`，避免安装过程重新建立不受控的构建环境。

每次复测使用新的可写`--basetemp`目录，避免Windows临时目录权限或旧目录冲突。本轮基础验证使用当前本地工作区的源码快照，而不是远程克隆；下载使用了本机pip缓存，不能据此保证另一台电脑的冷启动时间。首次安装遇到IncompleteRead网络中断，重试后安装成功，失败记录保留。

Node不在PATH时，`tests/test_workbench_assets.py`中的前端回归可能跳过。已有Node时可通过环境变量`MINGJIAN_NODE`指定绝对可执行文件路径，或另外运行`node scripts/check_workbench.cjs`。该环境变量仅指定测试运行时，不是模型密钥。

## 4 路径C 已有检查点真实推理

### 4.1 明确前提

需要完整环境及匹配的一对模型文件：`artifacts/lite_v1/model.pt`与`artifacts/lite_v1/tokenizer.json`。文件由本地训练产生，不随公开仓库或本交付包分发。

本轮实测环境为Python 3.13.15、torch 2.11.0+cu128、torchvision 0.26.0+cu128，推理选择`--device cpu`。已安装CUDA构建的PyTorch也可以在CPU上执行；这不等于本轮重新安装并验证了CPU-only发行包。

已有完整环境时直接执行4.2。新环境先按`requirements.lock`及目标设备准备依赖。已在本机使用的CUDA 12.8安装路径如下，安装耗时不计入已测的推理耗时：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

CPU-only或其他CUDA环境应使用PyTorch针对该平台提供的匹配构建；本轮没有对其他发行包重新做安装实测，不能保证锁定的CUDA包适用所有评委电脑。

### 4.2 启动与真实分析

```powershell
.\.venv\Scripts\python.exe scripts/run_web_demo.py --checkpoint artifacts/lite_v1/model.pt --device cpu --port 8002
```

浏览器打开`http://127.0.0.1:8002/`。先确认模型服务健康状态，再提交自行编写的文本与有权使用的单张配图。`http://127.0.0.1:8002/?example=review`为固定示例入口，页面明确标注，不能把固定示例截图当作实时推理证明。端口占用时选择其他端口。

也可以运行解释CLI，先验证无图输入：

```powershell
.\.venv\Scripts\python.exe scripts/explain_lite.py --checkpoint artifacts/lite_v1/model.pt --device cpu --text "这是一条自行编写的流程验收文本，请核查原始来源和发布时间。" --out work/self_authored_text.json
```

有合法配图时增加`--image <local-self-authored-image.png>`。输出中的`modalities.has_image`对应图片是否参与本次推理；缺图时图像贡献为0，证据卡为`insufficient / missing_image`。这只是执行行为验证，不是准确率评估。

### 4.3 本轮验证范围

2026-10-29在已有完整环境、已有检查点下，通过真实HTTP请求验证health、页面源码一致性、带合成纯色图的分析、仅文本分析和两条历史记录。使用自拟文本与程序生成的纯色PNG，不展示或分发研究样本。计时只覆盖该本机流程，不覆盖依赖安装、模型准备、浏览器人工操作，也不构成性能基准。

### 4.4 评委在本地准备真实模型

公开包不含研究数据与模型权重，克隆仓库后需要按[评委运行指南](reviewer_quickstart.md)从发布方取得完整归档，使用`prepare_local_model.py`在仓库外生成新模型。脚本不下载或重新分发研究数据；归档大小、SHA-256与官方划分均校验，通过后才训练并实际运行CPU解释CLI。依赖安装、数据下载和本地模型准备分别记录；同机新环境的成功不等于另一台电脑冷安装十分钟完成。

## 5 路径D 完整训练

本路径使用公开可获取研究数据，不声称数据已获明确授权。使用者按数据发布方渠道准备本地Weibo17归档；原始正文、图片、完整ID、归档和训练权重不进入Git或公开交付包。

自动入口使用完整7723条官方划分，最多8轮并保留早停机制，不接受`--limit`或跳过完整性校验。指定仓库外新目录，防止覆盖已有数据或模型：

```powershell
.\.venv-cpu\Scripts\python.exe scripts/prepare_local_model.py --archive <local-weibo.zip> --work-dir ..\mingjian-review-run --device cpu --threads 8
```

成功后，在同一工作目录执行脚本打印的工作台命令。`preparation_report.json`记录各阶段状态、耗时和该次新模型的聚合指标，详细日志和含路径、样本ID的训练文件留在本地。2026-10-30起，训练种子在构造模型前设置；新训练结果不能冒称旧检查点的权重或F1。

```powershell
.\.venv\Scripts\python.exe scripts/build_weibo17_local.py --archive <local-weibo.zip> --out <local-weibo17-processed>
.\.venv\Scripts\python.exe scripts/train_lite_baseline.py --data-dir <local-weibo17-processed> --out artifacts/lite_v1 --epochs 8 --batch-size 64 --lr 2e-3 --num-workers 8 --device cuda
```

使用官方ID划分：train/val/test为5415/843/1465，合计7723。标签固定`0=real`、`1=fake`。完整适配、训练与评估不属于“10分钟基础验证”承诺，本轮未重新训练分类器。训练产物包括model.pt、tokenizer.json、metrics.json及预测CSV，仅在本地保留。

已有研究测试集时可运行证据方向卡审计：

```powershell
.\.venv\Scripts\python.exe scripts/audit_evidence_cards.py --checkpoint artifacts/lite_v1/model.pt --data <local-weibo17-processed>/test.jsonl --limit 200 --min-strength 0.1 --min-coherence 0.1 --out work/evidence_card_audit.json
```

审计结果只描述该检查点、采样范围和规则，不能外推为图文语义一致性或事实认定。比例只使用真实输出；若几乎只出现一种状态，应披露证据卡区分能力不足。

## 6 环境变量与产品口径

lite推理无需API Key。当前`.env.example`中的预训练编码器字段为后续扩展预留，lite脚本不因此下载RoBERTa或ViT；设备由`--device`参数控制。无需为本轮推理填入任何外部密钥，真实`.env`不提交。

40%至60%含端点的评分统一展示为“待复核”。分类评估阈值0.5与产品待复核区间用途不同。Grad-CAM只解释模型相对关注区域，证据方向卡只比较梯度贡献，不替代外部核查。

## 7 提交前复核

按最终源码版本重跑所选路径并保存真实日志，区分环境安装、模型准备、推理和测试耗时。检查仓库最终提交与远程是否一致。历史Prompt和旧验证原件不回写；本轮发布验证另存带日期记录。
