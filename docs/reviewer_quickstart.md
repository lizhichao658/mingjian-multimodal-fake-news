# 评委真实模型运行指南

更新：2026-10-30。适用Windows PowerShell、Python 3.11至3.13；CPU也可完成训练及推理，不需要API Key。真实模型需要先准备研究归档并在本地训练，固定55.8%示例不执行模型推理。

## 1 获取源码和完整数据

```powershell
git clone https://github.com/lizhichao658/mingjian-multimodal-fake-news.git mingjian
cd mingjian
```

从[EANN数据发布方仓库](https://github.com/yaqingwang/EANN-KDD18)的完整Weibo数据入口自行取得`weibo.zip`，不要使用仓库快速开始子集。发布方当前指向[完整归档下载页](https://drive.google.com/file/d/14VQ7EWPiFeGzxp3XC2DeEHi-BEisDINn/view)；访问可能需要Google账号与相应网络，不能保证所有地区免登录下载。

完整归档大小为1,359,648,341字节，SHA-256为：

```text
06b4d5edc978ab385a75bdc064cb2d2a200e09953ac38ecc1b1e80d652381f68
```

将归档保存在仓库外。本项目只使用公开可获取研究数据，不重新分发原始文本、图片、完整ID或模型权重，不声称获得额外授权。数据下载时间单独计算。

## 2 安装独立CPU环境

```powershell
& .\scripts\install_cpu.ps1
```

该脚本创建新的`.venv-cpu`，先从[PyTorch官方CPU索引](https://download.pytorch.org/whl/cpu)安装匹配的torch/torchvision，再安装`requirements-cpu.lock`及当前项目。每一步失败即停止；已有环境不覆盖。需要其他位置时，指定`-EnvironmentPath <new-local-env>`；Python不在PATH时可用`-PythonExecutable <python.exe>`。

不使用Activate.ps1，也不要求改变系统执行策略。若机器策略禁止运行项目脚本，可逐行执行下列等价安装命令（现有`.venv-cpu`不要重复创建）：

```powershell
python -m venv .venv-cpu
.\.venv-cpu\Scripts\python.exe -m pip install "setuptools>=69" wheel
.\.venv-cpu\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-cpu\Scripts\python.exe -m pip install -r requirements-cpu.lock
.\.venv-cpu\Scripts\python.exe -m pip install -e ".[dev]" --no-deps --no-build-isolation
```

网络或安装失败时先修复对应步骤；不能跳过失败继续声称环境就绪。CUDA环境可按原四路径指南准备，不能把CUDA包当成CPU安装已验证。

## 3 校验数据并生成真实模型

将下列`<local-weibo.zip>`替换为刚取得的完整归档路径，保留引号；`..\mingjian-review-run`是仓库外一个尚不存在或为空的目录：

```powershell
.\.venv-cpu\Scripts\python.exe scripts/prepare_local_model.py --archive "<local-weibo.zip>" --work-dir ..\mingjian-review-run --device cpu --threads 8
```

脚本顺序执行：

1. 校验完整归档大小和SHA-256，按官方ID建立5415/843/1465条划分，共7723条，标签0=真实、1=虚假。
2. 使用全部划分训练，最多8轮，保留早停和验证集选择机制；禁止用冒烟子集代替正式训练。
3. 检查`model/model.pt`、`model/tokenizer.json`和`model/metrics.json`。
4. 用新模型在CPU执行自拟文本解释，验证概率有效、缺图及证据不足降级。
5. 只有上述步骤全部通过才记录`status=passed`并打印工作台启动命令。

工作目录包含`processed/`、`model/`、`logs/`、`cpu_health.json`和`preparation_report.json`。这些都是本地产物，不上传到源码仓库或公开提交包。失败时报告保留失败阶段和退出码，查看对应本地日志；换一个新工作目录重试，不覆盖原结果。

`--threads`控制CPU并行预算，`--num-workers`默认0以兼容Windows；可按硬件调整。训练速度取决于CPU、存储及并行配置。新训练的指标以本次`model/metrics.json`为准，不能冒称历史展示模型F1=0.8784或逐位一致权重。

## 4 启动并执行真实研判

```powershell
.\.venv-cpu\Scripts\python.exe scripts/run_web_demo.py --checkpoint ..\mingjian-review-run\model\model.pt --device cpu --port 8002
```

保持终端开启，用浏览器打开`http://127.0.0.1:8002/`，确认“本地服务已连接”，输入自拟文本及有权使用的单张图片，再点击“开始研判”。移除图片再次分析，应显示缺图降级。服务仅在本机处理，不依赖外部API。

默认采用工作台与API同源访问；不接受`Origin:null`、外站或跨端口网页调用。本地命令行无Origin请求可以用于健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8002/api/health
```

`?example=review`和“查看效果示例”只展示明确标注的固定数值及示意热区。真实分析的结果可能不同；Grad-CAM表示模型相对关注区域，不能替代事实核查。

## 5 验证口径

环境安装、官方数据下载、模型准备和HTTP推理分别计时。发布验证记录注明操作系统、Python/CPU构建、代码提交和数据前提。同一电脑的新环境、新克隆实测不能概括为所有评委电脑冷安装十分钟完成。

已有匹配模型时可直接从第4步启动；只有源码而未准备数据或模型时，可以运行固定示例和基础测试，但不能把它们称为真实模型分析。
