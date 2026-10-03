# 复现说明 v0.1（草稿）

> 目标：评委在干净环境中 10 分钟内看到项目结构和基础测试；完整模型推理路径在依赖与数据准备完成后补充。

## 当前可复现内容

1. 安装基础依赖；
2. 运行数据契约、配置与指标测试；
3. 在安装 PyTorch 后运行模型前向/反向冒烟测试。

## 最小步骤

```powershell
cd <repo>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts\smoke_test.py
python -m pytest tests
```

## 注意事项

- 无 GPU 时使用 CPU 模式；
- 模型权重下载需要网络或提前放置缓存；
- 比赛最终版本必须提供“无 API Key 演示模式”；
- 所有实验需记录 Git 提交、配置、随机种子与指标；
- 依赖锁定文件将在环境确定后生成。