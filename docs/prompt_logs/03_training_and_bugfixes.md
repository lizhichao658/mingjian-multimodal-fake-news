# Prompt 链 03：lite baseline 训练、解释器与真实 Bug 修复

> 日期：2026-10-03  
> 负责人：参赛者本人  
> 关联提交：`5195f2b`、`53e1510`、`5c11c3e`、`4b89fb4`、`d49e12f`  
> 核心目标：在 Weibo17 上真实训练文本 + 单图模型，输出可复核指标，并修复训练和解释器中的关键问题。

---

## 1. 初始意图

### 1.1 原始 Prompt（本人关键原句，按对话留痕整理）

```text
我现在该干什么？
按 MVP 推进。
只用 Weibo17 就行。
```

### 1.2 人工设定的验收标准

- 必须用真实 Weibo17 数据训练，不用随机伪造指标；
- 必须保存检查点、tokenizer、metrics 和预测 CSV；
- 必须能在 8 GB 显存 GPU 上完成；
- 训练代码要有单元测试；
- 解释器必须处理缺图，不得因为缺模态崩溃；
- 任何调参结果必须清楚说明阈值和限制；
- 不在代码中引入 ComP 或论文作者源码。

---

## 2. AI 建议

AI 建议的训练闭环：

1. 字符 tokenizer + Weibo17 Dataset；
2. 多核字符 TextCNN + 小 CNN + 门控融合；
3. AdamW、AMP、余弦退火、类别权重；
4. 以验证 AUC 选择检查点；
5. 在验证集搜索阈值；
6. 输出混淆矩阵、F1、AUC、ECE；
7. 用真实训练日志生成模型卡，不先填数字；
8. 用同一模型输出梯度归因和 Grad-CAM。

### 人工修改要求

- 不要只看 accuracy，增加 AUC、F1、ECE；
- 不要用测试集调阈值；
- 不要忽略验证损失上升；
- 不要把 dropout、权重衰减和早停写成“已经解决过拟合”；
- 不要把 Grad-CAM 解释成图像篡改证据。

---

## 3. 真实 Bug 1：`--limit` 按文件顺序截断导致类别偏置

### 现象

- 为了让快速冒烟训练节省时间，使用 `--limit` 读取少量样本；
- 截断按 JSONL 文件顺序执行，结果 train/val/test 恰好集中到同一类别；
- 验证集 AUC 出现 `NaN`，无法判断优化器是否真的工作。

### 根因

- 数据文件顺序与类别分布相关，直接 `head` 式截断不是随机或分层抽样；
- 小规模冒烟数据不能代表全量指标，也不能用于选模型。

### 修复

- 将 `--limit` 改为类别均衡抽样；
- 新增测试覆盖两类均有样本；
- 训练报告明确区分“冒烟运行”和“正式训练”；
- 正式结果只在完整 train/val/test 上生成。

### 关联提交

- `53e1510`：训练器、阈值调优和早停；
- `5c11c3e`：真实 Weibo17 端到端训练入口。

---

## 4. 真实 Bug 2：TextCNN 最大池化污染 padding

### 现象

- 变长文本 padding 后，卷积窗口可以覆盖到填充位置；
- 未加 mask 时，padding 位置的激活可能被最大池化选中，污染文本表示。

### 根因

- 常规 TextCNN 直接对卷积输出做 global max pooling；
- 这里使用字符序列和 padding，必须结合 attention mask 计算有效窗口。

### 修复

- 根据 `attention_mask` 计算有效长度；
- 对每个卷积核检查窗口是否完全落在有效 token 内；
- 无效窗口置为 `-inf`，池化时被 ReLU 和 clamp 处理；
- 增加 padding 相关模型测试。

### 关联提交

- `5195f2b`：模型实现；
- `tests/test_lite_model.py`：形状与 mask 回归。

---

## 5. 真实 Bug 3：解释器 Grad-CAM 网格和梯度张量错误

### 现象 A

- 测试最初假设 Grad-CAM 输出为 `2×2`；
- 小测试模型（16×16 输入）实际得到 `4×4` 特征图；真实 MVP 模型（128×128 输入，4 层 stride-2 卷积）得到 `8×8` 特征图。

### 根因

- 只凭直觉写了网格尺寸，没有按卷积下采样链路计算；
- 测试断言与实际模型结构不一致。

### 修复 A

- 修正测试为根据真实特征图形状断言；
- 解释器输出 `grid_shape`，避免调用方假设尺寸；
- 关联提交 `4b89fb4`。

### 现象 B

- 解释器把带梯度的 tensor 直接转成 Python float 时出现错误；
- 需要先从计算图中分离并移动到 CPU。

### 根因

- 梯度张量仍带 `requires_grad` 和 GPU 设备；
- JSON 序列化要求普通 Python 数值。

### 修复 B

- 统一使用 `value.detach().cpu()` 后再 `tolist()`/`float()`；
- 增加有图、缺图、异常输入测试；
- 关联提交 `4b89fb4`、`d49e12f`。

---

## 6. 真实 Bug 4：检查点阈值与推理决策阈值不一致

### 现象

- `artifacts/lite_v1/model.pt` 里记录的调优阈值是 `0.05`（验证集搜索得到）；
- 推理脚本直接拿 `0.05` 当决策阈值，而 README 和模型卡却声明默认使用更稳健的 `0.5`；
- 结果是“文档说一套、实际跑一套”，同一句话可能因为阈值不同得到不同结论。

### 根因

- 检查点只存了一个阈值字段，没有区分“审计用途的检查点阈值”和“实际决策阈值”；
- 文档更新后没有回归校验推理输出里的阈值字段。

### 修复

- `LitePredictor` 拆分 `threshold`（检查点记录的阈值，仅审计）和 `decision_threshold`（实际决策阈值，默认 `0.5`）；
- `analyze(..., decision_threshold=None)` 支持临时覆盖，输出同时给出 `checkpoint_threshold`、`decision_threshold` 和 `decision.threshold`；
- 更新 `tests/test_lite_predictor.py`，断言默认决策阈值 `0.5`、检查点阈值 `0.4`，并验证显式覆盖；
- 真实 CLI 验收：`decision.threshold == 0.5`、`checkpoint_threshold == 0.05`、`decision_threshold == 0.5`；
- 关联提交 `3f5a50a`。

---

## 7. 真实训练结果

训练命令：

```powershell
.\.venv\Scripts\python.exe scripts\train_lite_baseline.py `
  --data-dir "C:\Users\LX\Documents\Codex\2026-10-03\new-chat\work\weibo17_processed" `
  --out artifacts\lite_v1 `
  --epochs 8 --batch-size 64 --lr 2e-3 `
  --num-workers 8 --device cuda
```

运行摘要：

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU；
- 训练时长：`71.66s`；
- epochs：8；
- 最佳验证 AUC：`0.94365`；
- 训练损失：`0.4085 → 0.0186`；
- 验证损失：`0.3189 → 0.4887`（后期上升，存在过拟合）；
- 验证集搜索阈值：`0.05`；
- 测试集阈值 0.5：F1 `0.8784`、AUC `0.9437`、ECE `0.0918`；
- 测试集使用验证阈值 0.05：F1 `0.8770`，低于阈值 0.5。

### 人工判断

- 保留完整训练历史，不删除后期高验证损失；
- 默认工作台阈值仍用 `0.5`，不采用无法稳定外推的 `0.05`，并由 `LitePredictor.decision_threshold=0.5` 固化、回归测试兜底（见第 6 节）；
- 在模型卡和 README 中明确概率未充分校准；
- 不因为 AUC 尚可就声称跨域泛化已经解决。

---

## 8. 回归测试与验证命令

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_lite_model.py tests\test_lite_explainer.py tests\test_train_script.py `
  --basetemp="...\pytest_tmp_training_chain" -p no:cacheprovider
```

输出摘要：

```text
................                                                         [100%]
16 passed in 1.61s
```

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

输出摘要：

```text
All checks passed!
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests --basetemp="...\pytest_tmp_prompt_chains" -p no:cacheprovider
```

输出摘要：

```text
............................................................             [100%]
60 passed in 3.39s
```

---

## 9. 结论

本链展示了从“可运行”到“可解释”的真实迭代：AI 负责提出训练结构与修复方向，人工负责识别类别偏置、核实 padding mask、拒绝阈值幻觉、保留失败结果。第一轮模型已经能用于 MVP 演示，但概率校准、过拟合和跨域泛化仍需第二轮实验。