# 数据目录说明

- `raw/`：原始数据，只读，不提交 Git。
- `processed/`：清洗后的训练/验证/测试 JSONL，不提交 Git。
- `demo/`：可公开展示的小样例；其中的 `sample_cases.jsonl` 为接口示例，标注来源与用途。
- `demo/generated/`：脚本生成的合成测试图片，不提交 Git。

## JSONL 字段

每行一个 JSON 对象：

```json
{
  "sample_id": "demo-0001",
  "text": "新闻文本内容",
  "image_path": "data/demo/images/demo-0001.png",
  "label": 0,
  "split": "test",
  "source": "demo",
  "metadata": {"note": "仅用于接口演示"}
}
```

- `label`：0 表示真实/非谣言新闻，1 表示虚假/谣言新闻（EANN Weibo17 官方协议）。
- `split`：`train`、`val`、`test`。
- 不得把未经许可的数据集或个人信息提交到公开仓库。