# 第 6 章：分词、验证集与手工测试集

这一章讲两个层面：

1. 训练期要用的 tokenizer 和 `eval.jsonl`
2. 推理验收期要用的人工测试提示集

## 1. CatLM 的 tokenizer 是怎么来的

CatLM 没有复用原有 `data/tokenizer.json`。

它会在 [catlm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/catlm/prepare_data.py) 中重新训练一份：

- `data_cat_zh/tokenizer.json`

这一步对中文非常重要，因为你现在的语料分布已经从英文小鱼换成了中文小猫。

## 2. tokenizer 的特殊 token 仍然和 GuppyLM 对齐

CatLM 仍然保留：

- `<pad>`
- `<|im_start|>`
- `<|im_end|>`

这样做的好处是：

- 数据主格式不需要改
- 推理 prompt 逻辑不需要改
- 训练循环也不需要改

## 3. 如何确认 tokenizer 可用

你可以直接加载它：

```bash
./.conda-py311-mps/bin/python - <<'PY'
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data_cat_zh/tokenizer.json")
text = "<|im_start|>user\n你在窗边看什么<|im_end|>\n<|im_start|>assistant\n窗外有小鸟。我得盯着。<|im_end|>"
ids = tokenizer.encode(text).ids

print("token 数:", len(ids))
print(ids[:20])
print(tokenizer.decode(ids))
PY
```

## 4. 训练期的“测试数据集”是什么

对 CatLM 来说，训练期的测试集其实就是：

- `data_cat_zh/eval.jsonl`

它的作用和 GuppyLM 一样：

- 不参与梯度更新
- 只用于评估当前模型的 loss

## 5. `eval.jsonl` 和 `eval_openai.jsonl` 的区别

### 5.1 `eval.jsonl`

训练代码真正直接读取的是这个文件。

### 5.2 `eval_openai.jsonl`

这是同一批验证样本的 messages 格式导出版本，主要方便：

- 人工查看
- 导出到别的工具链
- 和外部格式打通

## 6. 推理期的人工测试集是什么

CatLM 另外提供了一组手工测试提示，放在：

- [catlm/eval_cases.py](/Users/changzechuan/AIProjects/guppylm/catlm/eval_cases.py)

这组文件不是训练必需，但建议在训练完以后用来验收模型的人设稳定性。

它覆盖的典型场景包括：

- 问候
- 食物
- 窗边观察
- 小鸟
- 噪音
- 打雷害怕
- 晒太阳
- 主人关系
- 抽象问题困惑
- 睡觉
- 玩耍
- 洗澡 / 水
- zoomies
- 看医生
- 巡视领地
- 道别

## 7. 如何查看人工测试提示集

```bash
./.conda-py311-mps/bin/python - <<'PY'
from catlm.eval_cases import get_eval_cases

for case in get_eval_cases():
    print(case["id"], "=>", case["prompt"])
PY
```

## 8. 一套推荐的“准备完成”检查

在开始正式训练前，建议至少跑下面三段检查。

### 8.1 数据结构检查

```bash
./.conda-py311-mps/bin/python - <<'PY'
import json

for path in ["data_cat_zh/train.jsonl", "data_cat_zh/eval.jsonl"]:
    with open(path, encoding="utf-8") as f:
        row = json.loads(next(f))
        assert "text" in row
        assert row["text"].startswith("<|im_start|>user\n")
        assert "<|im_start|>assistant\n" in row["text"]
        assert row["text"].endswith("<|im_end|>")

print("主训练格式检查通过")
PY
```

### 8.2 tokenizer 加载检查

```bash
./.conda-py311-mps/bin/python - <<'PY'
from tokenizers import Tokenizer
Tokenizer.from_file("data_cat_zh/tokenizer.json")
print("tokenizer 加载通过")
PY
```

### 8.3 dataset 兼容性检查

```bash
./.conda-py311-mps/bin/python - <<'PY'
from catlm.dataset import CatDataset

ds = CatDataset(
    path="data_cat_zh/train.jsonl",
    tokenizer_path="data_cat_zh/tokenizer.json",
    max_len=128,
)
print("可用样本数:", len(ds))
PY
```

只要这三步都过了，你就可以进入正式训练。
