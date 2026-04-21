# 第 5 章：从原始逻辑样本到完整训练数据集

这一章讲的是数据集真正落盘的过程。

核心文件是 [catlm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/catlm/generate_data.py) 和 [catlm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/catlm/prepare_data.py)。

## 1. 默认命令

在仓库根目录执行：

```bash
./.conda-py311-mps/bin/python -m catlm prepare
```

如果你已经 `conda activate` 了 `.conda-py311-mps`，也可以直接写：

```bash
python -m catlm prepare
```

## 2. 这条命令实际做了什么

[catlm/__main__.py](/Users/changzechuan/AIProjects/guppylm/catlm/__main__.py) 会调用 [catlm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/catlm/prepare_data.py) 的 `prepare()`。

`prepare()` 会顺序执行 3 件事：

1. 调用 [catlm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/catlm/generate_data.py) 生成数据
2. 回读 `train.jsonl` / `eval.jsonl` 里的 `text`
3. 训练并保存 `tokenizer.json`

这里有一个当前实现里非常重要的细节：

**CatLM 现在的数据生成不再只是“纯随机模板抽样”，而是“场景化生成 + 碰撞控制”两层一起工作。**

具体来说：

- 高碰撞主题已经大量改成了“场景化 input/output 成对生成”
- 生成总样本时还会对候选样本做筛选，尽量压低“同一个 input 对应太多不同 output”的情况
- 同时也会限制单个 `input` 被重复采样过多次，避免少数高频 prompt 过度主导整套数据
- 在切分 `train/eval` 时，也不再是简单随机按行切分，而是按 `input` 分组切分，保证同一个 `input` 不会同时出现在 `train` 和 `eval`

所以现在的生成逻辑，已经不是最早那种“只要随机抽到就直接落盘”的简单形式了。

## 3. CatLM 的默认输出文件

执行完成后，默认目录 [data_cat_zh](/Users/changzechuan/AIProjects/guppylm/data_cat_zh) 会得到：

- `samples_raw.jsonl`
- `train.jsonl`
- `eval.jsonl`
- `train_openai.jsonl`
- `eval_openai.jsonl`
- `tokenizer.json`

## 4. 每个文件分别代表什么

### 4.1 `samples_raw.jsonl`

这是逻辑样本源文件。

一行一个：

```json
{"input":"...","output":"...","category":"..."}
```

### 4.2 `train.jsonl`

这是本地训练主文件。

每行类似：

```json
{"text":"<|im_start|>user\n你饿了吗<|im_end|>\n<|im_start|>assistant\n饿。只要你碰一下冻干袋子，我就会立刻过去。<|im_end|>","category":"food"}
```

### 4.3 `eval.jsonl`

这是验证集主文件，格式和 `train.jsonl` 完全一致。

### 4.4 `train_openai.jsonl`

这是 OpenAI messages 格式导出版本。

### 4.5 `eval_openai.jsonl`

这是验证集的 OpenAI messages 格式导出版本。

### 4.6 `tokenizer.json`

这是基于 CatLM 自己语料训练出来的 tokenizer。

## 4.7 当前生成器额外做了什么

当前 [catlm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/catlm/generate_data.py) 里，除了基础的模板和词池之外，还实现了两套更强的控制逻辑。

### 4.7.1 场景化生成

很多高频主题现在不是简单地：

- 从一组 `user_msgs` 里随机选一句
- 再从一组 `assistant` 模板里随机选一句

而是会先选一个更具体的场景，再由这个场景同时决定：

- 用户输入怎么问
- 小猫应该怎么答

这样做的好处是：

- prompt 更具体
- output 更贴场景
- 同一个 input 不会轻易接上几十种无关回答

### 4.7.2 碰撞控制

当前生成器在落样本前，还会对候选样本做一层筛选，目标是尽量降低：

- 同一个 `input` 对应过多不同 `output`
- 同一个 `(input, output)` 对重复太多次

你可以把它理解成：

**“不是先生成什么就收什么，而是会优先保留更能提升数据有效多样性的样本。”**

这一步的目标不是为了让数据“完全不重复”，而是为了让：

- `distinct_inputs` 更高
- `unique_outputs` 更高
- `avg outputs per input` 更低

### 4.7.2 全局按 `input` 分组，并按 `category` 目标数切分训练集和验证集

当前 CatLM 的 `train/eval` 切分，已经不是最早那种：

- `random.shuffle(samples)`
- 然后直接按行数切成前 5% / 后 95%

现在的实现是：

- 先在全局范围内按 `input` 分组
- 再根据每个 `category` 目标验证样本数，贪心挑选一批 `input` 组进入 `eval`
- 剩余的 `input` 组全部进入 `train`

这意味着：

- 同一个 `input` 不会跨 `train/eval`
- 同时 `eval` 也不会只集中在极少数类别里
- 验证集不会再因为和训练集共享大量相同 prompt 而严重泄漏

这一步会让：

- `input_overlap = 0`

成为一个明确可检查的目标，同时也会让：

- `eval_categories`

成为新的检查项。

## 5. 默认样本规模

[catlm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/catlm/prepare_data.py) 默认参数是：

- `n_samples = 60000`
- `eval_ratio = 0.1`

所以默认情况下会得到：

- `train.jsonl`：54,000 行左右
- `eval.jsonl`：6,000 行左右

注意这里写“左右”，是因为当前切分单位已经不是单行样本，而是 `input` 分组。

也就是说，真实行数会尽量逼近 10%，但不保证永远严格等于 `54000 / 6000`。

## 6. 如何检查数据文件是否生成成功

### 6.1 看文件是否存在

```bash
ls -lh data_cat_zh
```

### 6.2 看行数

```bash
wc -l data_cat_zh/samples_raw.jsonl
wc -l data_cat_zh/train.jsonl
wc -l data_cat_zh/eval.jsonl
wc -l data_cat_zh/train_openai.jsonl
wc -l data_cat_zh/eval_openai.jsonl
```

### 6.3 随机抽几行看内容

```bash
sed -n '1,5p' data_cat_zh/samples_raw.jsonl
sed -n '1,5p' data_cat_zh/train.jsonl
sed -n '1,5p' data_cat_zh/train_openai.jsonl
```

### 6.4 检查当前这轮数据的碰撞情况

现在 CatLM 的数据准备，除了看文件有没有生成，还建议你直接看数据分布质量。

最重要的几个指标是：

- `distinct_inputs`
- `unique_outputs`
- `train_unique_texts`
- `text_overlap`
- `avg outputs per input`
- `input_overlap`
- `eval_categories`
- `top repeated inputs`

可以直接运行：

```bash
./.conda-py311-mps/bin/python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

rows = [json.loads(line) for line in Path("data_cat_zh/samples_raw.jsonl").open(encoding="utf-8") if line.strip()]
train = [json.loads(line) for line in Path("data_cat_zh/train.jsonl").open(encoding="utf-8") if line.strip()]
eval_ = [json.loads(line) for line in Path("data_cat_zh/eval.jsonl").open(encoding="utf-8") if line.strip()]
train_openai = [json.loads(line) for line in Path("data_cat_zh/train_openai.jsonl").open(encoding="utf-8") if line.strip()]
eval_openai = [json.loads(line) for line in Path("data_cat_zh/eval_openai.jsonl").open(encoding="utf-8") if line.strip()]

distinct_inputs = len({r["input"] for r in rows})
unique_outputs = len({r["output"] for r in rows})
train_unique_texts = len({r["text"] for r in train})
eval_unique_texts = len({r["text"] for r in eval_})

train_texts = {r["text"] for r in train}
eval_texts = {r["text"] for r in eval_}
train_inputs = {r["messages"][0]["content"] for r in train_openai}
eval_inputs = {r["messages"][0]["content"] for r in eval_openai}

by = defaultdict(set)
for r in rows:
    by[r["input"]].add(r["output"])

avg_outputs_per_input = sum(len(v) for v in by.values()) / max(1, len(by))

print("distinct_inputs =", distinct_inputs)
print("unique_outputs =", unique_outputs)
print("train_unique_texts =", train_unique_texts)
print("eval_unique_texts =", eval_unique_texts)
print("text_overlap =", len(train_texts & eval_texts))
print("avg_outputs_per_input =", round(avg_outputs_per_input, 2))
print("input_overlap =", len(train_inputs & eval_inputs))
print("eval_categories =", len({r["category"] for r in eval_}))
PY
```

## 7. 如果你想先做小规模烟雾测试

和 GuppyLM 一样，CatLM 默认命令没有暴露命令行参数覆盖 `n_samples`。

如果你想先小规模生成数据，可以用内联 Python：

```bash
./.conda-py311-mps/bin/python - <<'PY'
from catlm.prepare_data import prepare
prepare(data_dir="data_cat_zh_smoke", n_samples=220, eval_ratio=0.1)
PY
```

这样会生成一套小型烟雾数据集，不影响正式目录 `data_cat_zh/`。

## 8. 这一章结束时你的验收标准

继续往下之前，至少要确认：

1. `data_cat_zh/` 目录已生成
2. `train.jsonl` / `eval.jsonl` 可正常打开
3. `samples_raw.jsonl` 内容看起来像中文小猫，而不是别的角色
4. `tokenizer.json` 已落盘
5. `avg_outputs_per_input` 没有高得离谱
6. `input_overlap = 0`
7. `eval_categories` 覆盖正常，不是只剩很少几个主题

对于当前这套实现来说，第 5 条、第 6 条、第 7 条都很重要。

因为 CatLM 现在已经不只是关心“能不能生成数据”，还关心“这套数据是不是足够可学”。

如果这四条都满足，就可以进入 tokenizer 和测试集章节。
