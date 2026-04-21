# 第 3 章：CatLM 代码结构与 GuppyLM 的对标关系

这一章只讲代码结构，不讲数据内容。

## 1. CatLM 目录结构

当前新增的代码目录是：

```text
catlm/
├── __init__.py
├── __main__.py
├── config.py
├── dataset.py
├── eval_cases.py
├── generate_data.py
├── inference.py
├── model.py
├── mps.py
├── prepare_data.py
└── train.py
```

## 2. 每个文件的职责

### 2.1 入口

[catlm/__main__.py](/Users/changzechuan/AIProjects/guppylm/catlm/__main__.py) 提供了 3 个入口：

- `python -m catlm prepare`
- `python -m catlm train`
- `python -m catlm chat`

### 2.2 配置

[catlm/config.py](/Users/changzechuan/AIProjects/guppylm/catlm/config.py) 定义：

- 模型超参 `CatConfig`
- 训练超参 `TrainConfig`

这里最重要的变化是默认路径已经切到：

- `data_cat_zh`
- `checkpoints_catlm`

### 2.3 数据生成

[catlm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/catlm/generate_data.py) 负责：

- 定义中文小猫语料的世界观
- 生成逻辑样本
- 导出训练主格式
- 导出 OpenAI messages 格式

### 2.4 数据准备与 tokenizer

[catlm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/catlm/prepare_data.py) 负责：

1. 调用 `generate_dataset(...)`
2. 回读 `train.jsonl` / `eval.jsonl`
3. 训练 `tokenizer.json`

### 2.5 数据加载

[catlm/dataset.py](/Users/changzechuan/AIProjects/guppylm/catlm/dataset.py) 负责：

- 逐行读 JSONL
- 编码 `text`
- 截断到 `max_len`
- 生成 `(x, y)` 训练对

### 2.6 模型

[catlm/model.py](/Users/changzechuan/AIProjects/guppylm/catlm/model.py) 仍然是和 GuppyLM 对标的 vanilla transformer：

- 多头注意力
- ReLU FFN
- LayerNorm
- learned positional embeddings

### 2.7 训练

[catlm/train.py](/Users/changzechuan/AIProjects/guppylm/catlm/train.py) 负责：

- 自动选择 `cuda` / `mps` / `cpu`
- 构建 dataloader
- 跑训练循环
- 定期 eval
- 保存 checkpoint

### 2.8 推理

[catlm/inference.py](/Users/changzechuan/AIProjects/guppylm/catlm/inference.py) 负责：

- 加载 tokenizer
- 加载 checkpoint
- 按 `<|im_start|>...<|im_end|>` 格式构 prompt
- 返回单轮对话结果

### 2.9 手工测试样例

[catlm/eval_cases.py](/Users/changzechuan/AIProjects/guppylm/catlm/eval_cases.py) 维护的是：

- 一组独立于训练集的人工测试提示
- 每条提示希望看到的关键词
- 风格预期

它不是训练必需文件，但对验收很有用。

## 3. 和 GuppyLM 完全一致的核心逻辑

CatLM 虽然换了人设，但这些逻辑没有变：

### 3.1 数据主格式没变

训练主格式仍然是：

```text
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

### 3.2 OpenAI 导出格式没变

仍然是：

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### 3.3 tokenizer 特殊 token 没变

[catlm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/catlm/prepare_data.py) 里仍然保留：

- `<pad>`
- `<|im_start|>`
- `<|im_end|>`

### 3.4 模型规模也仍然对标 GuppyLM

默认超参仍然是：

- `vocab_size = 4096`
- `max_seq_len = 128`
- `d_model = 448`
- `n_layers = 7`
- `n_heads = 7`
- `ffn_hidden = 896`

这意味着 CatLM 仍然是一套小模型训练链路，不是去追求大模型能力。

## 4. CatLM 在结构上做出的两处关键改动

### 4.1 默认目录改成独立路径

这是为了保证不影响原项目已有的小鱼数据和模型。

### 4.2 数据生成器换成中文小猫内容域

也就是说：

- 训练格式没变
- 训练代码没变
- 模型结构没变
- 真正换掉的是训练语料的内容世界

这一点是 CatLM 成立的核心。
