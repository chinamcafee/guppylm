# GuppyLM 训练数据集准备指南

这份文档专门解释一件事：

**一个对当前 GuppyLM 项目“合法”的训练数据集，到底应该长什么样、由哪些文件组成、每个文件分别起什么作用、以及这些文件是如何一步一步准备出来的。**

本文档基于仓库当前已经准备好的数据集来讲解，也就是当前 `data/` 目录下的这 5 个文件：

- `data/train.jsonl`
- `data/eval.jsonl`
- `data/train_openai.jsonl`
- `data/eval_openai.jsonl`
- `data/tokenizer.json`

当前数据集的实际规模是：

- `train.jsonl`: 57,000 行
- `eval.jsonl`: 3,000 行
- `train_openai.jsonl`: 57,000 行
- `eval_openai.jsonl`: 3,000 行
- `tokenizer.json`: 一个基于当前数据训练出来的 BPE tokenizer

---

## 1. 先说结论：什么叫“合法”的训练数据集

对当前项目来说，真正会被本地训练代码直接读取的文件只有 3 个：

1. `data/train.jsonl`
2. `data/eval.jsonl`
3. `data/tokenizer.json`

也就是说，**只要这 3 个文件格式正确，当前项目就能训练。**

另外两个文件：

- `data/train_openai.jsonl`
- `data/eval_openai.jsonl`

不是本地训练所必需的，它们更像是“辅助导出格式”，方便你把同一份数据集拿去别的生态使用。

所以，最小合法集合是：

- `train.jsonl`
- `eval.jsonl`
- `tokenizer.json`

完整推荐集合是：

- `train.jsonl`
- `eval.jsonl`
- `train_openai.jsonl`
- `eval_openai.jsonl`
- `tokenizer.json`

---

## 2. 当前项目真正读取训练数据的代码在哪里

当前本地训练入口在 [guppylm/train.py](/Users/changzechuan/AIProjects/guppylm/guppylm/train.py)。

训练时它读取的是：

- [train.py](/Users/changzechuan/AIProjects/guppylm/guppylm/train.py#L60) 的 `data/train.jsonl`
- [train.py](/Users/changzechuan/AIProjects/guppylm/guppylm/train.py#L64) 的 `data/eval.jsonl`
- [train.py](/Users/changzechuan/AIProjects/guppylm/guppylm/train.py#L56) 的 `data/tokenizer.json`

数据加载逻辑在 [guppylm/dataset.py](/Users/changzechuan/AIProjects/guppylm/guppylm/dataset.py)：

- 它逐行读取 JSONL
- 对每一行取出 `data["text"]`
- 用 `tokenizer.json` 编码
- 生成语言模型训练所需的 `(x, y)` 序列

这意味着一个最核心的事实：

**对当前本地训练代码而言，`train.jsonl` / `eval.jsonl` 中最重要的字段是 `text`。**

`category` 字段不是训练必须字段，但当前项目会把它保留下来，便于统计、分析和后续数据管理。

---

## 3. 当前数据集的 5 个文件分别是干什么的

这一节逐个解释。

### 3.1 `data/train.jsonl`

这是**训练集主文件**。

本地训练时，大多数 step 都是从这里读样本。

当前文件中的每一行长这样：

```json
{"text": "<|im_start|>user\nwhat do you remember<|im_end|>\n<|im_start|>assistant\nmy brain is small but it has priorities. food. safety. water. you.<|im_end|>", "category": "memory"}
```

这里有两个字段：

- `text`: 真正喂给模型训练的完整文本
- `category`: 该样本属于哪个主题类别，例如 `memory`

对训练最关键的是 `text`，因为 [guppylm/dataset.py](/Users/changzechuan/AIProjects/guppylm/guppylm/dataset.py#L18) 只会取这个字段做编码。

### 3.2 `data/eval.jsonl`

这是**验证集主文件**。

它的格式和 `train.jsonl` 完全相同，只是用途不同：

- `train.jsonl` 用来更新模型参数
- `eval.jsonl` 用来评估当前模型效果，不参与梯度更新

当前文件中的一行示例：

```json
{"text": "<|im_start|>user\ndo you like music<|im_end|>\n<|im_start|>assistant\ni can feel it in the water. little vibrations.<|im_end|>", "category": "music"}
```

### 3.3 `data/train_openai.jsonl`

这是**同一份训练集的 OpenAI messages 格式导出版本**。

当前文件中的一行示例：

```json
{"messages": [{"role": "user", "content": "what do you remember"}, {"role": "assistant", "content": "my brain is small but it has priorities. food. safety. water. you."}]}
```

这个文件不是当前本地训练代码必需的。它的作用主要是：

- 方便对接别的工具链
- 方便导出到别的训练/微调格式
- 让数据以更直观的对话结构存在

### 3.4 `data/eval_openai.jsonl`

这是**验证集的 OpenAI messages 格式导出版本**。

作用和 `train_openai.jsonl` 一样，只不过对应的是验证集。

### 3.5 `data/tokenizer.json`

这是**tokenizer 文件**，也是当前训练和推理都必须存在的文件。

它决定了：

- 文本如何被拆成 token
- 特殊 token 的 ID 是什么
- 模型看到的词表是什么

当前 tokenizer 的几个关键事实：

- 类型：BPE
- 特殊 token：
  - `<pad>` 对应 id 0
  - `<|im_start|>` 对应 id 1
  - `<|im_end|>` 对应 id 2

它在训练和推理里都被使用：

- 训练读取 [guppylm/dataset.py](/Users/changzechuan/AIProjects/guppylm/guppylm/dataset.py#L12)
- 推理读取 [guppylm/inference.py](/Users/changzechuan/AIProjects/guppylm/guppylm/inference.py#L17)

---

## 4. 本项目数据集准备的真实链路是什么

当前项目的数据集不是先手工写出 `train.jsonl`，而是先从“逻辑样本”开始，再逐步导出为多个文件。

完整链路是：

1. 先定义一个逻辑样本：`input` / `output` / `category`
2. 再把逻辑样本格式化成 `train.jsonl` / `eval.jsonl` 用的 `text`
3. 再把同一份逻辑样本导出成 `train_openai.jsonl` / `eval_openai.jsonl`
4. 最后基于 `train.jsonl` 和 `eval.jsonl` 的文本重新训练 `tokenizer.json`

所以，你可以把整个数据准备过程理解成：

**“一份逻辑样本，派生出多种落地文件。”**

---

## 5. Step 1：先定义最原始的逻辑样本

当前仓库里，最原始的逻辑样本来自 [guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py)。

在那里，每条逻辑样本本质上长这样：

```python
{
    "input": "...",
    "output": "...",
    "category": "..."
}
```

例如一条逻辑样本可以理解为：

```json
{
  "input": "what do you remember",
  "output": "my brain is small but it has priorities. food. safety. water. you.",
  "category": "memory"
}
```

这一步非常重要，因为它定义的是“内容本身”，而不是训练格式。

也就是说，这一层回答的是：

- 用户说了什么
- Guppy 回什么
- 这条样本属于什么主题

**这是最适合人工编辑和扩充的层。**

如果你以后要加中文数据，最推荐修改的也是这一层，而不是直接乱改最终的 `train.jsonl`。

---

## 6. Step 2：把逻辑样本转换为训练主格式

当前项目通过 [guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py#L1625) 的 `format_sample()` 把逻辑样本转成训练文本。

转换规则是：

```python
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

例如：

原始逻辑样本：

```json
{
  "input": "what do you remember",
  "output": "my brain is small but it has priorities. food. safety. water. you.",
  "category": "memory"
}
```

被转换后就变成：

```text
<|im_start|>user
what do you remember<|im_end|>
<|im_start|>assistant
my brain is small but it has priorities. food. safety. water. you.<|im_end|>
```

然后它被写入 `train.jsonl` 或 `eval.jsonl` 中的 `text` 字段：

```json
{"text": "...上面的完整字符串...", "category": "memory"}
```

这就是当前本地训练真正吃进去的格式。

---

## 7. Step 3：把同一份逻辑样本转换为 OpenAI messages 格式

当前项目通过 [guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py#L1632) 的 `to_openai()` 做这件事。

转换后长这样：

```json
{
  "messages": [
    {"role": "user", "content": "what do you remember"},
    {"role": "assistant", "content": "my brain is small but it has priorities. food. safety. water. you."}
  ]
}
```

这一步生成的就是：

- `train_openai.jsonl`
- `eval_openai.jsonl`

这两个文件目前不是本地训练必须，但它们保留了比较通用的对话表示形式。

---

## 8. Step 4：切分训练集和验证集

当前项目的数据切分逻辑也在 [guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py#L1643) 里。

过程是：

1. 先生成总样本列表
2. `random.shuffle(samples)`
3. 按 `eval_ratio` 切分

当前默认是：

- 总样本数：60,000
- 验证比例：0.05

所以最终得到：

- 57,000 条训练样本
- 3,000 条验证样本

这也正是当前 `data/` 目录里的实际行数。

---

## 9. Step 5：把训练文本喂给 tokenizer 训练词表

当前 tokenizer 的训练逻辑在 [guppylm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/prepare_data.py)。

这里有两个关键信息：

### 9.1 tokenizer 不是手工写的，是训练出来的

`tokenizer.json` 不是固定模板文件，而是从文本语料中学出来的。

### 9.2 当前项目是基于 `train.jsonl` 和 `eval.jsonl` 里的 `text` 重新训练 tokenizer

也就是说，当前代码的顺序是：

1. 先有 `train.jsonl` / `eval.jsonl`
2. 再读出其中的 `text`
3. 用这些文本训练 BPE tokenizer
4. 保存为 `data/tokenizer.json`

所以有一个非常重要的规则：

**只要你的训练文本内容发生明显变化，你就应该重训 tokenizer。**

尤其是下面几种情况：

- 新增了中文
- 新增了大量新主题
- 修改了特殊 token 格式
- 改变了整体语料风格

如果你改了 `train.jsonl` / `eval.jsonl`，但不重训 `tokenizer.json`，训练结果通常会变差，严重时会让新语言的数据几乎学不好。

---

## 10. 现在从头解释：如何手把手准备当前这 5 个文件

这一节按照“从 0 到现有数据”的顺序来讲。

### 10.1 准备逻辑样本

先准备一批如下结构的样本：

```json
{
  "input": "用户输入",
  "output": "助手输出",
  "category": "主题类别"
}
```

你可以把它们存放在：

- Python 模板生成器里
- 一个中间 JSONL 文件里
- 一个 Excel / CSV 再转 JSONL 的流程里

当前项目选择的是：

- 在 [guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py) 里用模板函数生成

### 10.2 保证逻辑样本内容符合 GuppyLM 人设

当前项目的数据不是任意对话数据，而是“Guppy 这条鱼”的角色数据。因此逻辑样本要满足这些风格约束：

- 回答短
- 语气简单
- 以鱼缸、水、温度、食物、光线为中心
- 不擅长抽象人类概念
- 单轮对话为主

这一步不是格式要求，而是“数据质量要求”。

格式合法不等于训练效果好。角色一致性是这个项目里非常重要的一部分。

### 10.3 把逻辑样本写成 `train.jsonl` / `eval.jsonl`

准备方法是：

1. 把逻辑样本打乱
2. 切成训练集和验证集
3. 用 `format_sample()` 包上特殊 token
4. 每条样本写成一行 JSON

每行应该类似：

```json
{"text": "<|im_start|>user\nhi guppy<|im_end|>\n<|im_start|>assistant\nhello. the water is nice today.<|im_end|>", "category": "greeting"}
```

注意这几个点：

- 必须是 **JSONL**
- 必须是 **一行一个 JSON 对象**
- `text` 应该是完整对话文本，而不是裸 `input` / `output`
- 特殊 token 应与 tokenizer 定义一致

### 10.4 把同一批逻辑样本再写成 `train_openai.jsonl` / `eval_openai.jsonl`

每行应该类似：

```json
{"messages": [{"role": "user", "content": "hi guppy"}, {"role": "assistant", "content": "hello. the water is nice today."}]}
```

这一步不是本地训练必须，但推荐保留，因为：

- 更容易和别的工具联动
- 更容易人工检查
- 将来做别的导出时更方便

### 10.5 基于 `train.jsonl` 和 `eval.jsonl` 训练 `tokenizer.json`

当前项目做法是：

1. 读取 `train.jsonl` 的每一行 `text`
2. 读取 `eval.jsonl` 的每一行 `text`
3. 用这些文本训练 BPE tokenizer
4. 加入特殊 token：
   - `<pad>`
   - `<|im_start|>`
   - `<|im_end|>`
5. 保存到 `data/tokenizer.json`

最终才得到一套完整可训练的数据集。

---

## 11. 对当前项目来说，哪些字段是必须的，哪些是推荐的

### 11.1 对 `train.jsonl` / `eval.jsonl`

必须的：

- `text`

推荐保留：

- `category`

### 11.2 对 `train_openai.jsonl` / `eval_openai.jsonl`

必须的：

- `messages`

其中每个 message 必须至少有：

- `role`
- `content`

### 11.3 对 `tokenizer.json`

必须满足：

- 能被 `tokenizers.Tokenizer.from_file(...)` 正常读取
- 与 `text` 中的特殊 token 兼容
- 与模型训练时使用的数据文本是同一套语料分布下训练出来的

---

## 12. 一个“合法样本”至少应满足哪些条件

这一节非常重要。

对当前项目而言，一条放进 `train.jsonl` 或 `eval.jsonl` 的样本，最低要求是：

1. 是有效 JSON
2. 有 `text` 字段
3. `text` 是字符串
4. `text` 中包含合理的 user / assistant 结构
5. tokenizer 编码后长度至少大于等于 2

第 5 条来自 [guppylm/dataset.py](/Users/changzechuan/AIProjects/guppylm/guppylm/dataset.py#L20) 和 [guppylm/dataset.py](/Users/changzechuan/AIProjects/guppylm/guppylm/dataset.py#L22)：

- 太长的样本会被截断
- 太短的样本会被丢弃

所以如果你准备的数据非常脏，例如空字符串、极短字符串、格式错乱字符串，它虽然可能“看起来写进文件了”，但训练时可能会被跳过或造成质量问题。

---

## 13. 当前数据集为什么是单轮，而不是多轮

当前项目的数据格式虽然长得像聊天格式，但实际是单轮：

- 一条 user
- 一条 assistant

没有长上下文历史。

这是因为当前模型很小，最大上下文也只有 128 token。多轮对话很容易把上下文挤爆，导致质量下降。

所以，如果你以后继续扩数据，推荐仍然以**单轮对话**为主，除非你也同步提高模型和上下文长度。

---

## 14. 如果我想新增中文对话数据训练，应该如何修改当前数据集

下面开始回答你的第一个独立问题。

### 14.1 先说核心原则

如果你是想在**当前英文鱼人格数据集基础上混入中文对话数据**，推荐做法不是只改一个文件，而是要把整条数据链一起更新。

也就是说，你至少要同步更新：

1. 逻辑样本内容
2. `train.jsonl`
3. `eval.jsonl`
4. `train_openai.jsonl`
5. `eval_openai.jsonl`
6. `tokenizer.json`

**最不能做的事情**是：

- 只往 `train.jsonl` 里手工追加中文
- 却继续使用旧的英文 tokenizer

这样做 technically 能跑，但效果通常会很差。

### 14.2 推荐修改的是“逻辑样本层”，不是最终文件层

最推荐的做法是：

1. 在 [guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py) 里新增中文模板
2. 生成新的逻辑样本
3. 重新导出全部下游文件

你可以新增两种方式：

#### 方案 A：给现有主题增加中文变体

例如把 `greeting` 主题同时支持：

- 英文 user / 英文 assistant
- 中文 user / 中文 assistant

#### 方案 B：新增独立中文主题类别

例如新增：

- `greeting_zh`
- `food_zh`
- `light_zh`
- `tank_zh`

我更推荐方案 B，原因是：

- 更容易统计中文数据比例
- 更容易检查中文数据质量
- 将来要调整中英文配比时更方便

### 14.3 中文混入后的 `train.jsonl` 应该长什么样

格式不用变，仍然是：

```json
{"text": "<|im_start|>user\n你饿吗<|im_end|>\n<|im_start|>assistant\n我一直都很饿。我现在就去水面找吃的。<|im_end|>", "category": "food_zh"}
```

重点是：

- 外层格式完全不变
- 仍然是一条 user、一条 assistant
- 只是内容从英文变成了中文

### 14.4 中文混入后的 `train_openai.jsonl` 应该长什么样

```json
{"messages": [{"role": "user", "content": "你饿吗"}, {"role": "assistant", "content": "我一直都很饿。我现在就去水面找吃的。"}]}
```

### 14.5 中文混入后，tokenizer 应该怎么处理

必须重训。

当前 tokenizer 虽然是 ByteLevel BPE，理论上可以处理任何 UTF-8 文本，但：

- 旧 tokenizer 是按当前英文鱼缸语料分布训练出来的
- 新增中文后，token 统计分布会明显变化
- 不重训 tokenizer，中文会被切得很碎，学习效率会下降

因此推荐顺序是：

1. 先准备“英文 + 中文”的新 `train.jsonl` / `eval.jsonl`
2. 再基于混合语料重新训练 `tokenizer.json`
3. 然后再开始训练模型

### 14.6 中文混入后，是否需要增加词表大小

当前项目 `VOCAB_SIZE` 目标是 4096。

如果你只是少量加入中文，4096 也能工作。

但如果你准备明显增加中英文混合语料，推荐你重新评估是否提高词表大小，例如：

- 4096 继续用
- 或者提升到 6000 / 8192

不是说必须改，而是：

- 中英混合通常会带来更复杂的分词需求
- 更大的词表有时会更稳

### 14.7 中文混入后的内容风格应该怎么控制

如果你只是增加中文，但仍然希望模型保留“Guppy 是一条鱼”的人格，那么中文样本内容要保持和英文样本同样的世界观：

- 说话短
- 语义简单
- 围绕水、光、食物、温度、鱼缸
- 对抽象问题保持“鱼式理解”

例如：

不推荐：

```text
用户：你如何看待宏观经济周期？
助手：从宏观经济学角度看……
```

推荐：

```text
用户：你怎么看钱？
助手：我不知道钱是什么。它能吃吗。
```

---

## 15. 如果我只希望进行纯中文对话数据的训练，数据集应该如何从 0 到 1 地准备

下面开始回答你的第二个独立问题。

这个问题和“在现有英文数据集里混入中文”不一样。

这里的目标是：

**完全不依赖当前英文训练集，单独从 0 到 1 构建一套纯中文训练数据集。**

### 15.1 第一步：先定义你的中文角色设定

先不要急着写 `train.jsonl`。

第一步应该先明确：

- 你的助手是谁
- 说话风格是什么
- 能理解什么
- 不能理解什么
- 回答长度如何控制
- 是单轮还是多轮

如果你仍然想训练“中文 Guppy”，建议先写一页角色说明，至少包括：

- 它是一条小鱼
- 说话短
- 语气天真
- 关注鱼缸环境
- 经常提到食物、水、光、砂石、植物
- 遇到抽象概念时会按鱼的方式理解

### 15.2 第二步：先设计主题清单

推荐先列主题，再写具体样本。

例如中文版可以先列：

- 问候
- 饥饿
- 水温
- 光线
- 水质
- 鱼缸环境
- 害怕
- 睡觉
- 雨天
- 猫
- 主人
- 回忆
- 名字
- 意义
- 未来

这些主题就是未来 `category` 的来源。

### 15.3 第三步：先写逻辑样本，而不是直接写最终训练文本

建议你先收集成这种结构：

```json
{
  "input": "你今天开心吗",
  "output": "开心。水很安静，光也刚刚好。",
  "category": "feeling_zh"
}
```

这样做的好处是：

- 容易人工编写
- 容易批量检查
- 容易后续导出多种格式
- 容易控制主题平衡

### 15.4 第四步：保证中文样本风格一致

纯中文训练里，最重要的不是格式，而是风格一致性。

你需要保证：

- 语气统一
- 用词统一
- 角色稳定
- 避免一会儿像鱼，一会儿像百科全书

例如同一个问题：

```text
用户：你害怕吗？
```

你的不同回答应该是同一人格的不同变体，而不是互相打架：

推荐：

- “会。有时候大的影子靠近，我会躲一下。”
- “会一点。我不喜欢突然的震动。”
- “会。我会先躲到植物后面看看。”

不推荐：

- “恐惧是一种由杏仁核主导的高级情绪反应。”

### 15.5 第五步：切分训练集与验证集

准备好足够多的逻辑样本后，先打乱，再切分。

建议起步比例：

- 95% 训练
- 5% 验证

例如你有 20,000 条中文样本：

- 19,000 条训练
- 1,000 条验证

### 15.6 第六步：把中文逻辑样本转换成 `train.jsonl` / `eval.jsonl`

仍然使用当前项目兼容的格式：

```json
{"text": "<|im_start|>user\n你今天饿吗<|im_end|>\n<|im_start|>assistant\n我一直都很饿。小鱼总会想着吃的。<|im_end|>", "category": "food_zh"}
```

这是最重要的训练文件。

### 15.7 第七步：可选地导出 `train_openai.jsonl` / `eval_openai.jsonl`

对应样本写成：

```json
{"messages": [{"role": "user", "content": "你今天饿吗"}, {"role": "assistant", "content": "我一直都很饿。小鱼总会想着吃的。"}]}
```

如果你只是本地训练，其实不是必须。

但我仍然建议保留，因为这会让你的数据资产更通用。

### 15.8 第八步：基于中文训练文本重新训练 tokenizer

这一步对纯中文尤其重要。

你不能直接复用当前英文 tokenizer，因为：

- 语料变了
- 语言变了
- 字符分布变了
- 常见词片段完全不一样

所以纯中文数据集必须重新训练一个新的 `tokenizer.json`。

### 15.9 第九步：检查样本长度和上下文长度

当前模型的 `max_seq_len` 是 128。

虽然中文每个字的信息密度高，但你仍然要注意：

- 不要把单条样本写得过长
- 保持单轮结构简洁
- 尽量把有效信息放在短回复里

因为 [guppylm/dataset.py](/Users/changzechuan/AIProjects/guppylm/guppylm/dataset.py#L20) 会直接截断超长样本。

### 15.10 第十步：最后再训练模型

当你已经准备好：

- `data/train.jsonl`
- `data/eval.jsonl`
- `data/tokenizer.json`

之后，训练阶段和当前项目现有流程没有区别：

```bash
python -m guppylm train
```

---

## 16. 两种中文方案的区别，总结一下

### 16.1 方案一：在当前数据集基础上新增中文

适合你想做的事情：

- 保留英文能力
- 增加中文能力
- 做中英文混合对话训练

你应该做的事：

1. 给逻辑样本加中文数据
2. 重新导出 train/eval
3. 重新导出 openai 版本
4. 重新训练 tokenizer
5. 重新训练模型

### 16.2 方案二：从零准备纯中文数据集

适合你想做的事情：

- 完全转成中文模型
- 不再依赖当前英文数据
- 从头定义新的中文角色分布

你应该做的事：

1. 先定义角色
2. 先列主题
3. 写逻辑样本
4. 切分 train/eval
5. 生成 `train.jsonl` / `eval.jsonl`
6. 训练新的 `tokenizer.json`
7. 训练模型

---

## 17. 最后给你一个最实用的判断标准

以后你在改数据集时，可以用下面这句话检查自己有没有走偏：

**只要我准备出的 `train.jsonl` / `eval.jsonl` 能被当前代码读取，`text` 格式正确，`tokenizer.json` 是基于同一套语料重新训练出来的，那么这套数据集对当前项目就是合法的。**

如果你进一步还想要：

- 数据更容易管理
- 更容易扩语言
- 更容易导出到别的平台

那就继续保留：

- `category`
- `train_openai.jsonl`
- `eval_openai.jsonl`

这样你就拥有的是一套“当前项目能训练，也方便以后扩展”的完整数据资产。
