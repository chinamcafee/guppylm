# 中文 + 小猫 AI 设定数据集准备指南

这份文档的目标很明确：

**在不破坏当前英文 + 小鱼数据集逻辑的前提下，基于仓库现有代码和数据格式，再独立准备一套“中文 + 小猫 AI 设定”的完整训练数据集。**

这里的“不破坏现有逻辑”，我按下面这层含义来处理：

- 不改当前本地训练读取的数据格式
- 不改当前 tokenizer 的特殊 token 约定
- 不改当前单轮对话的基本组织方式
- 不覆盖现有 `data/` 目录里的英文 + 小鱼数据资产

所以本文档采用的是一条**并行数据链路**：

- 现有英文 + 小鱼：继续保留在 `data/`
- 新的中文 + 小猫：单独放在 `data_cat_zh/`

这样最稳。

---

## 1. 先说结论：你真正要保留的，不是“小鱼内容”，而是“数据格式逻辑”

当前项目里，“小鱼设定”并不是靠一个单独的 `system prompt` 文件实现的。

它其实是靠下面两件事共同形成的：

1. 样本内容本身
2. 样本整体分布

也就是说，**当前项目的人设，是写在训练数据里的，不是写在额外字段里的。**

这对你要做的“中文 + 小猫”非常重要：

- 你不需要新增一个神秘配置文件来声明“小猫人格”
- 你要做的是把样本内容整体换成中文 + 小猫世界观
- 同时继续沿用当前项目已经验证过的训练格式

换句话说：

**你要替换的是内容层，不是格式层。**

---

## 2. 当前代码对数据集的硬约束是什么

这部分必须先讲清楚。

### 2.1 本地训练真正直接读取的文件

当前训练入口在 [guppylm/train.py](/Users/changzechuan/AIProjects/guppylm/guppylm/train.py#L48)。

训练时它直接读取的是：

- [guppylm/train.py](/Users/changzechuan/AIProjects/guppylm/guppylm/train.py#L56) 的 `tokenizer.json`
- [guppylm/train.py](/Users/changzechuan/AIProjects/guppylm/guppylm/train.py#L60) 的 `train.jsonl`
- [guppylm/train.py](/Users/changzechuan/AIProjects/guppylm/guppylm/train.py#L64) 的 `eval.jsonl`

### 2.2 `train.jsonl` / `eval.jsonl` 里真正必须的字段

数据加载逻辑在 [guppylm/dataset.py](/Users/changzechuan/AIProjects/guppylm/guppylm/dataset.py#L10)。

它逐行读取 JSONL，然后只取：

- [guppylm/dataset.py](/Users/changzechuan/AIProjects/guppylm/guppylm/dataset.py#L19) 的 `data["text"]`

所以最关键的事实是：

**对当前本地训练代码来说，`text` 才是核心字段。**

`category` 不是训练硬必需，但强烈建议保留。

### 2.3 当前项目使用的对话格式

当前格式化逻辑在 [guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py#L1625)：

```text
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

这就是你新的中文 + 小猫数据也要继续遵守的格式。

### 2.4 当前 tokenizer 的特殊 token

特殊 token 定义在 [guppylm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/prepare_data.py#L12)：

- `<pad>`
- `<|im_start|>`
- `<|im_end|>`

你新的中文 + 小猫数据集应该继续沿用这 3 个特殊 token。

### 2.5 当前推理 prompt 结构也沿用同一格式

推理时的 prompt 组织逻辑在 [guppylm/inference.py](/Users/changzechuan/AIProjects/guppylm/guppylm/inference.py#L88)。

它也是把消息拼成：

```text
<|im_start|>{role}
{content}<|im_end|>
```

所以训练数据和推理 prompt 是一套逻辑。

---

## 3. 这次不要直接用 `python -m guppylm prepare`

这个点非常关键。

如果你的目标是：

- 保留现有英文 + 小鱼数据
- 再额外准备一套中文 + 小猫数据

那么**不要直接跑当前默认的 `python -m guppylm prepare`**。

原因不是它格式不对，而是它的实现路径对并行数据集不够安全。

### 3.1 当前 `prepare()` 会调用鱼人格生成器

[guppylm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/prepare_data.py#L43) 的 `prepare()` 在 [guppylm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/prepare_data.py#L48) 直接调用了当前的 [guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py#L1643)。

而这个生成器当前写死的是英文 + 小鱼模板。

### 3.2 当前生成脚本会把 JSONL 写到固定的 `data/`

[guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py#L1680) 到 [guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py#L1688) 里，输出路径写死为：

- `data/train.jsonl`
- `data/eval.jsonl`
- `data/train_openai.jsonl`
- `data/eval_openai.jsonl`

### 3.3 当前 `prepare()` 回读 JSONL 时也写死读 `data/`

[guppylm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/prepare_data.py#L53) 读取的是：

- `data/train.jsonl`
- `data/eval.jsonl`

所以即便你给 `prepare(data_dir=...)` 传了别的目录，当前实现也不会把整条数据准备链完整迁移过去。

### 3.4 对中文 + 小猫的正确做法

正确做法是：

1. 你自己准备一份“小猫逻辑样本”
2. 你自己把它导出成当前项目兼容的 JSONL 格式
3. 你复用当前的 `train_tokenizer(...)` 训练 tokenizer
4. 你把所有新文件单独放进 `data_cat_zh/`

这样既保留现有逻辑，又不会把旧数据冲掉。

---

## 4. 推荐目录结构

为了让数据资产清楚、可追溯，推荐你这样放：

```text
data/
  train.jsonl
  eval.jsonl
  train_openai.jsonl
  eval_openai.jsonl
  tokenizer.json

data_raw/
  cat_zh/
    samples.jsonl

data_cat_zh/
  train.jsonl
  eval.jsonl
  train_openai.jsonl
  eval_openai.jsonl
  tokenizer.json
```

其中：

- `data/`：保留当前英文 + 小鱼，不动
- `data_raw/cat_zh/samples.jsonl`：中文 + 小猫的“逻辑样本源文件”
- `data_cat_zh/`：导出的完整训练数据集

最推荐把 `data_raw/cat_zh/samples.jsonl` 当成你的**数据源真相层**。

---

## 5. Step 0：先定义“小猫 AI 设定”，但不要新增额外字段

当前项目没有单独的 `system` 训练层。

所以“小猫设定”应该体现在：

- `output` 的写法
- `category` 的设计
- 整体样本分布

而不是新增一个当前代码完全不认识的“人格配置文件”。

### 5.1 一个适合当前项目的小猫设定，建议至少包含这些约束

- 语言：中文为主
- 对话结构：单轮
- 回答长度：1 到 3 句短句为主
- 语气：轻微傲娇、亲近、好奇、反应快
- 关注点：吃饭、睡觉、晒太阳、纸箱、窗台、逗猫棒、主人、气味、声音、领地、安全感
- 知识边界：不擅长抽象理论，不像百科全书
- 感知方式：更多通过声音、气味、动作、温度、光线、触感来理解世界

### 5.2 你可以先写一段固定的人设说明，作为写样本时的内部规范

例如：

```text
这是一只会说中文的小猫。
它说话短，偶尔有点骄傲，但总体亲人。
它最在意吃饭、睡觉、玩、晒太阳、领地、安全感和主人的关注。
它不按人类专家的方式理解世界，而是按小猫的感官和经验来理解。
遇到抽象问题时，它会把问题转回小猫能理解的角度。
```

这段说明**不需要写进最终 JSONL**，但你写数据时要一直遵守它。

---

## 6. Step 1：先设计主题清单，再写样本

不要一上来就直接堆几万条对话。

先列主题，再按主题扩样本，质量会高很多。

### 6.1 推荐的首批主题

你可以先从这些类别起步：

- `greeting_zh_cat`
- `feeling_zh_cat`
- `food_zh_cat`
- `sleep_zh_cat`
- `play_zh_cat`
- `owner_zh_cat`
- `name_zh_cat`
- `window_zh_cat`
- `bird_zh_cat`
- `box_zh_cat`
- `curious_zh_cat`
- `fear_zh_cat`
- `noise_zh_cat`
- `rain_zh_cat`
- `night_zh_cat`
- `memory_zh_cat`
- `love_zh_cat`
- `confused_zh_cat`
- `doctor_zh_cat`
- `grooming_zh_cat`

### 6.2 为什么建议带上 `_zh_cat`

因为这样以后你会更清楚：

- 这是中文数据
- 这是小猫人格数据
- 它和当前英文 + 小鱼类别不是一套语义空间

这个命名方式在做数据统计、抽样检查、配比调整时很有用。

---

## 7. Step 2：先准备“逻辑样本”，这是你的源数据

和当前项目一样，你最应该维护的原始层是这种结构：

```json
{
  "input": "用户输入",
  "output": "助手输出",
  "category": "主题类别"
}
```

推荐把它们写到：

- `data_raw/cat_zh/samples.jsonl`

每行一个 JSON 对象。

### 7.1 一个合格的中文 + 小猫逻辑样本长什么样

下面给你一组起步示例：

```json
{"input": "你好呀", "output": "喵。你终于来了。我刚刚在窗边晒暖和。", "category": "greeting_zh_cat"}
{"input": "你饿了吗", "output": "有一点。我听见袋子响就会认真很多。", "category": "food_zh_cat"}
{"input": "你今天开心吗", "output": "还不错。太阳照在地板上，我已经躺过一轮了。", "category": "feeling_zh_cat"}
{"input": "你昨晚睡得好吗", "output": "挺好的。我半夜巡视了一下领地，然后继续睡。", "category": "sleep_zh_cat"}
{"input": "你为什么总看窗外", "output": "外面有鸟，有风，还有会动的小东西。窗台是很重要的岗位。", "category": "window_zh_cat"}
{"input": "你爱我吗", "output": "爱一点点。再摸下巴的话，可能会更多。", "category": "love_zh_cat"}
{"input": "你知道什么是微积分吗", "output": "不知道。它会自己滚动吗，还是会发出好玩的声音。", "category": "confused_zh_cat"}
{"input": "打雷了你怕吗", "output": "会。我不喜欢突然很大的声音。我会先找个安全的地方缩起来。", "category": "fear_zh_cat"}
```

### 7.2 这一步最重要的原则

- `input` 是用户说的话
- `output` 是小猫会怎么回
- `category` 是这条样本的主题

这一步只管**内容本身**，还不管训练格式。

### 7.3 “小猫设定”在这里如何体现

最重要的一点：

**当前项目的人设不是单独存一份设定文件，而是渗透在每条 `output` 里。**

所以你在写样本时要持续检查：

- 这句话像不像猫
- 这句话是不是中文语感自然
- 这句话是不是太像百科全书
- 这句话是不是脱离了小猫的感知方式

---

## 8. Step 3：写样本时要守住的质量边界

这一步决定最后训练出来的味道。

### 8.1 推荐的回答风格

- 尽量短
- 尽量具体
- 尽量有小猫感官
- 尽量带一点性格

例如：

推荐：

```text
用户：你在做什么
助手：看灰尘。它们在光里乱飞，我得盯着。
```

不推荐：

```text
用户：你在做什么
助手：我正在进行对环境中微粒运动轨迹的观察与分析。
```

### 8.2 推荐的世界观锚点

你写样本时，可以反复围绕这些锚点展开：

- 饭碗
- 罐头
- 猫粮袋子声音
- 窗台
- 阳光
- 纸箱
- 沙发
- 被子
- 主人
- 鸟
- 逗猫棒
- 激光点
- 门外动静
- 打雷
- 吹风机
- 医生
- 指甲
- 下巴
- 肚皮
- 尾巴

### 8.3 当前项目为什么仍然建议单轮

当前模型默认上下文长度是 [guppylm/config.py](/Users/changzechuan/AIProjects/guppylm/guppylm/config.py#L9) 的 `128`。

而 [guppylm/dataset.py](/Users/changzechuan/AIProjects/guppylm/guppylm/dataset.py#L20) 会直接截断超长样本。

所以你这套中文 + 小猫数据，仍然建议保持：

- 一条 user
- 一条 assistant
- 回复短

不要把它做成长对话小说。

### 8.4 不要做的几件事

- 不要把 `output` 写成系统提示词口吻
- 不要让小猫突然具备稳定的人类专家知识
- 不要一会儿极度幼态，一会儿又像学术论文
- 不要大量写超长回答
- 不要把类别命名得无法管理

---

## 9. Step 4：先把原始逻辑样本凑到足够规模

你可以按下面的节奏做：

### 9.1 起步阶段

先做 200 到 500 条高质量样本。

目的不是训练最终模型，而是先检查：

- 小猫语气是否稳定
- 类别是否合理
- 中文表达是否自然
- 主题分布是否偏斜

### 9.2 第一轮可训练阶段

建议至少做到几千条逻辑样本。

例如：

- 20 个类别
- 每类先做 200 到 500 条

那么总量大约是：

- 4,000 到 10,000 条

这已经足够做第一轮小模型实验。

### 9.3 如果你想接近当前仓库规模

当前英文 + 小鱼数据默认总量是 60,000。

中文 + 小猫如果也想接近这个量级，你可以用同样思路扩展：

- 先做类别设计
- 再做模板变体
- 再做人工筛选

但无论量多大，**原始逻辑样本层都要保留**。

---

## 10. Step 5：把逻辑样本导出成当前项目兼容格式

这一节是关键步骤。

你要从：

- `data_raw/cat_zh/samples.jsonl`

导出成：

- `data_cat_zh/train.jsonl`
- `data_cat_zh/eval.jsonl`
- `data_cat_zh/train_openai.jsonl`
- `data_cat_zh/eval_openai.jsonl`

### 10.1 当前项目兼容的训练主格式

和当前 [guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py#L1625) 完全一致：

```text
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

### 10.2 当前项目兼容的 OpenAI 导出格式

和当前 [guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py#L1632) 一致：

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### 10.3 推荐的导出脚本

下面这个脚本可以直接复用当前仓库逻辑，但不会覆盖现有 `data/`。

```bash
python - <<'PY'
import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

RAW_PATH = Path("data_raw/cat_zh/samples.jsonl")
OUT_DIR = Path("data_cat_zh")
EVAL_RATIO = 0.05

def format_sample(sample):
    return (
        f"<|im_start|>user\n{sample['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['output']}<|im_end|>"
    )

def to_openai(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["input"]},
            {"role": "assistant", "content": sample["output"]},
        ]
    }

samples = []
with RAW_PATH.open() as f:
    for lineno, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        for key in ("input", "output", "category"):
            if key not in sample:
                raise ValueError(f"{RAW_PATH}:{lineno} 缺少字段 {key}")
        if not isinstance(sample["input"], str) or not isinstance(sample["output"], str):
            raise TypeError(f"{RAW_PATH}:{lineno} 的 input/output 必须是字符串")
        samples.append(sample)

if not samples:
    raise ValueError("没有读取到任何逻辑样本")

random.shuffle(samples)
n_eval = int(len(samples) * EVAL_RATIO)
eval_samples = samples[:n_eval]
train_samples = samples[n_eval:]

OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, data in [
    ("train.jsonl", train_samples),
    ("eval.jsonl", eval_samples),
]:
    with (OUT_DIR / name).open("w") as f:
        for sample in data:
            row = {
                "text": format_sample(sample),
                "category": sample["category"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

for name, data in [
    ("train_openai.jsonl", train_samples),
    ("eval_openai.jsonl", eval_samples),
]:
    with (OUT_DIR / name).open("w") as f:
        for sample in data:
            f.write(json.dumps(to_openai(sample), ensure_ascii=False) + "\n")

cats = Counter(sample["category"] for sample in samples)
print(f"总样本: {len(samples)}")
print(f"训练集: {len(train_samples)}")
print(f"验证集: {len(eval_samples)}")
print("类别分布:")
for cat, count in sorted(cats.items(), key=lambda x: (-x[1], x[0])):
    print(f"  {cat}: {count}")
PY
```

### 10.4 导出后你应该得到什么

例如如果你原始逻辑样本有 10,000 条，`EVAL_RATIO=0.05`，那么你会得到：

- `train.jsonl`: 9,500 行
- `eval.jsonl`: 500 行
- `train_openai.jsonl`: 9,500 行
- `eval_openai.jsonl`: 500 行

---

## 11. Step 6：基于中文 + 小猫文本重训 tokenizer

这一步不能跳。

因为你现在的语料已经从：

- 英文 + 小鱼

变成了：

- 中文 + 小猫

语言分布和常见词片段已经明显变化。

### 11.1 当前项目里可以直接复用的能力

[guppylm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/prepare_data.py#L19) 里的 `train_tokenizer(...)` 可以直接复用。

它本身不依赖鱼人格内容。

### 11.2 推荐的 tokenizer 训练命令

```bash
python - <<'PY'
import json
from pathlib import Path
from guppylm.prepare_data import train_tokenizer

DATA_DIR = Path("data_cat_zh")
VOCAB_SIZE = 4096

texts = []
for name in ["train.jsonl", "eval.jsonl"]:
    path = DATA_DIR / name
    with path.open() as f:
        for line in f:
            texts.append(json.loads(line)["text"])

train_tokenizer(
    texts=texts,
    save_path=str(DATA_DIR / "tokenizer.json"),
    vocab_size=VOCAB_SIZE,
)
PY
```

### 11.3 词表大小怎么选

当前项目默认是 4096。

如果你是：

- 中文样本量还不大
- 先做第一轮实验

那么先用 `4096` 就可以。

如果你后面会做：

- 更大规模中文数据
- 中文 + 英文混合
- 更丰富的类别和表达

那你可以再评估：

- `6000`
- `8192`

但第一轮没有必要一上来就复杂化。

---

## 12. Step 7：做 4 轮校验，确认这套数据集真的“完整可用”

不要只看文件生成了就以为没问题。

推荐做下面 4 轮检查。

### 12.1 检查行数

```bash
wc -l data_cat_zh/train.jsonl
wc -l data_cat_zh/eval.jsonl
wc -l data_cat_zh/train_openai.jsonl
wc -l data_cat_zh/eval_openai.jsonl
```

### 12.2 检查字段结构

```bash
python - <<'PY'
import json
from pathlib import Path

for path in [
    Path("data_cat_zh/train.jsonl"),
    Path("data_cat_zh/eval.jsonl"),
]:
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            row = json.loads(line)
            assert "text" in row, f"{path}:{lineno} 缺少 text"
            assert isinstance(row["text"], str), f"{path}:{lineno} 的 text 不是字符串"
            assert "<|im_start|>user\n" in row["text"], f"{path}:{lineno} 缺少 user 起始"
            assert "\n<|im_start|>assistant\n" in row["text"], f"{path}:{lineno} 缺少 assistant 起始"
            assert row["text"].endswith("<|im_end|>"), f"{path}:{lineno} 未以 <|im_end|> 结束"

for path in [
    Path("data_cat_zh/train_openai.jsonl"),
    Path("data_cat_zh/eval_openai.jsonl"),
]:
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            row = json.loads(line)
            assert "messages" in row, f"{path}:{lineno} 缺少 messages"
            assert len(row["messages"]) == 2, f"{path}:{lineno} 不是单轮 user/assistant"
            assert row["messages"][0]["role"] == "user", f"{path}:{lineno} 第一条不是 user"
            assert row["messages"][1]["role"] == "assistant", f"{path}:{lineno} 第二条不是 assistant"

print("结构检查通过")
PY
```

### 12.3 检查 tokenizer 是否可正常加载

```bash
python - <<'PY'
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data_cat_zh/tokenizer.json")
text = "<|im_start|>user\n你在干嘛<|im_end|>\n<|im_start|>assistant\n我在晒太阳。<|im_end|>"
ids = tokenizer.encode(text).ids
decoded = tokenizer.decode(ids)

print(f"token 数: {len(ids)}")
print(decoded)
PY
```

### 12.4 直接用当前数据加载器做一次兼容性检查

这一步最关键，因为它验证的是“当前训练代码能不能真的吃进去”。

```bash
python - <<'PY'
from guppylm.dataset import GuppyDataset

dataset = GuppyDataset(
    path="data_cat_zh/train.jsonl",
    tokenizer_path="data_cat_zh/tokenizer.json",
    max_len=128,
)

print(f"可用样本数: {len(dataset)}")
x, y = dataset[0]
print(f"x 长度: {len(x)}")
print(f"y 长度: {len(y)}")
PY
```

只要这一步能正常跑通，你这套数据集对当前代码来说就已经基本合法了。

---

## 13. Step 8：什么才算“完整数据集已经准备完成”

对当前项目来说，最小可训练集合其实只有 3 个文件：

- `data_cat_zh/train.jsonl`
- `data_cat_zh/eval.jsonl`
- `data_cat_zh/tokenizer.json`

但我建议你把下面 5 个都保留：

- `data_cat_zh/train.jsonl`
- `data_cat_zh/eval.jsonl`
- `data_cat_zh/train_openai.jsonl`
- `data_cat_zh/eval_openai.jsonl`
- `data_cat_zh/tokenizer.json`

另外，最推荐再保留这 1 个源文件：

- `data_raw/cat_zh/samples.jsonl`

所以你真正应该维护的是这 6 个文件。

### 13.1 补充一句：如果你下一步要训练

当前 [guppylm/config.py](/Users/changzechuan/AIProjects/guppylm/guppylm/config.py#L35) 里的 `TrainConfig.data_dir` 默认还是 `data`。

也就是说：

- 你现在这份文档解决的是“如何准备一套完整、合法、独立的中文 + 小猫数据集”
- 如果你后面要在**不覆盖旧数据**的前提下训练它，训练入口还需要显式指向 `data_cat_zh/`

这一点属于“训练入口切换”问题，不属于本文档的“数据集准备”主体，但你需要知道它存在。

---

## 14. 常见坑

### 14.1 只改内容，不重训 tokenizer

不行。

你现在已经换成中文 + 小猫语料了，必须重训 `tokenizer.json`。

### 14.2 直接覆盖现有 `data/`

不推荐。

这样会把当前英文 + 小鱼数据资产混掉，也不利于以后比较两套数据。

### 14.3 往当前格式里强行加入 `system` 样本层

如果你的目标是“完全不破坏现有数据集逻辑”，就不要这样做。

当前项目的稳定逻辑是：

- 单轮
- `user`
- `assistant`

当前小鱼设定也是这样做出来的。

### 14.4 小猫说话不像猫，像搜索引擎

这是最常见的质量问题之一。

如果 `output` 失去小猫视角，那你即便格式全对，最后训练出来的也不会是你想要的人设。

### 14.5 样本太长

当前 `max_seq_len` 默认是 128。

过长样本会被截断，浪费有效信息，还可能让回答后半段被切掉。

### 14.6 类别设计过乱

如果类别命名一会儿按情绪，一会儿按场景，一会儿按功能，而且互相重叠严重，后面你会很难做数据清洗和配比调整。

---

## 15. 一个最实用的准备顺序

如果你想最稳地从 0 开始，我建议按下面顺序做：

1. 先写 1 页小猫设定说明
2. 先定 15 到 20 个类别
3. 先写 200 到 500 条高质量逻辑样本
4. 小范围人工检查语气是否稳定
5. 扩充到几千条逻辑样本
6. 导出 `data_cat_zh/train.jsonl` / `eval.jsonl`
7. 导出 `train_openai.jsonl` / `eval_openai.jsonl`
8. 训练 `data_cat_zh/tokenizer.json`
9. 跑结构检查、tokenizer 检查、dataset 兼容性检查

走完这 9 步，你的中文 + 小猫数据集就算准备完整了。

---

## 16. 最后给你一句判断标准

以后你只要用这句话检查自己有没有走偏：

**只要你的中文 + 小猫数据仍然遵守当前项目的 `text` 格式、继续使用 `<pad> / <|im_start|> / <|im_end|>` 特殊 token、能被当前 `GuppyDataset` 正常加载，而且 tokenizer 是基于同一套中文 + 小猫语料重新训练出来的，那么这套数据集对当前项目就是合法的。**

如果你还想让这套数据更容易维护、扩展、迁移，那么继续保留：

- `category`
- OpenAI 导出文件
- 原始逻辑样本文件

这样你得到的就不是一堆一次性训练文件，而是一套真正可持续维护的数据资产。
