# 第 4 章：中文小猫人设与原始逻辑样本

这一章对应的是 [study/dataset-preparation-zh-kitten-step-by-step.md](/Users/changzechuan/AIProjects/guppylm/study/dataset-preparation-zh-kitten-step-by-step.md) 里的“内容层”。

## 1. CatLM 的人设不是写在 system prompt 里的

当前 CatLM 没有额外引入一个统一的 `system` 样本层。

这和 GuppyLM 一样：

- 人设不靠外部 prompt 文件
- 人设直接写进训练数据里

因此 CatLM 的“小猫感”来自：

1. `output` 的写法
2. 主题分布
3. 语料整体风格

## 2. 当前 CatLM 的中文小猫设定

[catlm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/catlm/generate_data.py) 里的默认设定是：

- 说中文
- 句子短
- 以家猫感知世界
- 关注食物、窗台、阳光、气味、动静、领地和主人
- 不擅长抽象人类概念
- 会对声音、温度、光线、移动物体快速反应

## 3. CatLM 的原始逻辑样本结构

和 GuppyLM 一样，逻辑样本的基本结构仍然是：

```json
{
  "input": "用户输入",
  "output": "小猫输出",
  "category": "主题类别"
}
```

这一步还不是最终训练格式。

它只是“内容本体”。

## 4. 这批逻辑样本现在如何落地保存

和原始 GuppyLM 不同，CatLM 默认会把逻辑样本额外落一份文件：

- `data_cat_zh/samples_raw.jsonl`

这个文件来自 [catlm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/catlm/generate_data.py) 的 `generate_dataset(...)`。

它的价值是：

- 更方便人工抽样检查
- 更方便以后人工修正某些类别
- 更方便看“内容层”和“训练层”之间的区别

## 5. 目前 CatLM 预置了哪些类别

CatLM 当前预置了 40+ 个类别，例如：

- `greeting`
- `feeling`
- `food`
- `treat`
- `sleep`
- `play`
- `window`
- `bird`
- `box`
- `sun`
- `rain`
- `noise`
- `fear`
- `love`
- `grooming`
- `doctor`
- `water`
- `territory`
- `zoomies`
- `dream`

这些类别的目标不是做百科问答，而是把“小猫会怎么理解世界”分散到不同情景里。

## 6. 一条 CatLM 逻辑样本长什么样

生成后的 `samples_raw.jsonl` 里，一行会类似这样：

```json
{"input":"你饿了吗","output":"饿。只要你碰一下冻干袋子，我就会立刻过去。","category":"food"}
```

或者：

```json
{"input":"你在窗边看什么","output":"窗外有小鸟。我得盯着。","category":"window"}
```

## 7. 为什么要先关心逻辑样本，而不是先关心 train.jsonl

因为你以后要调整 CatLM 的风格时，最应该改的是：

- 类别设计
- 模板
- 逻辑样本分布

而不是直接手改最终 `train.jsonl`。

如果你直接在最终训练文件上做大量人工改写，会很难：

- 追踪改动来源
- 重复生成
- 做中期迭代

## 8. 当前 CatLM 的内容质量边界

你检查 `samples_raw.jsonl` 时，建议优先看这几件事：

1. 回答像不像猫
2. 中文是否自然
3. 句子是否太长
4. 有没有突然变成人类专家口吻
5. 类别之间是否过度重复

只要这一步控制住，后面的 tokenizer、训练和推理才有意义。
