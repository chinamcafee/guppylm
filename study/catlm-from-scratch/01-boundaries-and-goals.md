# 第 1 章：边界、目标与最终产物

这一章先讲清楚这次工作的边界。

## 1. 这次到底新建了什么

当前仓库里原有的是 [guppylm](/Users/changzechuan/AIProjects/guppylm/guppylm)。

这次新增的是一套并行实现：

- [catlm](/Users/changzechuan/AIProjects/guppylm/catlm)

它不是去改写原有 `guppylm`，而是独立保留了一份“结构对标、内容换成中文小猫”的版本。

## 2. 这次明确不做什么

这次实现明确遵守下面三条：

1. 不改 [guppylm](/Users/changzechuan/AIProjects/guppylm/guppylm) 里的任何文件
2. 不改现有 [data](/Users/changzechuan/AIProjects/guppylm/data) 里的任何文件
3. 不让新的中文小猫数据覆盖旧的英文小鱼数据

所以新的默认路径全部改成了独立目录。

## 3. CatLM 的默认输入输出目录

新的默认路径定义在 [catlm/config.py](/Users/changzechuan/AIProjects/guppylm/catlm/config.py)：

- 训练数据目录：`data_cat_zh/`
- 训练输出目录：`checkpoints_catlm/`

这意味着：

- 原有小鱼数据还在 `data/`
- 新的小猫数据会进 `data_cat_zh/`
- 原有小鱼模型 checkpoint 仍然可以放在 `checkpoints/`
- 新的小猫模型 checkpoint 会放进 `checkpoints_catlm/`

## 4. CatLM 对标了 GuppyLM 的哪些部分

对标关系如下：

- 模型结构：对标 [guppylm/model.py](/Users/changzechuan/AIProjects/guppylm/guppylm/model.py)
- 训练循环：对标 [guppylm/train.py](/Users/changzechuan/AIProjects/guppylm/guppylm/train.py)
- tokenizer 训练：对标 [guppylm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/prepare_data.py)
- 数据格式：对标 [guppylm/generate_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/generate_data.py) 里的 `format_sample()` / `to_openai()`
- 推理接口：对标 [guppylm/inference.py](/Users/changzechuan/AIProjects/guppylm/guppylm/inference.py)

但以下内容换成了中文小猫版本：

- 人设
- 数据生成模板
- 默认数据目录
- 默认输出目录
- 测试提示集

## 5. 这套链路最终会产出什么

如果你完整走完 CatLM 的默认流程，最后至少会得到这些文件：

### 5.1 数据侧

- `data_cat_zh/samples_raw.jsonl`
- `data_cat_zh/train.jsonl`
- `data_cat_zh/eval.jsonl`
- `data_cat_zh/train_openai.jsonl`
- `data_cat_zh/eval_openai.jsonl`
- `data_cat_zh/tokenizer.json`

### 5.2 模型侧

- `checkpoints_catlm/config.json`
- `checkpoints_catlm/best_model.pt`
- `checkpoints_catlm/final_model.pt`
- 若干 `checkpoints_catlm/step_*.pt`

## 6. 整条工作链路的顺序

实际顺序就是下面这 8 步：

1. 准备 Intel Mac + Metal 环境
2. 验证 `mps` 可用
3. 生成中文小猫逻辑样本
4. 导出 `train/eval/openai` 数据文件
5. 基于文本语料训练 tokenizer
6. 准备验证集与手工测试样例
7. 用 Metal 训练 CatLM
8. 用训练好的 checkpoint 做推理验收

后面的章节会按这个顺序展开。
