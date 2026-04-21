# CatLM 从零实现文档目录

这组文档对应的是当前仓库里新增的 [catlm](/Users/changzechuan/AIProjects/guppylm/catlm) 实现。

目标不是修改现有的 `guppylm/`，而是在**完全不改动 `guppylm/` 和 `data/` 任何内容**的前提下，新增一套：

- 中文 + 小猫 AI 设定的数据生成逻辑
- 独立的数据准备与 tokenizer 训练流程
- 独立的训练与推理入口
- 适用于 Intel Mac + AMD + Metal 的完整本地训练说明

阅读顺序按实际工作顺序来排：

1. [01-boundaries-and-goals.md](/Users/changzechuan/AIProjects/guppylm/study/catlm-from-scratch/01-boundaries-and-goals.md)
2. [02-environment-and-metal.md](/Users/changzechuan/AIProjects/guppylm/study/catlm-from-scratch/02-environment-and-metal.md)
3. [03-code-structure.md](/Users/changzechuan/AIProjects/guppylm/study/catlm-from-scratch/03-code-structure.md)
4. [04-persona-and-raw-data.md](/Users/changzechuan/AIProjects/guppylm/study/catlm-from-scratch/04-persona-and-raw-data.md)
5. [05-dataset-preparation.md](/Users/changzechuan/AIProjects/guppylm/study/catlm-from-scratch/05-dataset-preparation.md)
6. [06-tokenizer-and-test-set.md](/Users/changzechuan/AIProjects/guppylm/study/catlm-from-scratch/06-tokenizer-and-test-set.md)
7. [07-training-on-x86-mac-metal.md](/Users/changzechuan/AIProjects/guppylm/study/catlm-from-scratch/07-training-on-x86-mac-metal.md)
8. [08-inference-and-acceptance.md](/Users/changzechuan/AIProjects/guppylm/study/catlm-from-scratch/08-inference-and-acceptance.md)

如果你只想快速走通整条链路，最短路径是：

1. 先读第 2 章，确认 Intel Mac + Metal 环境无误
2. 再读第 5 章，生成 `data_cat_zh/`
3. 再读第 7 章，训练 `checkpoints_catlm/`
4. 最后读第 8 章，用 `python -m catlm chat --device mps` 做推理验收

这套文档默认你在仓库根目录 `/Users/changzechuan/AIProjects/guppylm` 执行命令。
