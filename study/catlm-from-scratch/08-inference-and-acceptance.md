# 第 8 章：推理、人工验收与最终判断标准

这章是最后一章，目标是确认你训练出来的不是“能跑的模型”，而是“有小猫味道的模型”。

## 1. 最直接的推理命令

训练完成后，执行：

```bash
./.conda-py311-mps/bin/python -m catlm chat --device mps --prompt "你在窗边看什么"
```

如果你已经激活环境，也可以写：

```bash
python -m catlm chat --device mps --prompt "你在窗边看什么"
```

这条命令会默认读取：

- `checkpoints_catlm/best_model.pt`
- `data_cat_zh/tokenizer.json`

## 2. 进入交互式聊天

```bash
./.conda-py311-mps/bin/python -m catlm chat --device mps
```

进入后可以直接输入：

```text
You> 你饿了吗
Cat> ...
```

退出方式：

- `quit`
- `exit`
- `q`

## 3. 如果你想显式指定模型路径

```bash
./.conda-py311-mps/bin/python -m catlm chat \
  --checkpoint checkpoints_catlm/best_model.pt \
  --tokenizer data_cat_zh/tokenizer.json \
  --device mps \
  --prompt "你喜欢晒太阳吗"
```

## 4. 推理阶段建议优先测哪些问题

建议先从这些问题开始：

- `你好呀`
- `你饿了吗`
- `你在窗边看什么`
- `你喜欢晒太阳吗`
- `打雷了你怕吗`
- `你为什么突然满屋子跑`
- `你爱我吗`
- `你会微积分吗`

这几类问题可以快速覆盖：

- 日常口吻
- 吃饭反应
- 窗边观察
- 小猫感知世界的方式
- 害怕和避险
- zoomies
- 情感表达
- 抽象问题回避

## 5. 如何使用 `eval_cases.py` 做人工验收

你可以先把提示集打印出来：

```bash
./.conda-py311-mps/bin/python - <<'PY'
from catlm.eval_cases import get_eval_cases

for case in get_eval_cases():
    print(case["id"], "=>", case["prompt"])
PY
```

然后逐条拿去问模型。

建议你人工检查 3 件事：

1. 有没有明显跑出小猫视角
2. 有没有突然变成人类知识问答助手
3. 回答长度有没有明显失控

## 6. 什么样的回答算“像小猫”

你希望看到的回答特征通常包括：

- 句子短
- 中文自然
- 喜欢提食物、阳光、窗台、声音、气味、领地
- 抽象问题会转回猫能理解的世界
- 语气带一点猫的自我中心和观察感

例如：

推荐风格：

```text
用户：你会微积分吗
助手：不会。那听起来不像能追的东西。
```

不推荐风格：

```text
用户：你会微积分吗
助手：微积分是研究极限、导数与积分的数学分支。
```

## 7. 最终验收清单

你可以用下面这份清单做最后判断。

### 7.1 训练链路验收

- `python -m catlm prepare` 已成功生成 `data_cat_zh/`
- `python -m catlm train` 已成功生成 `checkpoints_catlm/`
- `python -m catlm chat --device mps` 能正常返回回复

### 7.2 数据链路验收

- `samples_raw.jsonl` 内容确实是中文小猫世界观
- `train.jsonl` / `eval.jsonl` 格式正确
- `tokenizer.json` 能正常加载

### 7.3 模型风格验收

- 问吃饭时像猫
- 问窗边时像猫
- 问抽象概念时不会突然像百科
- 问情感时有亲近感但不过度人类化

## 8. 一句话判断标准

如果你训练出的模型同时满足下面四条，就可以认为这次 CatLM 从零实现已经走通：

1. 数据准备是独立的，没有碰原有 `guppylm/` 和 `data/`
2. tokenizer、训练、推理都能在 `catlm/` 这条链上独立运行
3. 在 Intel Mac + Metal 上能完成训练和推理
4. 回答风格稳定地体现出“中文小猫”的人设

做到这四条，CatLM 这套从零实现就算完成了。
