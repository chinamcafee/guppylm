# 第 7 章：在 Intel Mac + AMD + Metal 上训练 CatLM

这一章对应 GuppyLM 的本地训练指南，但训练对象换成了 `catlm`。

## 1. 默认训练命令

```bash
./.conda-py311-mps/bin/python -m catlm train
```

如果你已经激活了环境，也可以直接写：

```bash
python -m catlm train
```

## 2. CatLM 如何自动选择设备

[catlm/train.py](/Users/changzechuan/AIProjects/guppylm/catlm/train.py) 的设备选择逻辑仍然和 GuppyLM 一致：

1. 先试 `cuda`
2. 再试 `mps`
3. 最后回退 `cpu`

在 Intel Mac + AMD + Metal 的环境里，如果第 2 章的验证通过，这里应该打印：

```text
Device: mps
```

## 3. CatLM 的默认训练配置

默认配置在 [catlm/config.py](/Users/changzechuan/AIProjects/guppylm/catlm/config.py)：

- `batch_size = 32`
- `learning_rate = 3e-4`
- `max_steps = 25000`
- `eval_interval = 200`
- `save_interval = 500`
- `device = "auto"`
- `data_dir = "data_cat_zh"`
- `output_dir = "checkpoints_catlm"`

这意味着默认正式训练会：

- 从 `data_cat_zh/` 读数据
- 向 `checkpoints_catlm/` 写模型
- 自动优先走 `mps`

## 4. 训练前最好再确认一次输入文件

```bash
ls -lh data_cat_zh/train.jsonl
ls -lh data_cat_zh/eval.jsonl
ls -lh data_cat_zh/tokenizer.json
```

只要这三个文件没问题，训练侧的数据输入就齐了。

## 5. 正式训练时你会看到什么

训练开始后，终端日志通常会类似：

```text
Device: mps
CatLM: ~13.2M params
Train: 54000, Eval: 6000

Training for 25000 steps...
  Step |         LR |      Train |       Eval |     Time
--------------------------------------------------------
...
```

训练过程中会发生这些事：

- 每 100 step 打一次训练日志
- 每 `eval_interval` 做一次验证
- 每 `save_interval` 保存一次中间 checkpoint
- 如果验证 loss 更好，会更新 `best_model.pt`

## 6. 训练完成后会有哪些文件

```bash
ls -lh checkpoints_catlm
```

重点关注：

- `checkpoints_catlm/config.json`
- `checkpoints_catlm/best_model.pt`
- `checkpoints_catlm/final_model.pt`

如果这三个都在，说明一轮完整训练已经完成。

## 6.1 如果你想从已有 checkpoint 继续训练

当前 CatLM 现在额外支持一个独立续训命令：

```bash
./.conda-py311-mps/bin/python -m catlm resume-train
```

它和 `python -m catlm train` 的职责是分开的：

- `python -m catlm train`：始终从头训练
- `python -m catlm resume-train`：从 `checkpoints_catlm/` 里最新的训练 checkpoint 继续

如果你已经把 [catlm/config.py](/Users/changzechuan/AIProjects/guppylm/catlm/config.py) 里的 `max_steps` 从较小值改大，例如：

- 先训到 `10000`
- 再把 `max_steps` 改到 `20000`

那么接下来应执行的是：

```bash
./.conda-py311-mps/bin/python -m catlm resume-train
```

如果你想显式指定某个 checkpoint，也可以在命令后追加路径：

```bash
./.conda-py311-mps/bin/python -m catlm resume-train checkpoints_catlm/step_10000.pt
```

## 7. 如果你想先做烟雾训练

和 GuppyLM 一样，当前 `train.py` 没有暴露命令行参数来覆盖 `max_steps`。

因此最稳妥的烟雾测试方式仍然是临时修改 [catlm/config.py](/Users/changzechuan/AIProjects/guppylm/catlm/config.py) 里的 `TrainConfig`：

```python
batch_size: int = 8
max_steps: int = 20
eval_interval: int = 10
save_interval: int = 20
device: str = "mps"
output_dir: str = "checkpoints_catlm_smoke"
```

然后运行：

```bash
./.conda-py311-mps/bin/python -m catlm train
```

确认训练链路能通之后，再把配置改回正式值。

## 8. 关于 MPS 训练速度的现实预期

CatLM 仍然继承了和 GuppyLM 同样的事实：

- 训练能跑
- checkpoint 能保存
- 推理能走 `mps`
- 但 MPS 路径不是 CUDA AMP 的那种速度表现

所以这里要追求的是：

- 本地可训练
- 流程透明
- 角色模型可迭代

而不是追求超高速训练。

## 9. 这一章结束时的验收标准

你往下一章走之前，至少要确认：

1. 终端明确打印了 `Device: mps`
2. `checkpoints_catlm/best_model.pt` 已出现
3. `checkpoints_catlm/config.json` 已出现

如果没有这三条，先不要开始做推理验收。
