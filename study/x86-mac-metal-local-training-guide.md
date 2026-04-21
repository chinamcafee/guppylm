# GuppyLM 在 Intel Mac + AMD 显卡 + Metal 上的本地训练与推理指南

本文档基于当前仓库已经验证通过的方案整理，目标是让你在一台 `x86_64` 架构、带 AMD 独显、支持 Metal 的 Mac 上，完整走通以下流程：

1. 重建与当前仓库一致的 Conda 环境
2. 验证 PyTorch 的 `mps` 后端可用
3. 本地生成训练数据集与 tokenizer
4. 本地训练 GuppyLM
5. 使用训练好的本地模型进行推理

这份文档默认你在仓库根目录执行命令。

---

## 1. 已验证的环境前提

当前仓库中已经验证成功的关键点如下：

- 机器类型：Intel Mac，`osx-64`
- GPU：AMD 独显，Metal 可用
- Python：3.11
- PyTorch：2.2.2
- NumPy：1.26.4
- 设备后端：`mps`

注意：

- 这套方案依赖仓库根目录的 [environment.osx-64.lock.yml](/Users/changzechuan/AIProjects/guppylm/environment.osx-64.lock.yml)。
- 这个锁文件只适合 `osx-64`，也就是 Intel Mac。
- 不建议在这条训练链路上直接使用 Python 3.13；本仓库当前验证通过的是 Python 3.11 环境。

---

## 2. 第一步：获取项目代码

如果你还没有本项目：

```bash
git clone <你的 git 仓库地址>
cd guppylm
```

如果你已经在项目目录内，可以直接进入下一步。

---

## 3. 第二步：重建与当前项目一致的 Conda 环境

仓库不会提交本地虚拟环境目录 `.conda-py311-mps/`，但会提交环境锁文件。因此正确做法不是复制目录，而是从锁文件重建。

### 3.1 创建环境

```bash
conda env create -p ./.conda-py311-mps -f environment.osx-64.lock.yml
```

### 3.2 激活环境

```bash
conda activate /绝对路径/guppylm/.conda-py311-mps
```

例如：

```bash
conda activate /Users/changzechuan/AIProjects/guppylm/.conda-py311-mps
```

### 3.3 如果环境已经存在，按锁文件更新

```bash
conda env update -p ./.conda-py311-mps -f environment.osx-64.lock.yml --prune
```

`--prune` 的作用是删除锁文件里没有的包，让环境尽量和锁文件保持一致。

---

## 4. 第三步：验证 MPS / Metal 是否真的可用

训练能否走 Metal，关键不在于 `torch.device("mps")` 能不能写出来，而在于 `torch.backends.mps.is_available()` 是否为真，以及是否真的能在 `mps` 上创建 tensor。

### 4.1 用一条命令验证

```bash
python -c "import torch; print('torch=', torch.__version__); print('mps_built=', torch.backends.mps.is_built()); print('mps_available=', torch.backends.mps.is_available()); print(torch.ones(1, device='mps'))"
```

如果输出类似下面这样，就说明当前机器上的 Metal 训练路径可用：

```text
torch= 2.2.2
mps_built= True
mps_available= True
tensor([1.], device='mps:0')
```

如果这里失败，先不要往下训练。优先检查：

- 当前是否真的激活了 `.conda-py311-mps`
- 是否是在 Intel Mac `osx-64` 环境里
- 是否误装了别的 Python / torch 版本

---

## 5. 第四步：理解训练前后会用到哪些目录

在正式开始前，先明确本项目的输入输出路径。

### 5.1 数据准备阶段会生成

在 [guppylm/prepare_data.py](/Users/changzechuan/AIProjects/guppylm/guppylm/prepare_data.py) 中，执行 `prepare()` 后会生成：

- `data/train.jsonl`
- `data/eval.jsonl`
- `data/train_openai.jsonl`
- `data/eval_openai.jsonl`
- `data/tokenizer.json`

### 5.2 训练阶段会生成

在 [guppylm/train.py](/Users/changzechuan/AIProjects/guppylm/guppylm/train.py) 中，默认输出目录是 `checkpoints/`，会产生：

- `checkpoints/config.json`
- `checkpoints/best_model.pt`
- `checkpoints/final_model.pt`
- 以及中间的 `step_*.pt`

### 5.3 推理阶段默认读取

在 [guppylm/inference.py](/Users/changzechuan/AIProjects/guppylm/guppylm/inference.py) 中，默认读取：

- checkpoint：`checkpoints/best_model.pt`
- tokenizer：`data/tokenizer.json`

所以，只要你走默认路径，训练完成后可以直接进入推理，不需要额外搬运文件。

---

## 6. 第五步：本地准备数据集与 tokenizer

本项目的数据不是从远端下载的训练主数据，而是本地脚本动态生成的。入口在 [guppylm/__main__.py](/Users/changzechuan/AIProjects/guppylm/guppylm/__main__.py)。

### 6.1 直接使用默认命令准备完整数据

```bash
python -m guppylm prepare
```

这一步会做三件事：

1. 生成 60,000 条合成对话样本
2. 切分出训练集与验证集
3. 使用生成后的文本训练一个 BPE tokenizer

### 6.2 你应该看到的结果

成功后，终端一般会看到类似这些信息：

- `Generating 60000 samples...`
- `Generated 60000 samples ...`
- `Training BPE tokenizer ...`
- `Tokenizer saved to data/tokenizer.json`

### 6.3 检查数据文件是否生成成功

```bash
ls -lh data
```

正常情况下至少应该看到：

- `train.jsonl`
- `eval.jsonl`
- `tokenizer.json`

---

## 7. 第六步：开始本地训练

### 7.1 直接按默认配置训练

```bash
python -m guppylm train
```

训练脚本会自动选设备，逻辑在 [guppylm/train.py](/Users/changzechuan/AIProjects/guppylm/guppylm/train.py)：

1. 先尝试 `cuda`
2. 然后尝试 `mps`
3. 最后回退到 `cpu`

在你的 Intel Mac + AMD + Metal 环境里，如果第三步的验证通过，这里应该打印：

```text
Device: mps
```

### 7.2 当前默认训练超参

默认训练配置在 [guppylm/config.py](/Users/changzechuan/AIProjects/guppylm/guppylm/config.py) 中：

- `batch_size = 32`
- `max_steps = 10000`
- `eval_interval = 200`
- `save_interval = 500`
- `device = "auto"`
- `data_dir = "data"`
- `output_dir = "checkpoints"`

也就是说，如果你不改任何代码，完整训练会：

- 从 `data/` 读取训练数据
- 把权重保存到 `checkpoints/`
- 自动尝试使用 `mps`

### 7.3 如果你想强制指定 MPS

可以把 [guppylm/config.py](/Users/changzechuan/AIProjects/guppylm/guppylm/config.py) 中的：

```python
device: str = "auto"
```

改成：

```python
device: str = "mps"
```

不过在当前验证通过的环境里，通常没必要，`auto` 就会选到 `mps`。

### 7.4 如果你想先做小规模烟雾测试

当前 `train.py` 没有命令行参数来覆盖 `max_steps`。因此如果你想快速验证流程，最直接的方法是临时修改 [guppylm/config.py](/Users/changzechuan/AIProjects/guppylm/guppylm/config.py) 里的这些字段：

```python
batch_size: int = 8
max_steps: int = 20
eval_interval: int = 10
save_interval: int = 20
device: str = "mps"
output_dir: str = "checkpoints-smoke-mps"
```

然后运行：

```bash
python -m guppylm train
```

确认训练链路通了之后，再把它改回完整训练配置。

### 7.5 关于 MPS 训练速度的现实预期

当前训练代码只给 CUDA 开了 AMP，MPS 路径走的是普通精度训练，不是 CUDA 的混合精度路径。因此：

- 能训练
- 能保存 checkpoint
- 但速度不会像 NVIDIA CUDA 那样快

这不是“不能训练”，只是性能特征不同。

---

## 8. 第七步：判断训练是否完成

训练过程中，你会看到类似这样的日志：

```text
Device: mps
GuppyLM: 8,726,016 params (8.7M)
Train: 57000, Eval: 3000

Training for 10000 steps...
  Step |         LR |      Train |       Eval |     Time
--------------------------------------------------------
...
```

训练结束后会打印类似：

```text
Done! <总耗时>s, best eval: <数值>
```

### 8.1 检查模型产物

```bash
ls -lh checkpoints
```

重点关注：

- `checkpoints/best_model.pt`
- `checkpoints/final_model.pt`
- `checkpoints/config.json`

如果这三个文件都在，说明本地训练已经完成并且推理所需的关键文件已经具备。

---

## 9. 第八步：使用本地训练好的模型进行推理

本项目已经内置了推理入口，不需要你自己再写推理脚本。

### 9.1 单轮推理

```bash
python -m guppylm chat --device mps --prompt "tell me a joke"
```

这条命令会：

1. 加载 `checkpoints/best_model.pt`
2. 加载 `data/tokenizer.json`
3. 在 `mps` 上执行推理
4. 输出一条回答后退出

### 9.2 进入交互式聊天

```bash
python -m guppylm chat --device mps
```

进入后可以直接输入：

```text
You> hi guppy
Guppy> ...
```

退出方式：

- 输入 `quit`
- 输入 `exit`
- 输入 `q`

### 9.3 如果你想显式指定模型路径

当你不使用默认 `checkpoints/` 路径时，可以这样写：

```bash
python -m guppylm chat \
  --checkpoint checkpoints/best_model.pt \
  --tokenizer data/tokenizer.json \
  --device mps \
  --prompt "what is the meaning of life"
```

如果你把模型保存到了别的目录，例如 `checkpoints-smoke-mps/`，只需要把 `--checkpoint` 改成对应路径即可。

---

## 10. 第九步：完整流程的最短命令版

如果你已经确认是 Intel Mac，并且当前目录就是仓库根目录，那么完整流程可以压缩成下面这几步。

### 10.1 首次在新机器上执行

```bash
conda env create -p ./.conda-py311-mps -f environment.osx-64.lock.yml
conda activate /绝对路径/guppylm/.conda-py311-mps
python -c "import torch; print(torch.backends.mps.is_available()); print(torch.ones(1, device='mps'))"
python -m guppylm prepare
python -m guppylm train
python -m guppylm chat --device mps --prompt "tell me a joke"
```

### 10.2 后续重复训练时执行

```bash
conda activate /绝对路径/guppylm/.conda-py311-mps
python -m guppylm prepare
python -m guppylm train
python -m guppylm chat --device mps
```

---

## 11. 常见问题

### 11.1 为什么我能写 `torch.device("mps")`，但训练还是没走 MPS？

因为真正的判断条件不是对象能不能构造，而是：

```python
torch.backends.mps.is_available()
```

它必须返回 `True`，而且还要能实际创建 MPS tensor。

### 11.2 为什么我明明是 Mac，却装不上你当前用的 torch？

因为这条训练方案是按 `osx-64` 的 Intel Mac 锁死的。你如果换成：

- Apple Silicon
- Python 3.13
- 其他来源的 Python 发行版

就可能拿不到同一套 wheel。

最稳妥的方式就是直接使用 [environment.osx-64.lock.yml](/Users/changzechuan/AIProjects/guppylm/environment.osx-64.lock.yml) 来重建。

### 11.3 为什么推理时找不到模型？

默认聊天命令会去找：

- `checkpoints/best_model.pt`
- `data/tokenizer.json`

如果你训练时改了输出目录，就要在推理时显式传入 `--checkpoint`。

### 11.4 为什么训练可以跑，但比预期慢？

因为当前代码里只有 CUDA 路径启用了 AMP，MPS 路径没有。对于 Intel Mac + AMD + Metal，这意味着：

- 功能可用
- 训练可完成
- 但速度不是最优

---

## 12. 推荐的实际操作顺序

如果你是第一次在新机器上跑，建议这样做：

1. 先重建 Conda 环境
2. 先验证 `mps` 可用
3. 先跑 `python -m guppylm prepare`
4. 先把 `TrainConfig.max_steps` 改小做一次烟雾测试
5. 确认能生成 `best_model.pt`
6. 用 `python -m guppylm chat --device mps --prompt ...` 测试推理
7. 最后再恢复完整训练参数，跑正式训练

这样做的好处是，你可以先确认环境、数据、训练、推理四条链路都通，再投入完整训练时间。
