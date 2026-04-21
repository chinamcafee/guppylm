# 第 2 章：Intel Mac 环境准备与 Metal 验证

这一章完全沿用 [study/x86-mac-metal-local-training-guide.md](/Users/changzechuan/AIProjects/guppylm/study/x86-mac-metal-local-training-guide.md) 的环境前提，但把后续训练对象换成了 `catlm`。

## 1. 环境前提

当前仓库已经验证过的本地训练环境是：

- Intel Mac
- `osx-64`
- AMD 独显
- Metal 可用
- Python 3.11
- PyTorch 2.2.2

最关键的环境锁文件还是仓库根目录的 [environment.osx-64.lock.yml](/Users/changzechuan/AIProjects/guppylm/environment.osx-64.lock.yml)。

## 2. 创建 Conda 环境

在仓库根目录执行：

```bash
conda env create -p ./.conda-py311-mps -f environment.osx-64.lock.yml
```

如果环境已经存在，则更新：

```bash
conda env update -p ./.conda-py311-mps -f environment.osx-64.lock.yml --prune
```

## 3. 激活环境

```bash
conda activate /Users/changzechuan/AIProjects/guppylm/.conda-py311-mps
```

如果你不想先 `conda activate`，也可以直接用环境里的 Python：

```bash
./.conda-py311-mps/bin/python -V
```

## 4. 验证 Metal / MPS 是否真的可用

先不要急着训练 CatLM。先确认 PyTorch 的 `mps` 后端可用。

```bash
./.conda-py311-mps/bin/python -c "import torch; print('torch=', torch.__version__); print('mps_built=', torch.backends.mps.is_built()); print('mps_available=', torch.backends.mps.is_available()); print(torch.ones(1, device='mps'))"
```

如果输出类似：

```text
torch= 2.2.2
mps_built= True
mps_available= True
tensor([1.], device='mps:0')
```

说明这台 Intel Mac 的 Metal 路径已经准备好。

## 5. 为什么 CatLM 也可以直接用这套环境

因为 [catlm/model.py](/Users/changzechuan/AIProjects/guppylm/catlm/model.py)、[catlm/train.py](/Users/changzechuan/AIProjects/guppylm/catlm/train.py) 和 [catlm/inference.py](/Users/changzechuan/AIProjects/guppylm/catlm/inference.py) 仍然是 PyTorch 本地实现。

它没有新增新的底层依赖。

所以环境层面仍然对标 GuppyLM：

- 还是同一个 torch
- 还是同一个 tokenizers
- 还是同一条 `mps` 设备选择逻辑

## 6. 这章结束时你应该达到的状态

你继续往后走之前，至少应该确认下面 3 件事：

1. `.conda-py311-mps/` 已存在
2. 环境里的 `python` 能导入 `torch`
3. `torch.backends.mps.is_available()` 为 `True`

如果这三条还没满足，后面关于数据、训练、推理的步骤都先不要往下做。
