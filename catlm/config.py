"""CatLM configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CatConfig:
    vocab_size: int = 4096
    max_seq_len: int = 128
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_hidden: int = 768
    dropout: float = 0.1

    # Special tokens
    pad_id: int = 0
    bos_id: int = 1           # <|im_start|>
    eos_id: int = 2           # <|im_end|>


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 12000
    eval_interval: int = 200
    eval_max_batches: Optional[int] = None
    save_interval: int = 500
    early_stop_patience: Optional[int] = 12
    early_stop_min_delta: float = 1e-3
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42
    data_dir: str = "data_cat_zh"
    output_dir: str = "checkpoints_catlm"
