"""CatLM training loop."""

import json
import math
import os
import random
import time

import torch

from .config import CatConfig, TrainConfig
from .dataset import get_dataloader
from .model import CatLM


def get_device(config):
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config.device)


def get_lr(step, config):
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return config.min_lr + (config.learning_rate - config.min_lr) * coeff


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    model.eval()
    total_loss, n = 0, 0
    for x, y in loader:
        if n >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(1, n)


def _cat_config_from_dict(data):
    valid_fields = {f.name for f in CatConfig.__dataclass_fields__.values()}
    return CatConfig(**{k: v for k, v in data.items() if k in valid_fields})


def _make_optimizer(model, tc):
    return torch.optim.AdamW(
        model.parameters(), lr=tc.learning_rate,
        weight_decay=tc.weight_decay, betas=(0.9, 0.95),
    )


def _make_scaler(device):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def _get_rng_state(device):
    state = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state, device):
    if not state:
        return
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda" and "cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def _save_training_checkpoint(
    path,
    step,
    model,
    mc,
    tc,
    optimizer,
    scaler,
    best_eval,
    train_losses=None,
    eval_loss=None,
    device=None,
):
    ckpt = {
        "step": step,
        "next_step": step + 1,
        "model_state_dict": model.state_dict(),
        "config": vars(mc),
        "train_config": vars(tc),
        "best_eval": best_eval,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    if eval_loss is not None:
        ckpt["eval_loss"] = eval_loss
    if train_losses is not None:
        ckpt["train_losses"] = train_losses
    if device is not None:
        ckpt.update(_get_rng_state(device))
    torch.save(ckpt, path)


def _write_run_config(tc, mc):
    os.makedirs(tc.output_dir, exist_ok=True)
    with open(os.path.join(tc.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": vars(mc), "train": vars(tc)}, f, indent=2)


def _build_loaders(tc, mc):
    tokenizer_path = os.path.join(tc.data_dir, "tokenizer.json")
    train_loader = get_dataloader(
        os.path.join(tc.data_dir, "train.jsonl"), tokenizer_path,
        mc.max_seq_len, tc.batch_size, shuffle=True,
    )
    eval_loader = get_dataloader(
        os.path.join(tc.data_dir, "eval.jsonl"), tokenizer_path,
        mc.max_seq_len, tc.batch_size, shuffle=False,
    )
    return tokenizer_path, train_loader, eval_loader


def _run_training(
    model,
    mc,
    tc,
    device,
    train_loader,
    eval_loader,
    optimizer,
    scaler,
    start_step=0,
    best_eval=float("inf"),
    losses=None,
):
    model.train()
    step = start_step
    losses = list(losses or [])
    t0 = time.time()

    print(f"\nTraining for {tc.max_steps} steps...")
    print(f"{'Step':>6} | {'LR':>10} | {'Train':>10} | {'Eval':>10} | {'Time':>8}")
    print("-" * 56)

    while step < tc.max_steps:
        for x, y in train_loader:
            if step >= tc.max_steps:
                break

            x, y = x.to(device), y.to(device)
            lr = get_lr(step, tc)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

            if step % 100 == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])
                elapsed = time.time() - t0
                print(f"{step:6d} | {lr:10.6f} | {avg:10.4f} | {'--':>10} | {elapsed:7.1f}s")

            if step > 0 and step % tc.eval_interval == 0:
                el = evaluate(model, eval_loader, device)
                avg_train = sum(losses[-tc.eval_interval:]) / min(len(losses), tc.eval_interval)
                elapsed = time.time() - t0
                print(f"{step:6d} | {lr:10.6f} | {avg_train:10.4f} | {el:10.4f} | {elapsed:7.1f}s")

                if el < best_eval:
                    best_eval = el
                    _save_training_checkpoint(
                        os.path.join(tc.output_dir, "best_model.pt"),
                        step=step,
                        model=model,
                        mc=mc,
                        tc=tc,
                        optimizer=optimizer,
                        scaler=scaler,
                        best_eval=best_eval,
                        eval_loss=el,
                        device=device,
                    )
                    print(f"  -> Best model (eval={el:.4f})")

            if step > 0 and step % tc.save_interval == 0:
                _save_training_checkpoint(
                    os.path.join(tc.output_dir, f"step_{step}.pt"),
                    step=step,
                    model=model,
                    mc=mc,
                    tc=tc,
                    optimizer=optimizer,
                    scaler=scaler,
                    best_eval=best_eval,
                    train_losses=losses,
                    device=device,
                )

            step += 1

    _save_training_checkpoint(
        os.path.join(tc.output_dir, "final_model.pt"),
        step=step,
        model=model,
        mc=mc,
        tc=tc,
        optimizer=optimizer,
        scaler=scaler,
        best_eval=best_eval,
        train_losses=losses,
        device=device,
    )

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.0f}s, best eval: {best_eval:.4f}")


def _find_resume_checkpoint(output_dir):
    candidates = []
    if not os.path.isdir(output_dir):
        return None

    for name in os.listdir(output_dir):
        if name.startswith("step_") and name.endswith(".pt"):
            try:
                step = int(name[len("step_"):-3])
            except ValueError:
                continue
            candidates.append((step, os.path.join(output_dir, name)))

    final_path = os.path.join(output_dir, "final_model.pt")
    if os.path.exists(final_path):
        try:
            ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
            candidates.append((int(ckpt.get("step", -1)), final_path))
        except Exception:
            pass

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def train():
    mc = CatConfig()
    tc = TrainConfig()
    device = get_device(tc)
    torch.manual_seed(tc.seed)
    random.seed(tc.seed)

    print(f"Device: {device}")

    model = CatLM(mc).to(device)
    print(model.param_summary())

    _, train_loader, eval_loader = _build_loaders(tc, mc)
    print(f"Train: {len(train_loader.dataset):,}, Eval: {len(eval_loader.dataset):,}")

    optimizer = _make_optimizer(model, tc)
    scaler = _make_scaler(device)

    _write_run_config(tc, mc)
    _run_training(
        model=model,
        mc=mc,
        tc=tc,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scaler=scaler,
    )


def resume_train(checkpoint_path=None):
    tc = TrainConfig()
    device = get_device(tc)

    checkpoint_path = checkpoint_path or _find_resume_checkpoint(tc.output_dir)
    if not checkpoint_path:
        raise FileNotFoundError(
            f"No resumable checkpoint found in {tc.output_dir}. "
            "Run `python -m catlm train` first."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"{checkpoint_path} is not a training checkpoint with model_state_dict.")

    if "config" in ckpt:
        mc = _cat_config_from_dict(ckpt["config"])
    else:
        mc = CatConfig()

    model = CatLM(mc).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer = _make_optimizer(model, tc)
    scaler = _make_scaler(device)

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        print("Warning: optimizer state not found; resuming from weights only.")

    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_step = int(ckpt.get("next_step", ckpt.get("step", 0)))
    best_eval = float(ckpt.get("best_eval", ckpt.get("eval_loss", float("inf"))))
    losses = ckpt.get("train_losses", [])
    _restore_rng_state(ckpt, device)

    print(f"Device: {device}")
    print(f"Resuming from: {checkpoint_path}")
    print(f"Resume step: {start_step}")
    print(f"Target max steps: {tc.max_steps}")

    if start_step >= tc.max_steps:
        raise ValueError(
            f"Checkpoint step {start_step} is already >= configured max_steps {tc.max_steps}. "
            "Increase max_steps in catlm/config.py first."
        )

    print(model.param_summary())

    _, train_loader, eval_loader = _build_loaders(tc, mc)
    print(f"Train: {len(train_loader.dataset):,}, Eval: {len(eval_loader.dataset):,}")

    _write_run_config(tc, mc)
    _run_training(
        model=model,
        mc=mc,
        tc=tc,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scaler=scaler,
        start_step=start_step,
        best_eval=best_eval,
        losses=losses,
    )


if __name__ == "__main__":
    train()
