"""Training entrypoint for GPT-2 fine-tuning."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import tiktoken
import wandb

from hellaswag import iterate_examples, render_example
from model import GPT, GPT2Config


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime configuration for GPT-2 training."""

    project: str = "gpt2-training"
    data_root: Path = field(default_factory=lambda: Path("edu_fineweb10B"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    total_batch_size: int = 524_288
    micro_batch_size: int = 64
    block_size: int = 1_024
    max_steps: int = 19_073
    warmup_steps: int = 715
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    val_interval: int = 100
    val_steps: int = 20
    hellaswag_interval: int = 250
    sample_interval: int = 250
    checkpoint_interval: int = 5_000
    num_return_sequences: int = 4
    sample_length: int = 32
    sample_prompt: str = "Hello, I'm a language model,"
    seed: int = 1_337
    hellaswag_split: str = "val"
    use_compile: bool = True
    model: GPT2Config = field(default_factory=lambda: GPT2Config(vocab_size=50_304))

    def __post_init__(self) -> None:
        if isinstance(self.data_root, str):
            self.data_root = Path(self.data_root)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.model.block_size != self.block_size:
            self.model.block_size = self.block_size

    @property
    def min_lr(self) -> float:
        return self.max_lr * self.min_lr_ratio

    def grad_accum_steps(self, world_size: int) -> int:
        tokens_per_micro_batch = self.micro_batch_size * self.block_size
        if self.total_batch_size % (tokens_per_micro_batch * world_size) != 0:
            raise ValueError(
                "total_batch_size must be divisible by micro_batch_size * block_size * world_size"
            )
        return self.total_batch_size // (tokens_per_micro_batch * world_size)

    def to_wandb_config(self, world_size: int) -> Dict[str, int | float | str]:
        return {
            "batch_size": self.micro_batch_size,
            "sequence_length": self.block_size,
            "total_batch_size": self.total_batch_size,
            "grad_accum_steps": self.grad_accum_steps(world_size),
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "vocab_size": self.model.vocab_size,
            "n_layer": self.model.n_layer,
            "n_head": self.model.n_head,
            "n_embd": self.model.n_embd,
        }


def setup_distributed() -> Tuple[bool, int, int, int, torch.device, bool]:
    """Initialise distributed training and return process metadata."""

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA support.")
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(ddp_local_rank)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        master_process = True

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


def load_tokens(filename: Path) -> torch.Tensor:
    """Load token ids from disk."""

    tokens = np.load(str(filename), allow_pickle=False)
    return torch.tensor(tokens, dtype=torch.long)


class DataloaderLite:
    """Minimalistic dataloader that iterates over pre-tokenised shards."""

    def __init__(
        self,
        batch_size: int,
        block_size: int,
        process_rank: int,
        num_processes: int,
        split: str,
        data_root: Path,
        master_process: bool,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.batch_size = batch_size
        self.block_size = block_size
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.master_process = master_process

        shards = sorted(data_root.glob(f"*{split}*"))
        if not shards:
            raise FileNotFoundError(f"No shards found for split '{split}' in {data_root}.")

        self.shards = shards
        if self.master_process:
            print(f"Found {len(self.shards)} shards for split {split}")

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.process_rank * self.batch_size * self.block_size

    def _advance_shard(self) -> None:
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        if self.current_shard == 0:
            np.random.shuffle(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.process_rank * self.batch_size * self.block_size

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        total_tokens = self.batch_size * self.block_size
        batch = self.tokens[self.current_position : self.current_position + total_tokens + 1]
        if batch.numel() < total_tokens + 1:
            self._advance_shard()
            batch = self.tokens[self.current_position : self.current_position + total_tokens + 1]

        x = batch[:-1].view(self.batch_size, self.block_size)
        y = batch[1:].view(self.batch_size, self.block_size)

        self.current_position += total_tokens * self.num_processes
        if self.current_position + (total_tokens * self.num_processes) >= len(self.tokens):
            self._advance_shard()

        return x, y


def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    """Return the index of the completion with the lowest loss."""

    pred_logits = logits[..., :-1, :].contiguous()
    target_tokens = tokens[..., 1:].contiguous()
    flat_logits = pred_logits.view(-1, pred_logits.size(-1))
    flat_targets = target_tokens.view(-1)
    token_losses = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    token_losses = token_losses.view(tokens.size(0), -1)

    completion_mask = mask[..., 1:].contiguous()
    masked_losses = token_losses * completion_mask
    avg_loss = masked_losses.sum(dim=1) / completion_mask.sum(dim=1)
    return int(avg_loss.argmin().item())


def evaluate_train_loss(loss_accum: torch.Tensor) -> float:
    """Convert accumulated training loss tensor to a float."""

    return float(loss_accum.item())


def evaluate_validation_loss(
    model: nn.Module,
    loader: DataloaderLite,
    device: torch.device,
    val_steps: int,
    ddp: bool,
) -> float:
    """Evaluate validation loss over a fixed number of steps."""

    was_training = model.training
    model.eval()
    total_loss = torch.zeros(1, device=device)
    for _ in range(val_steps):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        if loss is None:
            continue
        total_loss += loss.detach() / val_steps
    if ddp:
        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
    if was_training:
        model.train()
    return float(total_loss.item())


def evaluate_hellaswag(
    model: nn.Module,
    device: torch.device,
    ddp: bool,
    ddp_rank: int,
    ddp_world_size: int,
    split: str,
) -> float:
    """Evaluate accuracy on the HellaSwag benchmark."""

    was_training = model.training
    model.eval()
    num_correct = 0
    num_total = 0
    for index, example in enumerate(iterate_examples(split)):
        if index % ddp_world_size != ddp_rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(tokens)
        pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct += int(pred_norm == label)

    if ddp:
        total_tensor = torch.tensor([num_correct, num_total], dtype=torch.long, device=device)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_correct, num_total = total_tensor.tolist()

    if was_training:
        model.train()

    return num_correct / max(num_total, 1)


def generate_samples(
    model: nn.Module,
    device: torch.device,
    enc: tiktoken.Encoding,
    prompt: str,
    num_sequences: int,
    max_length: int,
    seed_offset: int = 0,
) -> Iterable[str]:
    """Generate samples from the model given a prompt."""

    was_training = model.training
    model.eval()
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + seed_offset)

    generated = tokens
    while generated.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(generated)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        idx = torch.multinomial(topk_probs, 1, generator=sample_rng)
        next_token = torch.gather(topk_indices, -1, idx)
        generated = torch.cat((generated, next_token), dim=1)

    sequences = generated[:, :max_length].tolist()
    if was_training:
        model.train()

    for sequence in sequences:
        yield enc.decode(sequence)


def get_learning_rate(step: int, config: TrainingConfig) -> float:
    """Cosine schedule with linear warmup."""

    if step < config.warmup_steps:
        return config.max_lr * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return config.min_lr
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


def main() -> None:
    config = TrainingConfig()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_distributed()
    grad_accum_steps = config.grad_accum_steps(ddp_world_size)

    if master_process:
        print("total desired batch size:", config.total_batch_size)
        print("gradient accumulation steps:", grad_accum_steps)

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)

    enc = tiktoken.get_encoding("gpt2")

    train_loader = DataloaderLite(
        batch_size=config.micro_batch_size,
        block_size=config.block_size,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="train",
        data_root=config.data_root,
        master_process=master_process,
    )
    val_loader = DataloaderLite(
        batch_size=config.micro_batch_size,
        block_size=config.block_size,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
        data_root=config.data_root,
        master_process=master_process,
    )

    torch.set_float32_matmul_precision("high")

    model = GPT(config.model)
    model.to(device)
    if config.use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.max_lr,
        betas=config.betas,
        device_type=device.type,
    )

    if master_process:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(project=config.project, config=config.to_wandb_config(ddp_world_size))

    for step in range(config.max_steps):
        step_start = time.time()
        last_step = step == config.max_steps - 1

        if step % config.val_interval == 0 or last_step:
            val_loss = evaluate_validation_loss(model, val_loader, device, config.val_steps, ddp)
            if master_process:
                print(f"step {step + 1:4d} | val loss {val_loss:.6f}")
                wandb.log({"val/val_loss": val_loss, "step": step})
                if step > 0 and (step % config.checkpoint_interval == 0 or last_step):
                    checkpoint_path = config.checkpoint_dir / f"model-step-{step}.pth"
                    torch.save(model.state_dict(), checkpoint_path)

        if step % config.hellaswag_interval == 0 or last_step:
            accuracy = evaluate_hellaswag(
                model,
                device,
                ddp,
                ddp_rank,
                ddp_world_size,
                config.hellaswag_split,
            )
            if master_process:
                print(f"HellaSwag accuracy: {accuracy:.4f}")
                wandb.log({"hella_swag_accuracy": accuracy, "step": step})

        if (step % config.sample_interval == 0 and step > 0) or last_step:
            for idx, sample in enumerate(
                generate_samples(
                    model,
                    device,
                    enc,
                    config.sample_prompt,
                    config.num_return_sequences,
                    config.sample_length,
                    seed_offset=ddp_rank,
                )
            ):
                if master_process:
                    print(f"rank {ddp_rank} sample {idx}: {sample}")

        model.train()
        optimizer.zero_grad()
        loss_accum = torch.zeros(1, device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            if loss is None:
                continue
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        lr = get_learning_rate(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.time() - step_start) * 1000
        tokens_processed = (
            train_loader.batch_size * train_loader.block_size * grad_accum_steps * ddp_world_size
        )
        tokens_per_sec = tokens_processed / (elapsed / 1000)
        train_loss = evaluate_train_loss(loss_accum)
        if master_process:
            print(
                f"step {step + 1:4d} | loss {train_loss:.6f} | LR {lr:.6f} | norm {norm:.4f} | "
                f"time {elapsed:.2f} ms | tokens per sec {tokens_per_sec:.2f}"
            )
            wandb.log(
                {
                    "loss": train_loss,
                    "learning_rate": lr,
                    "grad_norm": norm.item() if hasattr(norm, "item") else float(norm),
                    "step_time_ms": elapsed,
                    "tokens_per_sec": tokens_per_sec,
                    "step": step,
                }
            )

    if master_process:
        wandb.finish()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
