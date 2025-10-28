"""Model definitions for GPT-2 training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPT2Config:
    """Configuration for the GPT-2 model."""

    vocab_size: int = 50304
    block_size: int = 1_024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads.")

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.NANOGPT_SCALE_INIT = 1  # match NanoGPT initialisation behaviour

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        bias = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", bias.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(batch_size, time, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(batch_size, time, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(batch_size, time, self.n_head, channels // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, time, channels)
        return self.c_proj(y)


class MLP(nn.Module):
    """Feed-forward block used inside the transformer."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):
    """Transformer block containing attention and MLP sublayers."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """GPT-2 language model."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight  # type: ignore[assignment]
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std += (2 + self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, time = idx.size()
        if time > self.config.block_size:
            raise ValueError("Sequence length exceeds block size.")

        pos = torch.arange(time, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer["wpe"](pos)
        tok_emb = self.transformer["wte"](idx)
        x = tok_emb + pos_emb
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_name: str) -> "GPT":
        valid_names = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        if model_name not in valid_names:
            raise ValueError(f"Unknown model name '{model_name}'.")

        from transformers import GPT2LMHeadModel

        print(f"Loading {model_name} from Hugging Face...")
        config_args: Dict[str, int] = {
            "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},
            "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1_024},
            "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1_280},
            "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1_600},
        }[model_name]
        config_args["vocab_size"] = 50_304
        config_args["block_size"] = 1_024

        config = GPT2Config(**config_args)
        model = cls(config)

        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_name)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith((".attn.masked_bias", ".attn.bias"))]
        transposed = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}

        if len(sd_keys) != len(sd_keys_hf):
            missing = set(sd_keys) - set(sd_keys_hf)
            raise RuntimeError(f"Key length mismatch when loading pretrained weights. Missing keys: {missing}.")

        for key in sd_keys_hf:
            if any(key.endswith(name) for name in transposed):
                if sd[key].shape[::-1] != sd_hf[key].shape:
                    raise RuntimeError("Shape mismatch for transposed weight copy.")
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].t())
            else:
                if sd[key].shape != sd_hf[key].shape:
                    raise RuntimeError("Shape mismatch during pretrained weight loading.")
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key])

        print(f"Loaded {model_name} from Hugging Face.")
        return model

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        param_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}
        decay_params = [param for param in param_dict.values() if param.dim() >= 2]
        nodecay_params = [param for param in param_dict.values() if param.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        print(
            "num decayed parameter tensors: "
            f"{len(decay_params)}, with {sum(param.numel() for param in decay_params):,} parameters"
        )
        print(
            "num non-decayed parameter tensors: "
            f"{len(nodecay_params)}, with {sum(param.numel() for param in nodecay_params):,} parameters"
        )
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args: Dict[str, bool] = {"fused": True} if use_fused else {}
        print(f"using fused AdamW: {use_fused}")
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
