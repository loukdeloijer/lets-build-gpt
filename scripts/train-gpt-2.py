from dataclasses import dataclass
from typing import Optional, Tuple

import os
import numpy as np
import inspect
import tiktoken
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import time

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed import get_rank, get_world_size


@dataclass
class GPT2Config:
    vocab_size: int = 50257  # size of vocabulary
    block_size: int = 1024
    n_layer: int = 12  # number of transformer blocks
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.NANOGPT_SCALE_INIT = 1 # weird way to do this, but we follow Karpathy

        self.n_head: int = config.n_head
        self.n_embd: int = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # type: ignore
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # FlashAttention

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh') #outdated but use for compatibility with GPT-2
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class GPT(nn.Module):


    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # positional
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),  # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # we want to share the weights between the token embedding and the output head
        # as per the paper
        # 30% of the parameters are saved with this trick!
        self.transformer.wte.weight = self.lm_head.weight # type: ignore

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std += (2 + self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        # idx of shape (B, T)

        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."

        # forward the token and position embeddings
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd) # type: ignore

        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd) # type: ignore

        x = tok_emb + pos_emb  # (B, T, n_embd)
        for block in self.transformer.h: # type: ignore

            x = block(x)

        x = self.transformer.ln_f(x)  # (B, T, n_embd) # type: ignore
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None
        return logits, loss



    @classmethod
    def from_pretrained(cls, model_name: str):

        assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

        from transformers import GPT2LMHeadModel
        print(f"Loading {model_name} from Huggingface...")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_name]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # create a from-scratch model and load state dict
        config = GPT2Config(**config_args)
        model = GPT(config)


        sd = model.state_dict()
        sd_keys = set(sd.keys())
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # ignore attn.bias

        model_hf = GPT2LMHeadModel.from_pretrained(model_name)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()

        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f"key length mismatch {len(sd_keys)} vs {len(sd_keys_hf)}, missing keys: {set(sd_keys) - set(sd_keys_hf)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd[k].shape[::-1] == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy
                assert sd[k].shape == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        print(f"Loaded {model_name} from Huggingface...")
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

num_return_sequences=5
max_length=30

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# dataloader

def load_tokens(filename):
     npt =  np.load(filename)
     ptt = torch.tensor(npt, dtype=torch.long)
     return ptt

class DataloaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ['train', 'val']

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(shards) > 0, "No shards found for split"

        if master_process:
            print(f"Found {len(shards)} shards for split {split}")

        self.current_shard = 0
        self.tokens = load_tokens(shards[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T

        batch = self.tokens[self.current_position : self.current_position + self.B * self.T + 1]

        x = (batch[:-1]).view(B, T)
        y = (batch[1:]).view(B, T)

        # advance the position in the tensor

        self.current_position += self.B * self.T * self.num_processes

        # if loading the next batch would go out of bounds, reset current position
        if self.current_position + (self.B * self.T * self.num_processes) >= len(self.tokens):
            self.current_shard += 1
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x, y


#set up DDP
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(ddp_local_rank)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    master_process = True

#reproducability
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B = 16 # can increase to 64?
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batchsize is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print("total desired batch size:", total_batch_size)
    print("gradient accumulation steps:", grad_accum_steps)

train_loader = DataloaderLite(B=16, T=1024, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")

torch.set_float32_matmul_precision('high')

# model
model = GPT(GPT2Config(vocab_size=50304)) #nice divisible number
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model


# learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # 373 million tokens / 524288 tokens per batch
max_steps = 19073 # 10B tokens / 524288 tokens per batch

def get_learning_rate(it):
    # linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps


    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 +math.cos(math.pi + decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# optimizer
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e4, betas=(0.9, 0.95), device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) #ugly, but no context manager

        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient norm clipping to prevent normal from being shocked
    lr = get_learning_rate(step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / (t1 - t0)
    if master_process:
        print(f"step {step+1:4d}| loss {loss_accum:.6f}| LR {lr}| norm: {norm:.4f} | time {dt:.4f} ms | tokens per sec {tokens_per_sec:.2f}")

if ddp:
    dist.destroy_process_group()

# skip sampling for now
# print("Model successfully created")

# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) #(8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  #(5,8)
# x = tokens.to(device)

# torch.manual_seed(42)
# while x.size(1) < max_length:

#     with torch.no_grad():
#         logits = model(x)  # (B, T, C)
#         logits = logits[:, -1, :]  # becomes (B, C)
#         probs = F.softmax(logits, dim=-1)  # (B, C)
#         topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)

#         ix= torch.multinomial(topk_probs, num_samples=1)

#         xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
#         x = torch.cat((x, xcol), dim=1)  # (B, T+1)

# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(f">, {decoded}")