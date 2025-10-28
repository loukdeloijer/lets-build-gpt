"""Utilities for downloading and evaluating the HellaSwag dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Tuple

import tiktoken
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from datasets import load_dataset


DATA_CACHE_DIR = Path(__file__).resolve().parent / "hellaswag"
enc = tiktoken.get_encoding("gpt2")


def download(split: str) -> Path:
    """Download the requested split and return the path to the cached file."""

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = DATA_CACHE_DIR / f"hellaswag_{split}.jsonl"
    if not filename.exists():
        print(f"Downloading HellaSwag {split} split from Hugging Face...")
        split_name = "validation" if split == "val" else split
        dataset = load_dataset("Rowan/hellaswag", split=split_name)
        with filename.open("w") as handle:
            for item in dataset:
                converted_item: Dict[str, object] = {
                    "ind": item["ind"],
                    "activity_label": item["activity_label"],
                    "ctx_a": item["ctx_a"],
                    "ctx_b": item["ctx_b"],
                    "ctx": item["ctx"],
                    "split": split,
                    "split_type": item["split_type"],
                    "label": int(item["label"]),
                    "endings": item["endings"],
                    "source_id": item["source_id"],
                }
                handle.write(json.dumps(converted_item) + "\n")
        print(f"Saved to {filename}")
    return filename


def render_example(example: Dict[str, object]) -> Tuple[Dict[str, object], torch.Tensor, torch.Tensor, int]:
    """Render a raw example into tensors expected by the evaluation code."""

    ctx = str(example["ctx"])
    label = int(example["label"])
    endings: Iterable[str] = example["endings"]  # type: ignore[assignment]

    ending_tokens: List[List[int]] = []
    data: Dict[str, object] = {"label": label, "ctx_tokens": None, "ending_tokens": ending_tokens}
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens

    tok_rows = []
    mask_rows = []
    for ending in endings:
        end_tokens = enc.encode(" " + ending)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        ending_tokens.append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for idx, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[idx, : len(tok_row)] = torch.tensor(tok_row, dtype=torch.long)
        mask[idx, : len(mask_row)] = torch.tensor(mask_row, dtype=torch.long)

    return data, tokens, mask, label


def iterate_examples(split: str) -> Generator[Dict[str, object], None, None]:
    """Yield preprocessed examples for the specified split."""

    filename = download(split)
    with filename.open("r") as handle:
        for line in handle:
            yield json.loads(line)


@torch.no_grad()
def evaluate(model_type: str, device: str) -> None:
    """Evaluate a pretrained model on HellaSwag."""

    torch.set_float32_matmul_precision("high")
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        logits = model(tokens).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(tokens.size(0), -1)

        shift_mask = mask[..., 1:].contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        pred = int(sum_loss.argmin().item())
        pred_norm = int(avg_loss.argmin().item())

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print("Endings:")
            for idx, ending in enumerate(example["endings"]):
                print(f"{idx} (loss: {avg_loss[idx].item():.4f}) {ending}")
            print(f"predicted: {pred_norm}, actual: {label}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
