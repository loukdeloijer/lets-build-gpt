"""Utility script for preparing FineWeb-Edu shards."""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class FineWebConfig:
    """Configuration for FineWeb shard preparation."""

    output_dir: Path = Path("edu_fineweb10B")
    remote_name: str = "sample-10BT"
    shard_size: int = int(1e8)
    tokenizer_name: str = "gpt2"
    chunksize: int = 16


_ENCODING: tiktoken.Encoding | None = None
_EOT_TOKEN: int | None = None


def _worker_init(tokenizer_name: str) -> None:
    """Initialise tokenizer state in worker processes."""

    global _ENCODING, _EOT_TOKEN
    _ENCODING = tiktoken.get_encoding(tokenizer_name)
    _EOT_TOKEN = _ENCODING._special_tokens["<|endoftext|>"]


def tokenize_document(doc: Dict[str, str]) -> np.ndarray:
    """Tokenise a single document into uint16 token ids."""

    if _ENCODING is None or _EOT_TOKEN is None:
        raise RuntimeError("Tokenizer has not been initialised in the worker process.")
    tokens = [_EOT_TOKEN]
    tokens.extend(_ENCODING.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.int64)
    if not ((0 <= tokens_np).all() and (tokens_np < 2**16).all()):
        raise ValueError("Token dictionary too large for uint16 storage.")
    return tokens_np.astype(np.uint16)


def write_shard(path: Path, tokens: np.ndarray) -> None:
    """Persist a shard of tokens to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tokens)


def _process_documents(documents: Iterable[np.ndarray], config: FineWebConfig) -> None:
    """Stream tokenised documents into shards of fixed size."""

    shard_index = 0
    shard_buffer = np.empty((config.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar: tqdm | None = None

    for tokens in documents:
        if token_count + len(tokens) < config.shard_size:
            shard_buffer[token_count : token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=config.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = config.output_dir / f"edufineweb_{split}_{shard_index:06d}"
            remainder = config.shard_size - token_count
            if progress_bar is None:
                progress_bar = tqdm(total=config.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(remainder)
            shard_buffer[token_count : token_count + remainder] = tokens[:remainder]
            write_shard(filename, shard_buffer)
            shard_index += 1
            progress_bar = None
            shard_buffer[: len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = config.output_dir / f"edufineweb_{split}_{shard_index:06d}"
        write_shard(filename, shard_buffer[:token_count])


def prepare_shards(config: FineWebConfig) -> None:
    """Download the dataset and prepare token shards."""

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=config.remote_name, split="train")
    num_procs = max(1, mp.cpu_count() // 2)
    with mp.Pool(num_procs, initializer=_worker_init, initargs=(config.tokenizer_name,)) as pool:
        iterator = pool.imap(tokenize_document, dataset, chunksize=config.chunksize)
        _process_documents(iterator, config)


if __name__ == "__main__":
    prepare_shards(FineWebConfig())
