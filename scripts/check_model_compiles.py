"""Smoke test that ensures the GPT model compiles without launching training."""

from importlib import util
from pathlib import Path

import torch


def load_model_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "train-gpt-2.py"
    spec = util.spec_from_file_location("train_gpt_2", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to locate train-gpt-2.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        raise RuntimeError("torch.compile is unavailable in this environment")

    module = load_model_module()
    GPT2Config = module.GPT2Config
    GPT = module.GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPT2Config(vocab_size=50304)
    model = GPT(config).to(device)
    compiled_model = compile_fn(model)

    batch_size = 2
    seq_len = 16
    dummy = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    with torch.no_grad():
        logits, loss = compiled_model(dummy)

    assert loss is None
    expected_shape = (batch_size, seq_len, config.vocab_size)
    if logits.shape != expected_shape:
        raise AssertionError(f"Unexpected logits shape {logits.shape}, expected {expected_shape}")

    print(f"Model compiled successfully on {device} with output shape {logits.shape}.")

if __name__ == "__main__":
    main()
