from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer


def load_tokenizer(path: str):
    try:
        return AutoTokenizer.from_pretrained(path, use_fast=True, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(path, use_fast=True)


def _load_qwen3_model_class():
    try:
        from transformers import Qwen3ForCausalLM  # type: ignore

        return Qwen3ForCausalLM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Qwen3 model class not found. Install transformers>=4.51.0.") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True, help="Directory saved by training (runs/last)")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.ckpt_dir)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None and tokenizer.pad_token_id is not None:
        tokenizer.bos_token = tokenizer.pad_token
    Qwen3ForCausalLM = _load_qwen3_model_class()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[device] cuda:0 name={torch.cuda.get_device_name(0)} bf16_supported={torch.cuda.is_bf16_supported()}")
    else:
        print("[device] cpu")
    model = Qwen3ForCausalLM.from_pretrained(args.ckpt_dir, dtype=torch.bfloat16 if device.type == "cuda" else None)
    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=args.temp,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
