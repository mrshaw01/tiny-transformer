from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import List

os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def load_tokenizer(name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--max_bytes", type=int, default=100_000_000, help="Max raw text bytes to consume for training")
    p.add_argument("--val_ratio", type=float, default=0.01)
    return p.parse_args()


_WS_RE = re.compile(r"[ \t]+")


def wikipedia_intro(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return ""
    first = text.split("\n\n", 1)[0].strip()
    first = _WS_RE.sub(" ", first)
    first = re.sub(r"\n+", "\n", first).strip()
    return first


def stable_float01(s: str) -> float:
    h = hashlib.sha1(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def write_packed_bin(
    *,
    train_out_path: Path,
    val_out_path: Path,
    tokenizer,
    stream,
    seq_len: int,
    max_bytes: int,
    val_ratio: float,
) -> tuple[int, int, int, int, int]:
    train_out_path.parent.mkdir(parents=True, exist_ok=True)

    train_tokens = 0
    val_tokens = 0
    train_sequences = 0
    val_sequences = 0

    consumed_bytes = 0
    train_buffer: List[int] = []
    val_buffer: List[int] = []

    pbar = tqdm(total=max_bytes, desc="packing:wikipedia_intros", unit="B", unit_scale=True)
    try:
        with train_out_path.open("wb") as f_train, val_out_path.open("wb") as f_val:
            eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
            flush_train_tokens_threshold = seq_len * 64
            flush_val_tokens_threshold = seq_len * 16

            def flush_buffer(buffer: List[int], *, f, is_train: bool) -> None:
                nonlocal train_tokens, train_sequences, val_tokens, val_sequences
                n_seq = len(buffer) // seq_len
                if n_seq <= 0:
                    return
                take = n_seq * seq_len
                arr = np.asarray(buffer[:take], dtype=np.uint32)
                arr.tofile(f)
                del buffer[:take]
                if is_train:
                    train_tokens += take
                    train_sequences += n_seq
                else:
                    val_tokens += take
                    val_sequences += n_seq

            batch_size = 256
            batch_texts: List[str] = []
            batch_is_val: List[bool] = []

            def process_batch() -> None:
                if not batch_texts:
                    return
                enc = tokenizer(
                    batch_texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                for ids, is_val in zip(enc["input_ids"], batch_is_val):
                    if eos_id is not None:
                        ids.append(eos_id)
                    if is_val:
                        val_buffer.extend(ids)
                    else:
                        train_buffer.extend(ids)

                if len(train_buffer) >= flush_train_tokens_threshold:
                    flush_buffer(train_buffer, f=f_train, is_train=True)
                if len(val_buffer) >= flush_val_tokens_threshold:
                    flush_buffer(val_buffer, f=f_val, is_train=False)

                batch_texts.clear()
                batch_is_val.clear()

            for ex in stream:
                title = (ex.get("title") or "").strip()
                if not title:
                    continue

                intro = wikipedia_intro(ex.get("text") or "")
                if len(intro) < 200:
                    continue

                byte_len = len(intro.encode("utf-8"))
                if consumed_bytes + byte_len > max_bytes:
                    break

                consumed_bytes += byte_len
                pbar.update(byte_len)

                batch_texts.append(intro)
                batch_is_val.append(stable_float01(title) < val_ratio)
                if len(batch_texts) >= batch_size:
                    process_batch()

                if (train_sequences + val_sequences) > 0 and (train_sequences + val_sequences) % 2000 == 0:
                    pbar.set_postfix(
                        train_seqs=train_sequences,
                        val_seqs=val_sequences,
                        train_tok=train_tokens,
                        val_tok=val_tokens,
                    )

            process_batch()
            flush_buffer(train_buffer, f=f_train, is_train=True)
            flush_buffer(val_buffer, f=f_val, is_train=False)
    finally:
        pbar.close()

    return train_tokens, train_sequences, val_tokens, val_sequences, consumed_bytes


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer("Qwen/Qwen3-0.6B")
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer missing eos_token_id")
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer missing pad_token_id")
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.pad_token

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    train_tokens, train_seqs, val_tokens, val_seqs, consumed_bytes = write_packed_bin(
        train_out_path=out_dir / "train.bin",
        val_out_path=out_dir / "val.bin",
        tokenizer=tokenizer,
        stream=ds,
        seq_len=args.seq_len,
        max_bytes=args.max_bytes,
        val_ratio=args.val_ratio,
    )

    meta = {
        "seq_len": args.seq_len,
        "dtype": "uint32",
        "vocab_size": int(len(tokenizer)),
        "tokenizer_name": "Qwen/Qwen3-0.6B",
        "eos_token_id": int(tokenizer.eos_token_id),
        "bos_token_id": int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else None,
        "train_tokens": int(train_tokens),
        "train_sequences": int(train_seqs),
        "val_tokens": int(val_tokens),
        "val_sequences": int(val_seqs),
        "val_ratio": float(args.val_ratio),
        "max_bytes": int(args.max_bytes),
        "consumed_bytes": int(consumed_bytes),
        "source": "wikimedia/wikipedia:20231101.en:intros",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote train_seqs={train_seqs} val_seqs={val_seqs} "
          f"train_tokens={train_tokens} val_tokens={val_tokens} consumed_bytes={consumed_bytes}")
    # Work around a rare hard crash on interpreter shutdown observed with some
    # `datasets`/`pyarrow` streaming stacks by skipping Python finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
