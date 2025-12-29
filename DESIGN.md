# tiny-transformer — design notes

This repo is intentionally small and pragmatic: it demonstrates an end-to-end **scratch** training run for a **Qwen3-style** decoder-only model that finishes in **<10 minutes** on **1×A100**, and can generate non-trivial text samples.

## What we train

- The training script builds and trains `Qwen3ForCausalLM` (next-token prediction).
  - Backbone: `Qwen3Model`
  - HF base utilities: `Qwen3PreTrainedModel`
- Code lives in `tiny_transformer/models/qwen3/` and is vendored from `transformers` so you can edit internals.

## Constraints / reality check

- Training a “real” 0.6B model from scratch to general-text quality in <10 minutes is not realistic.
- This repo targets: “pipeline works” + “loss drops fast” + “samples show local coherence / memorization on a small general-text subset”.

## Model configuration

The model is defined by a single JSON config: `configs/qwen3_demo.json`.

Key properties (as implemented):

- Decoder-only Transformer with RoPE + RMSNorm + SwiGLU + GQA (Qwen-family style).
- Context length: `512`
- Tokenizer vocab: Qwen tokenizer (`vocab_size` in config is kept compatible with Qwen reserved ids).

To change model size/shape, edit `configs/qwen3_demo.json` only.

## Tokenizer

We follow Qwen3: use the official tokenizer via HuggingFace:

- `AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")`

Notes:

- `len(tokenizer)` can differ from `config.vocab_size` due to reserved/added tokens; training allows mismatch as long as token ids are `< config.vocab_size`.
- Scripts set `TOKENIZERS_PARALLELISM=false` to avoid common fork/parallelism warnings.

## Dataset: Wikipedia intros (packed)

Data prep script: `tiny_transformer/prepare_dataset.py`

Source:

- `wikimedia/wikipedia`, config `20231101.en`, split `train`, streaming.

Processing:

- Extract first paragraph (“intro”), basic whitespace cleanup, skip very short intros.
- Tokenize in batches, append `eos_token_id`, then pack into fixed-length sequences (`seq_len`, default 512).
- Deterministic train/val split by hashing the article title.

Output files:

- `data/train.bin` and `data/val.bin` (uint32 token ids, packed, memmap-friendly)
- `data/meta.json` (seq_len, vocab_size, eos/bos ids, counts, etc.)

Implementation note:

- The script uses `os._exit(0)` after writing outputs to work around a rare hard crash during interpreter shutdown seen on some `datasets`/`pyarrow` streaming stacks.

## Training loop

Training script: `tiny_transformer/train.py`

Core choices:

- `transformers.Trainer` (single GPU)
- bf16 on A100 (`--bf16`)
- TF32 enabled for matmuls for speed
- AdamW fused optimizer when available (`optim="adamw_torch_fused"`)
- Warmup + cosine LR schedule
- Packed memmap dataset (`tiny_transformer/data/packed_dataset.py`)

Logging / checkpoints / stopping:

- Samples printed every `--sample_steps` (default 200) via a callback.
- If validation is enabled (`--eval_steps`):
  - Best checkpoint by `eval_loss` is saved to `runs/best` on each eval improvement.
  - Optional early stopping: `--early_stopping_patience` (+ optional `--early_stopping_threshold`).
- Optional wall-time cap: `--max_train_minutes` stops training after N minutes.
- Final checkpoint is always saved to `runs/last` along with `runs/last/run_info.json`.

## Repository structure (current)

```
tiny-transformer/
  README.md
  DESIGN.md
  environment.yml
  .pre-commit-config.yaml
  .style.yapf
  .clang-format
  configs/
    qwen3_demo.json
  tiny_transformer/
    __init__.py
    cli.py
    prepare_dataset.py
    train.py
    sample.py
    data/
      packed_dataset.py
    models/
      qwen3/
        configuration_qwen3.py
        modeling_qwen3.py
```

## Minimal commands

- Prepare data:
  - `python -m tiny_transformer.prepare_dataset --out_dir data --seq_len 512 --max_bytes 100000000`
- Train:
  - `python -m tiny_transformer.train --config configs/qwen3_demo.json --data_dir data --out_dir runs --steps 2000 --bf16`
- Sample:
  - `python -m tiny_transformer.sample --ckpt_dir runs/best --prompt "Once upon a time" --max_new_tokens 200`
