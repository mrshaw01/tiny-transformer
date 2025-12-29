# Tiny Transformer — Design (Qwen3-style, scratch-only, <10 min)

## Goals

- Build a small, readable repo that can train a **Qwen3-style** decoder-only Transformer **from scratch** on **1×A100**.
- Training run completes in **< 10 minutes** (including periodic text generation).
- Outputs **meaningful text** by using a small, clean dataset and a small model (see “Reality check”).
- Follow the **Qwen3 repo conventions**: HuggingFace Transformers, Qwen tokenizer, Qwen-style architecture choices.

## Environment

Use a **conda** environment for reproducibility.

- Provide an `environment.yml` at repo root.
- Target platform: Linux + CUDA on 1×A100.
- Key deps: `python` (3.10), `pytorch` (CUDA build), `transformers` (4.x), `datasets`, `accelerate`.

## Reality check (important)

- Training a **0.6B** model **from scratch** to produce broadly coherent **general web text** in **<10 minutes** is not realistic.
- What we _can_ do in <10 minutes:
  - Prove the full pipeline works (data → train → sample) and that loss drops quickly.
  - Produce “meaningful” text in the sense of **clear local structure** and **memorization/near-memorization** on a **small curated general-text subset**.
  - Or: produce coherent text on an “easy” distribution (e.g., short stories) — but that would not be “general text”.

This repo design focuses only on the **<10 minute** track:

- A **small Qwen3-style** model trained from scratch on a **small general-text subset**.

## Non-goals (for this phase)

- SOTA benchmarks.
- Multi-GPU / distributed training.
- RLHF / DPO (can be added later).

## Model architecture (Qwen3-style decoder-only)

We mirror modern Qwen-family choices (what the “author does”) while keeping implementation small:

- **RoPE** positional encoding.
- **RMSNorm** (pre-norm).
- **SwiGLU** MLP.
- **Grouped-query attention (GQA)**.
- **Tied** token embedding + LM head.

Design rule: keep architecture defined by a single config file (`configs/qwen3_demo.json`) so model changes are config-only.

## Code layout (current)

- Model code is vendored from HuggingFace Transformers into `tiny_transformer/models/qwen3/` so we can modify it locally.
- Training/sampling imports `Qwen3Config`/`Qwen3ForCausalLM` from `tiny_transformer.models.qwen3` (not from `transformers`).
- We only keep the **causal LM** path (no seq-cls / token-cls / QA heads).

### Concrete target config (only one)

We keep the **Qwen3 tokenizer vocab** (large) but reduce hidden size / depth so the run fits <10 minutes.

**Config `qwen3_demo` (scratch-only, <10 minutes)**

- Tokenizer vocab size: **Qwen3 tokenizer** (large; embedding-heavy)
- Layers: **8**
- Hidden size: **384**
- Attention heads: **6** (head_dim=64)
- KV heads (GQA): **2**
- MLP: **SwiGLU**, intermediate size ≈ **(8/3)×d_model** rounded to a multiple of 64
- Context length: **512**
- RoPE: Qwen-style settings (e.g., `rope_theta` per Qwen config)

## Tokenization

Follow Qwen3: use the **official Qwen3 tokenizer** via HuggingFace:

- `AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")`
- We do **not** train a tokenizer in this repo.
- Note: the tokenizer’s `len(tokenizer)` can be smaller than the model `config.vocab_size` (reserved tokens); we keep `config.vocab_size` and ensure all token IDs fit.

## Dataset (to get visible improvement fast)

You asked for “general text” and to choose what the author does. Practically:

- For <10 min: a **small, clean subset** of general text so the model can overfit enough to generate fluent snippets.

We use exactly one dataset source for the repo:

- **Wikipedia intros**: take the first paragraph of each article from `wikimedia/wikipedia` (English), then take the first ~**100MB** of cleaned text for the <10 minute run.

Data pipeline design:

- Convert text to token IDs with the Qwen3 tokenizer.
- Pack into fixed-length sequences of **512** tokens.
- Store as `data/train.bin` / `data/val.bin` (memmap-friendly), plus `data/meta.json` (seq_len, tokenizer, vocab size).
- Implementation notes (current code):
  - Uses batched tokenization to improve throughput.
  - Uses a hard exit (`os._exit(0)`) after writing outputs to avoid a rare `datasets/pyarrow` shutdown crash on some systems.

## Training loop (single A100, <10 min)

We follow the Qwen3 “Transformers-first” approach:

- Use **Transformers** `Trainer` with `Qwen3ForCausalLM` built from `configs/qwen3_demo.json`.
- bf16 training on A100.
- Use PyTorch SDPA attention via Transformers (FlashAttention path when available).
- Enable TF32 matmul on CUDA for speed.

### Optimizer & schedule (scratch training)

- AdamW (fused when available via `adamw_torch_fused`)
- Warmup **200 steps** + cosine decay
- Grad clip: 1.0
- Weight decay: 0.1 (exclude biases and norm weights)

### Throughput knobs

Fixed for this repo (no alternatives):

- `seq_len=512`
- `micro_batch_size=32`, `grad_accum=1` (keeps the vocab-sized softmax memory manageable)
- Effective tokens/step: `32 * 512 = 16,384`

### Checkpointing and sampling

- Save:
  - `runs/last`: final model + tokenizer + `run_info.json`
  - `runs/best`: best-by-`eval_loss` checkpoint (updated on each eval)
- Generate samples every **200 steps** (configurable) and print to stdout.
- Optional stopping controls:
  - Early stopping: `--early_stopping_patience` / `--early_stopping_threshold`
  - Wall-time cap: `--max_train_minutes`

## Minimal evaluation

- Train loss and a small validation loss.
- Tokens/sec and step time.
- A rolling log of generated samples.

Success criteria for this phase:

- Generated samples show clear local coherence on the Wikipedia-intro subset (expected to be narrow and somewhat memorized).

## Repository structure (current)

Keep the repo tiny, with a clear split between “data prep” and “train/sample”, and a single importable Python package.

```
tiny-transformer/
  DESIGN.md
  README.md
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
      __init__.py
      packed_dataset.py
    models/
      __init__.py
      qwen3/
        __init__.py
        configuration_qwen3.py
        modeling_qwen3.py
```

## CLI (what you run)

Prepare packed token data (Wikipedia intros):

- `python -m tiny_transformer.prepare_dataset --out_dir data --seq_len 512 --max_bytes 100000000`

Train:

- `python -m tiny_transformer.train --config configs/qwen3_demo.json --data_dir data --out_dir runs --steps 2000 --bf16`
- Better convergence (still <10 minutes on 1×A100): `python -m tiny_transformer.train --config configs/qwen3_demo.json --data_dir data --out_dir runs --steps 5000 --bf16 --eval_steps 500 --save_steps 500 --early_stopping_patience 2 --max_train_minutes 9.5`

Sample:

- `python -m tiny_transformer.sample --ckpt_dir runs/best --prompt "Once upon a time" --max_new_tokens 200`

Optional single-entry CLI wrapper:

- `python -m tiny_transformer.cli prepare-data ...`
- `python -m tiny_transformer.cli train ...`
- `python -m tiny_transformer.cli sample ...`
