# tiny-transformer — design notes

This repo is intentionally small and pragmatic: it demonstrates an end-to-end **scratch** training run for a decoder-only
model that finishes in **<10 minutes** on **1×A100**, and can generate non-trivial text samples.

## What we train

We maintain **both** implementations to compare them under the same training loop + dataset:

- Qwen3: `Qwen3ForCausalLM` (`tiny_transformer/models/qwen3/`)
- Qwen3-Next: `Qwen3NextForCausalLM` (`tiny_transformer/models/qwen3_next/`)

`tiny_transformer/train.py` selects the model by reading `model_type` from the JSON config you pass to `--config`.

## Constraints / reality check

- Training a “real” 0.6B model from scratch to general-text quality in <10 minutes is not realistic.
- This repo targets: “pipeline works” + “loss drops fast” + “samples show local coherence / memorization on a small general-text subset”.

## Model configuration

The model is defined by a JSON config file under `configs/`:

- `configs/qwen3_demo.json` (`model_type: qwen3`)
- `configs/qwen3_next_demo.json` (`model_type: qwen3_next`)

Key properties (shared):

- Decoder-only Transformer with RoPE + RMSNorm + SwiGLU.
- Qwen tokenizer vocab (`~151k`) to match Qwen reserved ids.
- The demo configs intentionally share the same core size knobs (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`) so runs are comparable.

Key properties (Qwen3-Next specific):

- Token mixer can be `full_attention` or `linear_attention` per layer (controlled by `layer_types`).
- Context length: `512`
- Tokenizer vocab: Qwen tokenizer (`vocab_size` in config is kept compatible with Qwen reserved ids).

To change model size/shape, edit the config you are using.

## Tokenizer

We follow Qwen: use the official tokenizer via HuggingFace:

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
  - Best checkpoint by `eval_loss` is saved to `<out_dir>/best` on each eval improvement.
  - Optional early stopping: `--early_stopping_patience` (+ optional `--early_stopping_threshold`).
- Optional wall-time cap: `--max_train_minutes` stops training after N minutes.
- Final checkpoint is always saved to `<out_dir>/last` along with `<out_dir>/last/run_info.json`.
  - In practice, use different `--out_dir` per model when comparing (e.g. `runs_qwen3` vs `runs_qwen3_next`).

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
    qwen3_next_demo.json
  tiny_transformer/
    __init__.py
    prepare_dataset.py
    train.py
    sample.py
    data/
      packed_dataset.py
    models/
      qwen3/
        configuration_qwen3.py
        modeling_qwen3.py
      qwen3_next/
        configuration_qwen3_next.py
        modeling_qwen3_next.py
```

## Minimal commands

- Prepare data:
  - `python -m tiny_transformer.prepare_dataset --out_dir data --seq_len 512 --max_bytes 100000000`
- Train:
  - Qwen3: `python -m tiny_transformer.train --config configs/qwen3_demo.json --data_dir data --out_dir runs_qwen3 --steps 2000 --bf16`
  - Qwen3-Next: `python -m tiny_transformer.train --config configs/qwen3_next_demo.json --data_dir data --out_dir runs_qwen3_next --steps 2000 --bf16`
- Sample:
  - Qwen3: `python -m tiny_transformer.sample --ckpt_dir runs_qwen3/best --prompt "Once upon a time" --max_new_tokens 200`
  - Qwen3-Next: `python -m tiny_transformer.sample --ckpt_dir runs_qwen3_next/best --prompt "Once upon a time" --max_new_tokens 200`
