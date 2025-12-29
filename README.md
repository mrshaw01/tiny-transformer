# tiny-transformer

Scratch-only, Qwen3-style decoder-only language model training demo that runs in **<10 minutes** on **1×A100**.

- Model code is vendored into `tiny_transformer/models/qwen3/` so you can modify internals locally.
- Data is a small streaming subset of **Wikipedia intros** packed into fixed-length token sequences.

## Setup (conda)

```bash
conda env create -f environment.yml
conda activate tiny-transformer
```

Optional (recommended):

```bash
pre-commit install
```

## Prepare dataset

Creates `data/train.bin`, `data/val.bin`, `data/meta.json`.

```bash
python -m tiny_transformer.prepare_dataset --out_dir data --seq_len 512 --max_bytes 100000000
```

Notes:

- This uses `datasets` streaming and downloads from HuggingFace (needs network access).
- `data/` is in `.gitignore` (do not commit it).

## Train (scratch)

Fast demo run:

```bash
python -m tiny_transformer.train --config configs/qwen3_demo.json --data_dir data --out_dir runs --steps 2000 --bf16
```

Better convergence (still typically <10 minutes on 1×A100):

```bash
python -m tiny_transformer.train --config configs/qwen3_demo.json --data_dir data --out_dir runs --steps 5000 --bf16 --eval_steps 500 --save_steps 500 --early_stopping_patience 2 --max_train_minutes 9.5
```

Outputs:

- `runs/last`: final checkpoint + `run_info.json`
- `runs/best`: best-by-`eval_loss` checkpoint (only when eval is enabled via `--eval_steps`)

## Sample

```bash
python -m tiny_transformer.sample --ckpt_dir runs/best --prompt "Once upon a time" --max_new_tokens 200
```

## What to edit

- Model config: `configs/qwen3_demo.json`
- Model code (vendored): `tiny_transformer/models/qwen3/`
- Training loop / stopping / checkpointing: `tiny_transformer/train.py`
- Dataset packing: `tiny_transformer/prepare_dataset.py`

More details: `DESIGN.md`.
