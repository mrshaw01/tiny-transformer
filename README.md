# tiny-transformer

Scratch-only decoder-only language model training demo that runs in **<10 minutes** on **1×A100**.

- Both model implementations are vendored so you can modify internals locally:
  - `tiny_transformer/models/qwen3/`
  - `tiny_transformer/models/qwen3_next/`
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
python -m tiny_transformer.train --config configs/qwen3_demo.json --data_dir data --out_dir runs_qwen3 --steps 2000 --bf16
```

To train Qwen3-Next instead, use:

```bash
python -m tiny_transformer.train --config configs/qwen3_next_demo.json --data_dir data --out_dir runs_qwen3_next --steps 2000 --bf16
```

Better convergence (still typically <10 minutes on 1×A100):

```bash
python -m tiny_transformer.train --config configs/qwen3_demo.json --data_dir data --out_dir runs_qwen3 --steps 5000 --bf16 --eval_steps 500 --save_steps 500 --early_stopping_patience 2 --max_train_minutes 9.5
```

Qwen3-Next variant:

```bash
python -m tiny_transformer.train --config configs/qwen3_next_demo.json --data_dir data --out_dir runs_qwen3_next --steps 5000 --bf16 --eval_steps 500 --save_steps 500 --early_stopping_patience 2 --max_train_minutes 9.5
```

Outputs:

- `<out_dir>/last`: final checkpoint + `run_info.json`
- `<out_dir>/best`: best-by-`eval_loss` checkpoint (only when eval is enabled via `--eval_steps`)

## Sample

```bash
python -m tiny_transformer.sample --ckpt_dir runs_qwen3/best --prompt "Once upon a time" --max_new_tokens 200
```

Qwen3-Next variant:

```bash
python -m tiny_transformer.sample --ckpt_dir runs_qwen3_next/best --prompt "Once upon a time" --max_new_tokens 200
```

## What to edit

- Model configs:
  - `configs/qwen3_demo.json`
  - `configs/qwen3_next_demo.json`
- Model code (vendored):
  - `tiny_transformer/models/qwen3/`
  - `tiny_transformer/models/qwen3_next/`
- Training loop / stopping / checkpointing: `tiny_transformer/train.py`
- Dataset packing: `tiny_transformer/prepare_dataset.py`

More details: `DESIGN.md`.

Math deep dives:

- Qwen3: `TRAINING_MATH_QWEN3.md`
- Qwen3-Next: `TRAINING_MATH_QWEN3_NEXT.md`
