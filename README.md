# tiny-transformer

Scratch-only, Qwen3-style decoder-only training demo that runs in **<10 minutes** on **1×A100**.

Qwen3 model code is vendored into `src/models/qwen3/` so you can modify the architecture locally.

## Setup (conda)

```bash
conda env create -f environment.yml
conda activate tiny-transformer
```

## Pre-commit

```bash
conda env update -f environment.yml --prune
pre-commit install
pre-commit run --all-files
```

## Prepare dataset (Wikipedia intros)

```bash
python scripts/prepare_dataset.py --out_dir data --seq_len 512 --max_bytes 100000000
```

## Train (scratch)

```bash
python -m src.train --config configs/qwen3_demo.json --data_dir data --out_dir runs --steps 2000 --bf16
```

For slightly better convergence (still typically <10 minutes on 1×A100), try:

```bash
python -m src.train --config configs/qwen3_demo.json --data_dir data --out_dir runs --steps 5000 --bf16 --eval_steps 500 --save_steps 500 --early_stopping_patience 2 --max_train_minutes 9.5
```

## Sample

```bash
python -m src.sample --ckpt_dir runs/best --prompt "Once upon a time" --max_new_tokens 200
```

## Notes

- Training defaults are tuned for speed and memory with the large Qwen vocab (`~151k`); override with `--micro_batch_size` / `--grad_accum` if needed.
- If you see `huggingface/tokenizers` fork/parallelism warnings, set `TOKENIZERS_PARALLELISM=false` (the training script does this by default).
- `runs/last/run_info.json` contains the exact args, dataset meta, and library versions for reproducibility.
- Checkpoints: `runs/last` is the final weights; `runs/best` is the best-by-`eval_loss` checkpoint (recommended for sampling).
- Qwen-style stopping: run for a fixed `--steps`, evaluate every `--eval_steps`, save the best checkpoint by `eval_loss` to `runs/best`, optionally early stop (`--early_stopping_patience`), and optionally cap wall time (`--max_train_minutes`).
