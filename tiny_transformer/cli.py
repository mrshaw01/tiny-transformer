from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tiny-transformer")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub_prepare = sub.add_parser("prepare-data", help="Prepare packed Wikipedia-intros dataset")
    sub_prepare.add_argument("--out_dir", type=str, required=True)
    sub_prepare.add_argument("--seq_len", type=int, default=512)
    sub_prepare.add_argument("--max_bytes", type=int, default=100_000_000)
    sub_prepare.add_argument("--val_ratio", type=float, default=0.01)

    sub_train = sub.add_parser("train", help="Train Qwen3-style causal LM (scratch)")
    sub_train.add_argument("--config", type=str, required=True)
    sub_train.add_argument("--data_dir", type=str, required=True)
    sub_train.add_argument("--out_dir", type=str, default="runs")
    sub_train.add_argument("--steps", type=int, default=2000)
    sub_train.add_argument("--seed", type=int, default=1337)
    sub_train.add_argument("--lr", type=float, default=3e-4)
    sub_train.add_argument("--warmup_steps", type=int, default=200)
    sub_train.add_argument("--weight_decay", type=float, default=0.1)
    sub_train.add_argument("--micro_batch_size", type=int, default=32)
    sub_train.add_argument("--grad_accum", type=int, default=1)
    sub_train.add_argument("--bf16", action="store_true")
    sub_train.add_argument("--log_steps", type=int, default=50)
    sub_train.add_argument("--eval_steps", type=int, default=500)
    sub_train.add_argument("--save_steps", type=int, default=500)
    sub_train.add_argument("--sample_steps", type=int, default=200)
    sub_train.add_argument("--sample_prompt", type=str, default="Once upon a time")
    sub_train.add_argument("--sample_max_new_tokens", type=int, default=120)
    sub_train.add_argument("--sample_temp", type=float, default=0.9)
    sub_train.add_argument("--sample_top_p", type=float, default=0.95)
    sub_train.add_argument("--early_stopping_patience", type=int, default=0)
    sub_train.add_argument("--early_stopping_threshold", type=float, default=0.0)
    sub_train.add_argument("--max_train_minutes", type=float, default=0.0)

    sub_sample = sub.add_parser("sample", help="Sample text from a checkpoint directory")
    sub_sample.add_argument("--ckpt_dir", type=str, required=True)
    sub_sample.add_argument("--prompt", type=str, required=True)
    sub_sample.add_argument("--max_new_tokens", type=int, default=200)
    sub_sample.add_argument("--temp", type=float, default=0.9)
    sub_sample.add_argument("--top_p", type=float, default=0.95)
    sub_sample.add_argument("--seed", type=int, default=1234)

    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "prepare-data":
        from tiny_transformer import prepare_dataset

        prepare_dataset.main([
            "--out_dir",
            args.out_dir,
            "--seq_len",
            str(args.seq_len),
            "--max_bytes",
            str(args.max_bytes),
            "--val_ratio",
            str(args.val_ratio),
        ])
        return

    if args.cmd == "train":
        from tiny_transformer import train

        argv2 = [
            "--config", args.config, "--data_dir", args.data_dir, "--out_dir", args.out_dir, "--steps",
            str(args.steps)
        ]
        argv2 += ["--seed", str(args.seed), "--lr", str(args.lr), "--warmup_steps", str(args.warmup_steps)]
        argv2 += ["--weight_decay", str(args.weight_decay), "--micro_batch_size", str(args.micro_batch_size)]
        argv2 += ["--grad_accum", str(args.grad_accum), "--log_steps", str(args.log_steps)]
        argv2 += ["--eval_steps", str(args.eval_steps), "--save_steps", str(args.save_steps)]
        argv2 += ["--sample_steps", str(args.sample_steps), "--sample_prompt", args.sample_prompt]
        argv2 += ["--sample_max_new_tokens", str(args.sample_max_new_tokens)]
        argv2 += ["--sample_temp", str(args.sample_temp), "--sample_top_p", str(args.sample_top_p)]
        argv2 += ["--early_stopping_patience", str(args.early_stopping_patience)]
        argv2 += ["--early_stopping_threshold", str(args.early_stopping_threshold)]
        argv2 += ["--max_train_minutes", str(args.max_train_minutes)]
        if args.bf16:
            argv2.append("--bf16")
        train.main(argv2)
        return

    if args.cmd == "sample":
        from tiny_transformer import sample

        sample.main([
            "--ckpt_dir",
            args.ckpt_dir,
            "--prompt",
            args.prompt,
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--temp",
            str(args.temp),
            "--top_p",
            str(args.top_p),
            "--seed",
            str(args.seed),
        ])
        return

    raise AssertionError(f"Unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    main()
