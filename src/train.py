from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from transformers import AutoTokenizer
from transformers import set_seed
from transformers import Trainer
from transformers import TrainerCallback
from transformers import TrainingArguments

from src.data import default_data_collator
from src.data import PackedDatasetMeta
from src.data import PackedMemmapDataset
from src.models.qwen3 import Qwen3Config
from src.models.qwen3 import Qwen3ForCausalLM


def load_tokenizer(name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to configs/qwen3_demo.json")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing train.bin/val.bin/meta.json")
    p.add_argument("--out_dir", type=str, default="runs", help="Output directory")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--micro_batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--sample_steps", type=int, default=200)
    p.add_argument("--sample_prompt", type=str, default="Once upon a time")
    p.add_argument("--sample_max_new_tokens", type=int, default=120)
    p.add_argument("--sample_temp", type=float, default=0.9)
    p.add_argument("--sample_top_p", type=float, default=0.95)
    p.add_argument("--early_stopping_patience", type=int, default=0)
    p.add_argument("--early_stopping_threshold", type=float, default=0.0)
    p.add_argument("--max_train_minutes", type=float, default=0.0)
    return p.parse_args()


class SampleCallback(TrainerCallback):

    def __init__(
        self,
        *,
        tokenizer,
        every_steps: int,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        self.tokenizer = tokenizer
        self.every_steps = every_steps
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    @torch.inference_mode()
    def on_step_end(self, args, state, control, **kwargs):
        if self.every_steps <= 0:
            return control
        if state.global_step == 0 or state.global_step % self.every_steps != 0:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n[sample step={state.global_step}]\n{text}\n")
        if was_training:
            model.train()
        return control


class MaxTimeCallback(TrainerCallback):

    def __init__(self, max_minutes: float):
        self.max_seconds = float(max_minutes) * 60.0
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not self.max_seconds or self.start_time is None:
            return control
        if time.time() - self.start_time >= self.max_seconds:
            control.should_training_stop = True
        return control


class BestEvalLossCallback(TrainerCallback):

    def __init__(self, out_dir: Path, tokenizer):
        self.out_dir = out_dir
        self.tokenizer = tokenizer
        self.best_loss = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return control
        loss = float(metrics["eval_loss"])
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            best_dir = self.out_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model = kwargs.get("model")
            if model is not None:
                model.save_pretrained(str(best_dir))
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(best_dir))
            print(f"[best] step={state.global_step} eval_loss={loss:.4f} -> {best_dir}")
        return control


class EvalLossEarlyStoppingCallback(TrainerCallback):

    def __init__(self, patience: int, threshold: float):
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.best_loss = None
        self.bad_evals = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.patience <= 0:
            return control
        if not metrics or "eval_loss" not in metrics:
            return control

        loss = float(metrics["eval_loss"])
        if self.best_loss is None:
            self.best_loss = loss
            self.bad_evals = 0
            return control

        improved = (self.best_loss - loss) > self.threshold
        if improved:
            self.best_loss = loss
            self.bad_evals = 0
        else:
            self.bad_evals += 1
            if self.bad_evals >= self.patience:
                print(f"[early_stop] step={state.global_step} eval_loss={loss:.4f} "
                      f"(best={self.best_loss:.4f}) bad_evals={self.bad_evals}/{self.patience}")
                control.should_training_stop = True

        return control


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[device] cuda:0 name={torch.cuda.get_device_name(0)} "
              f"capability={torch.cuda.get_device_capability(0)} "
              f"bf16_supported={torch.cuda.is_bf16_supported()}")
        print(f"[device] cuda_device_count={torch.cuda.device_count()}")
    else:
        print("[device] cpu")

    data_dir = Path(args.data_dir)
    meta = PackedDatasetMeta.load(data_dir / "meta.json")
    train_ds = PackedMemmapDataset(data_dir / "train.bin", meta)
    eval_ds = PackedMemmapDataset(data_dir / "val.bin", meta) if (data_dir / "val.bin").exists() else None
    if eval_ds is not None and args.eval_steps is None:
        raise ValueError("Validation requires --eval_steps")

    cfg_dict = json.loads(Path(args.config).read_text())
    config = Qwen3Config(**cfg_dict)

    tokenizer = load_tokenizer("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer missing pad_token_id")
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.pad_token

    if tokenizer.eos_token_id != meta.eos_token_id:
        raise ValueError(
            f"Tokenizer eos_token_id ({tokenizer.eos_token_id}) != dataset eos_token_id ({meta.eos_token_id})")
    # HF tokenizers expose both `vocab_size` (base vocab) and `len(tokenizer)` (includes added/reserved tokens).
    tokenizer_total_size = len(tokenizer)
    if int(tokenizer_total_size) > int(config.vocab_size):
        raise ValueError(f"Tokenizer size ({tokenizer_total_size}) > config vocab_size ({config.vocab_size})")
    if int(meta.vocab_size) > int(config.vocab_size):
        raise ValueError(f"Dataset vocab_size ({meta.vocab_size}) > config vocab_size ({config.vocab_size})")
    if tokenizer_total_size != int(config.vocab_size):
        print(f"[warn] tokenizer size ({tokenizer_total_size}) != config vocab_size ({config.vocab_size}); "
              "this is OK as long as tokenizer ids are < config.vocab_size.")

    model = Qwen3ForCausalLM(config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "run"

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        max_steps=args.steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16,
        fp16=False,
        tf32=True,
        max_grad_norm=1.0,
        logging_steps=args.log_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        load_best_model_at_end=False,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=min(4,
                                   os.cpu_count() or 1),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        optim="adamw_torch_fused",
    )

    callbacks = [
        SampleCallback(
            tokenizer=tokenizer,
            every_steps=args.sample_steps,
            prompt=args.sample_prompt,
            max_new_tokens=args.sample_max_new_tokens,
            temperature=args.sample_temp,
            top_p=args.sample_top_p,
        )
    ]
    if eval_ds is not None:
        callbacks.append(BestEvalLossCallback(out_dir=out_dir, tokenizer=tokenizer))
    if args.max_train_minutes and args.max_train_minutes > 0:
        callbacks.append(MaxTimeCallback(args.max_train_minutes))
    if eval_ds is not None and args.early_stopping_patience and args.early_stopping_patience > 0:
        callbacks.append(
            EvalLossEarlyStoppingCallback(
                patience=args.early_stopping_patience,
                threshold=args.early_stopping_threshold,
            ))

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )
    # Transformers>=4.57 uses `processing_class` instead of `tokenizer`.
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:  # pragma: no cover
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    last_dir = out_dir / "last"
    last_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(last_dir))
    tokenizer.save_pretrained(str(last_dir))

    run_info = {
        "timestamp": int(time.time()),
        "platform": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "device": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "args": vars(args),
        "data_meta": json.loads((data_dir / "meta.json").read_text()),
        "model_config_path": args.config,
    }
    (last_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
