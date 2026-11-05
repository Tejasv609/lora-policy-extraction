#!/usr/bin/env python3
"""LoRA fine-tune Qwen2.5-0.5B-Instruct for structured policy extraction.

Real training route: a Kaggle GPU kernel (see README; the self-contained
notebook lora-policy-extraction.ipynb carries this same logic). The --smoke
flag is a tiny CPU sanity run (few examples, few steps) that only proves
the code path works -- it does not produce a useful adapter.

Trains on prompt -> JSON completion pairs from generate_data.py, masking the
prompt tokens so loss is computed on the JSON completion only.

Usage (Colab T4):
    pip install -r requirements.txt
    python generate_data.py
    python train.py --data data/train.jsonl --output outputs/lora-policy

Usage (CPU smoke test, no useful adapter expected):
    python train.py --smoke

Output: a PEFT adapter directory (adapter_model.safetensors + config),
NOT a full merged model. Merge/export is a separate, optional step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# LoRA config -- the numbers the README documents:
#   r=16, alpha=32 (scaling 2.0), dropout 0.05, all attention + MLP projections.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--output", default="outputs/lora-policy")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny CPU run: 16 examples, 3 steps. Proves the path, trains nothing useful.")
    args = ap.parse_args()

    # Imports are inside main so `python -m py_compile train.py` works
    # on a machine with no torch installed (this VM included).
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from torch.utils.data import Dataset

    rows = load_rows(Path(args.data))
    if args.smoke:
        rows = rows[:16]
        args.epochs = 1.0
    print(f"training rows: {len(rows)}  model: {args.model}  smoke: {args.smoke}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    print(f"cuda available: {has_cuda}  (smoke on CPU is expected to be slow but tiny)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
    )
    model.config.use_cache = False
    if has_cuda:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    class SFTDataset(Dataset):
        def __init__(self, rows):
            self.examples = []
            for row in rows:
                prompt = row["prompt"] + "\n"
                full = prompt + row["completion"] + tokenizer.eos_token
                p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                f_ids = tokenizer(full, add_special_tokens=False)["input_ids"][: args.max_length]
                labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
                labels = labels[: args.max_length]
                self.examples.append({"input_ids": f_ids, "labels": labels})

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i):
            return self.examples[i]

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [tokenizer.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=3 if args.smoke else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no" if args.smoke else "epoch",
        bf16=has_cuda,
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=SFTDataset(rows),
                      data_collator=collate)
    trainer.train()
    if not args.smoke:
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"saved LoRA adapter to {args.output}")
    else:
        print("smoke run finished -- no adapter saved (expected).")


if __name__ == "__main__":
    main()
