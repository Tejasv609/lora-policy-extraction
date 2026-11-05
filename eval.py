#!/usr/bin/env python3
"""Evaluate base vs LoRA on held-out structured-extraction examples.

Two modes:
  1. Score mode (no torch needed -- this is what validates the logic here):
       python eval.py --score predictions.jsonl --gold data/heldout.jsonl
     predictions.jsonl: one {"prediction": "<model output text>"} per line,
     aligned by order with the gold file. A stub predictions file can be
     produced by hand or by a script; see README.

  2. Generate mode (needs torch + transformers, i.e. Colab):
       python eval.py --generate --model Qwen/Qwen2.5-0.5B-Instruct \
           --gold data/heldout.jsonl --out predictions-base.jsonl
       python eval.py --generate --model Qwen/Qwen2.5-0.5B-Instruct \
           --adapter outputs/lora-policy \
           --gold data/heldout.jsonl --out predictions-lora.jsonl
     Then score each predictions file with mode 1 and compare.

Metrics (all on the gold held-out set):
  json_valid       -- fraction of outputs parsing as JSON
  category_accuracy, product_accuracy
  field_accuracy   -- category-specific field exactly right
  overall_exact    -- category + product + field all right
Pass bar (README): LoRA overall_exact >= 0.80 AND >= base + 0.20,
with json_valid == 1.00. Failing that, iterate (more epochs/data), don't
claim the fine-tune worked.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIELDS = {
    "eligibility": "min_credit_score",
    "interest_rate": "interest_rate_percent",
    "repayment": "max_tenure_months",
    "documentation": "required_documents",
    "collateral": "collateral_type",
}


def extract_json(text: str):
    """Return the first {...} object in text, parsed, or None."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def values_equal(gold, pred) -> bool:
    if isinstance(gold, list) and isinstance(pred, list):
        return gold == pred
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return float(gold) == float(pred)
    return gold == pred


def score(gold_rows: list[dict], predictions: list[str]) -> dict:
    assert len(gold_rows) == len(predictions), (
        f"gold ({len(gold_rows)}) and predictions ({len(predictions)}) lengths differ")
    n = len(gold_rows)
    counts = {"json_valid": 0, "category": 0, "product": 0, "field": 0, "overall": 0}
    failures = []
    for i, (gold, text) in enumerate(zip(gold_rows, predictions)):
        parsed = extract_json(text)
        ok_cat = ok_prod = ok_field = False
        if parsed is not None:
            counts["json_valid"] += 1
            ok_cat = parsed.get("category") == gold["category"]
            ok_prod = parsed.get("product") == gold["product"]
            field = FIELDS[gold["category"]]
            ok_field = field in parsed and values_equal(gold["value"], parsed[field])
        counts["category"] += ok_cat
        counts["product"] += ok_prod
        counts["field"] += ok_field
        overall = ok_cat and ok_prod and ok_field
        counts["overall"] += overall
        if not overall:
            failures.append(i)
    # Per-product breakdown (headline metrics unchanged): product-field
    # accuracy and overall exact-match per gold product, for v3 diagnosis.
    per_product = {}
    for prod in sorted({g["product"] for g in gold_rows}):
        idx = [i for i, g in enumerate(gold_rows) if g["product"] == prod]
        prod_ok = overall_ok = 0
        for i in idx:
            parsed = extract_json(predictions[i])
            if parsed is None:
                continue
            field = FIELDS[gold_rows[i]["category"]]
            p_ok = parsed.get("product") == gold_rows[i]["product"]
            o_ok = (parsed.get("category") == gold_rows[i]["category"] and p_ok
                    and field in parsed
                    and values_equal(gold_rows[i]["value"], parsed[field]))
            prod_ok += p_ok
            overall_ok += o_ok
        per_product[prod] = {
            "n": len(idx),
            "product_accuracy": prod_ok / len(idx),
            "overall_exact": overall_ok / len(idx),
        }
    return {
        "n": n,
        "json_valid": counts["json_valid"] / n,
        "category_accuracy": counts["category"] / n,
        "product_accuracy": counts["product"] / n,
        "field_accuracy": counts["field"] / n,
        "overall_exact": counts["overall"] / n,
        "per_product": per_product,
        "failed_indices": failures,
    }


def load_predictions(path: Path) -> list[str]:
    preds = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line)["prediction"])
    return preds


def load_gold(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def generate(model_name: str, adapter: str | None, gold_rows: list[dict], out: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    with open(out, "w", encoding="utf-8") as f:
        for row in gold_rows:
            ids = tokenizer(row["prompt"] + "\n", return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=128, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({"prediction": text}, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/heldout.jsonl")
    ap.add_argument("--score", help="predictions.jsonl to score (no torch needed)")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir, if any")
    ap.add_argument("--out", default="predictions.jsonl")
    args = ap.parse_args()

    gold = load_gold(Path(args.gold))
    if args.generate:
        generate(args.model, args.adapter, gold, Path(args.out))
        return
    if not args.score:
        ap.error("need --score <predictions.jsonl> or --generate")
    result = score(gold, load_predictions(Path(args.score)))
    print(json.dumps(result, indent=2))
    passed = result["overall_exact"] >= 0.80 and result["json_valid"] == 1.0
    print("PASS" if passed else "FAIL",
          "-- pass bar is overall_exact >= 0.80 and json_valid == 1.00 "
          "(compare base vs LoRA; LoRA should also beat base by >= 0.20)")


if __name__ == "__main__":
    main()
