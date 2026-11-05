#!/usr/bin/env python3
"""Generate a small SYNTHETIC dataset for structured loan-policy extraction.

No real company, customer, or copyrighted policy text is used. Every snippet
is built from the templates below, with a fictional institution name, so the
gold JSON is known exactly by construction. This is deliberate: for a
fine-tuning demo, clean, verifiable labels beat scraped real text.

Output schema (one JSON object per line):
  {"prompt": str, "completion": str, "category": str, "product": str,
   "field": str, "value": <int | float | str | list[str]>}

`field`/`value` are the category-specific fact the model must extract:
  eligibility   -> min_credit_score (int)
  interest_rate -> interest_rate_percent (float)
  repayment     -> max_tenure_months (int)
  documentation -> required_documents (list of str)
  collateral    -> collateral_type (str)

Usage:
    python generate_data.py [--train 240] [--heldout 60] [--seed 42]
Writes data/train.jsonl and data/heldout.jsonl next to this script.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTITUTION = "Example Finance"  # fictional, synthetic-only

PRODUCTS = ["personal loan", "home loan", "auto loan", "small business loan", "education loan"]

DOC_SETS = [
    ["identity proof", "address proof", "income proof"],
    ["identity proof", "bank statements", "salary slips"],
    ["identity proof", "address proof", "property documents"],
    ["identity proof", "business registration", "bank statements"],
    ["identity proof", "admission letter", "fee structure"],
]

COLLATERAL = ["property", "vehicle", "fixed deposit", "gold", "none"]

# Each template takes the sampled value and the product, and returns a
# policy snippet that NAMES the product in the sentence itself (v3 fix).
# v2 templates said only "this product", so product was unlearnable: 0 of
# 300 v2 prompts named any product, and 49 train policy texts mapped to
# multiple gold products. Do NOT revert to a generic "this product" form,
# and do NOT add a separate "Product: X" header — the signal must be in
# the policy text the model reads.
TEMPLATES = {
    "eligibility": [
        lambda v, p: f"Applicants for the {p} at {INSTITUTION} must have a minimum credit score of {v}.",
        lambda v, p: f"To qualify for the {p}, the applicant needs a credit score of at least {v}.",
    ],
    "interest_rate": [
        lambda v, p: f"The interest rate for the {p} is {v}% per annum, fixed for the loan term.",
        lambda v, p: f"The {p} carries an annual interest rate of {v}%.",
    ],
    "repayment": [
        lambda v, p: f"The maximum repayment tenure for the {p} is {v} months.",
        lambda v, p: f"Borrowers may repay the {p} over up to {v} months.",
    ],
    "documentation": [
        lambda v, p: f"Applicants for the {p} must submit the following documents: {', '.join(v)}.",
        lambda v, p: f"Required documents for the {p} are {', '.join(v)}.",
    ],
    "collateral": [
        lambda v, p: f"The {p} is secured against {v} as collateral.",
        lambda v, p: f"Collateral required for the {p}: {v}.",
    ],
}

FIELDS = {
    "eligibility": "min_credit_score",
    "interest_rate": "interest_rate_percent",
    "repayment": "max_tenure_months",
    "documentation": "required_documents",
    "collateral": "collateral_type",
}


def sample_value(rng: random.Random, category: str):
    if category == "eligibility":
        return rng.randrange(650, 801, 10)
    if category == "interest_rate":
        return round(rng.uniform(8.0, 14.5), 1)
    if category == "repayment":
        return rng.choice([12, 24, 36, 48, 60, 84, 120, 180, 240])
    if category == "documentation":
        return rng.choice(DOC_SETS)
    return rng.choice(COLLATERAL)


def snippet_from_prompt(prompt: str) -> str:
    """Extract the policy text from a generated prompt."""
    marker = "Policy text: "
    start = prompt.index(marker) + len(marker)
    end = prompt.index("\n\nReturn", start)
    return prompt[start:end]


def validate(rows: list[dict], label: str) -> dict:
    """v3 validation: product signal must be present and unambiguous.

    Checks every row: (1) the prompt names its gold product, and
    (2) no policy text maps to more than one gold product.
    Raises AssertionError on any failure; returns the counts.
    """
    named = 0
    text_to_products: dict[str, set[str]] = {}
    for row in rows:
        gold = json.loads(row["completion"])
        assert gold["category"] == row["category"] and gold["product"] == row["product"]
        assert gold[row["field"]] == row["value"]
        if row["product"] in row["prompt"]:
            named += 1
        text_to_products.setdefault(snippet_from_prompt(row["prompt"]), set()).add(row["product"])
    contradictory = {t: ps for t, ps in text_to_products.items() if len(ps) > 1}
    assert named == len(rows), f"{label}: only {named}/{len(rows)} prompts name their gold product"
    assert not contradictory, f"{label}: {len(contradictory)} policy texts map to multiple products"
    return {"rows": len(rows), "prompts_naming_gold_product": named, "contradictory_texts": 0}


def make_example(rng: random.Random) -> dict:
    category = rng.choice(list(TEMPLATES))
    product = rng.choice(PRODUCTS)
    value = sample_value(rng, category)
    snippet = rng.choice(TEMPLATES[category])(value, product)
    gold = {"category": category, "product": product, FIELDS[category]: value}
    prompt = (
        "Extract the loan policy details as JSON.\n\n"
        f"Policy text: {snippet}\n\n"
        "Return only JSON with keys: category, product, and the one "
        "category-specific field."
    )
    return {
        "prompt": prompt,
        "completion": json.dumps(gold, ensure_ascii=False),
        "category": category,
        "product": product,
        "field": FIELDS[category],
        "value": value,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=240)
    ap.add_argument("--heldout", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train = [make_example(rng) for _ in range(args.train)]
    heldout = [make_example(rng) for _ in range(args.heldout)]
    for label, rows in (("train", train), ("heldout", heldout)):
        print(f"validation {label}: {validate(rows, label)}")

    out = HERE / "data"
    out.mkdir(exist_ok=True)
    for name, rows in (("train.jsonl", train), ("heldout.jsonl", heldout)):
        with open(out / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {out / name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
