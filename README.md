# LoRA Fine-Tune: Structured Extraction from Loan-Policy Text

**Status: v3 COMPLETE AND PASSED (2026-11-04).** v3 ran on Kaggle kernel
version 3 (private, GPU, ~11 min): BASE overall 0.0, LoRA overall 1.0
(json_valid 1.0, category 1.0, product 1.0, field 1.0), per-product 1.0
across all five products on the 60 held-out examples. The v3 adapter is
saved at `kaggle-output-v3/outputs/lora-policy/`. Published to GitHub on
2026-11-06: https://github.com/Tejasv609/lora-policy-extraction.

## v2 diagnosis → v3 fix (the only intended v3 change)

v2 (Kaggle kernel, Tesla T4, ~10 min, 8,798,208 trainable params) completed
training but failed the success bar: BASE overall 0.0, LoRA overall 0.233
(json_valid 1.0, category 1.0, product 0.233, field 1.0). The bar needs
overall ≥ 0.80, json_valid == 1.00, and LoRA ≥ base + 0.20 — the first two
parts were met or exceeded, the overall bar was not.

Root cause, verified from the v2 data: **product was unlearnable in v2.**
0 of 300 v2 prompts named any product, and 49 train policy texts mapped to
multiple gold products (e.g. the same sentence labelled personal / home /
small-business / education / auto loan), because every template said only
"this product". Product accuracy 0.233 was essentially chance over 5
products, while category 1.0 and field 1.0 were learnable. More data alone
would not fix this.

v3 fix, and nothing else changed: every policy template — in
`generate_data.py` and in the notebook's data-generation cell — now names
the product naturally in the sentence itself (e.g. "Applicants for the
home loan at Example Finance must have…", "The interest rate for the
auto loan is…", "Borrowers may repay the education loan over…"). No
separate "Product: X" header was added; the product signal is in the
policy text the model reads. Seed 42, 240 train / 60 held-out, the same 5
products and 5 categories, the same LoRA config, the same success bar,
and the torchao>=0.16 install pin from v2 are all unchanged, so the fix
is isolated to the data signal. A new validation check (generator and
notebook) asserts every prompt names its gold product and no policy text
maps to more than one gold product, and eval now reports per-product
accuracy alongside the unchanged headline metrics. The v2 adapter is kept
locally at `kaggle-output-v2/outputs/lora-policy/` for comparison and is
not included in this repo. v3 has since
run and passed (see Status above); the honest caveat remains that 1.0
is on a synthetic held-out set, so it proves the pipeline and the data
fix, not generalisation to messy real policy text.

## Why this task, and why it is distinct

Tejasv's existing portfolio already covers two other LLM skills:

- **lending-copilot (RAG):** retrieve the right document, then answer.
- **underwriting-copilot (agent):** orchestrate tools/steps to reach a decision.

This project covers the third skill neither of those shows: **adapting a model
itself**. Given a short loan-policy snippet, a small model must output one
valid JSON object with the category, the product, and the one fact that
matters for that category — no prose, no retrieval, no tool calls. That is
the skill AI-engineer roles actually test when they ask about fine-tuning:
can you take a base model that rambles, and make it reliably emit structured
output you can feed downstream?

## Task

Input: one synthetic loan-policy sentence (e.g. an interest rate, a tenure
limit, a credit-score floor, a document list, or a collateral requirement).

Output: exactly one JSON object, e.g.

```json
{"category": "interest_rate", "product": "home loan", "interest_rate_percent": 9.5}
```

Categories and their extracted field:

| Category | Field | Type |
|---|---|---|
| eligibility | `min_credit_score` | int |
| interest_rate | `interest_rate_percent` | float |
| repayment | `max_tenure_months` | int |
| documentation | `required_documents` | list of strings |
| collateral | `collateral_type` | string |

## Base model and LoRA config

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct` — small enough to train on a
  free Colab T4, large enough that structured-output behaviour is a real
  test, and openly licensed.
- **LoRA:** r=16, alpha=32 (scaling 2.0), dropout=0.05, target modules
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
  (all attention + MLP projections).
- **Training:** lr 2e-4, effective batch 16 (4 × grad-accum 4), 3 epochs,
  prompt tokens masked so loss is only on the JSON completion.
- **Output:** a PEFT adapter directory only, not a merged model.

## Dataset

**Fully synthetic, no real data.** `generate_data.py` builds snippets from
templates with a fictional institution ("Example Finance") and known gold
labels, so every label is verifiable by construction. Default split:
240 train / 60 held-out, seeded (seed 42), balanced across the 5 categories
by random sampling. No real company is named and no scraped policy text is
used — do not present these examples as real policies.

```bash
python generate_data.py            # writes data/train.jsonl, data/heldout.jsonl
```

## How to train on a Kaggle GPU (the real training route)

This VM has no GPU, so training is not done here. The training route is
**Kaggle, not Colab**: `lora-policy-extraction.ipynb` in this folder is a
self-contained Kaggle kernel (data generation, LoRA training, and base-vs-
LoRA evaluation in one notebook, GPU enabled, internet enabled for the
model download). `train.py` / `eval.py` / `generate_data.py` are the same
logic as standalone scripts for local reference; the notebook is the
Kaggle-executable copy.

**Quota note — read this before pushing.** Tejasv mentioned 40 hours, but
the record from the SHL work is **30 free GPU hours per week** on Kaggle,
so plan against 30, not 40. This run is sized far inside that: 240 examples
× 3 epochs = 45 optimizer steps on a 0.5B model. Honest estimate for the
whole kernel (pip install + model download + training + greedy generation
on 60 held-out examples for both models) is roughly **20–45 minutes of GPU
time**, depending on download speed and queue — a few hours at most even
if it runs slowly, not anywhere near the weekly quota. The estimate is
not a guarantee; if the kernel is queued or the download stalls, it takes
longer, but it does not need a second session.

**How v3 was pushed (2026-11-04, with Tejasv's approval — a pushed
kernel starts executing automatically). The steps below are the record
of that push:**

1. Verify the Kaggle credential first — do not assume the old one works
   (it expired before, in October, and no `~/.kaggle` credential exists
   on this VM). Install the Kaggle CLI if needed (`pip install kaggle`),
   place a current API token at `~/.kaggle/kaggle.json` (never in chat),
   and confirm with `kaggle kernels list --mine`. If that fails with an
   auth error, stop and get a fresh token before pushing anything.
2. From this folder, with `kernel-metadata.json` (id
   `TEJASV002/lora-policy-extraction`, private, GPU on, internet on)
   alongside the notebook, run:
   ```bash
   cd ~/workspace/lora-finetune
   kaggle kernels push -p .
   ```
3. Watch it, do not walk away blind:
   ```bash
   kaggle kernels status TEJASV002/lora-policy-extraction
   kaggle kernels output TEJASV002/lora-policy-extraction --path ./kaggle-output
   ```
   Check the kernel's status page / logs for errors in the first minutes
   (pip install and model download are the usual failure points). If it
   errors, fix the notebook locally and only then re-push with approval.
4. Success is the notebook's own final print: LoRA `overall` ≥ 0.80,
   `json_valid` == 1.00, and LoRA ≥ base + 0.20. The adapter lands in the
   kernel output at `outputs/lora-policy` — download it from there; it is
   not saved anywhere else.

## CPU smoke test (proves the code path, trains nothing useful)

```bash
python train.py --smoke
```

This runs 16 examples for 3 steps on CPU and saves **no adapter**. It is a
syntax/wiring check only — do not report its output as a result.

## What success looks like

On the 60 held-out examples, scored by `eval.py`:

- `json_valid` = 1.00 (every output parses as JSON)
- `overall_exact` (category + product + field all correct) ≥ 0.80 for LoRA
- LoRA `overall_exact` ≥ base `overall_exact` + 0.20

If LoRA does not clear that bar, the honest next step is more epochs or
more training data — not claiming the fine-tune worked. The base model is
expected to score poorly on exact structured output; that gap is the point.

## What was and was not validated here

Validated on this CPU-only VM: Python syntax for all scripts, dataset
generation and format, the eval scoring logic against stub predictions
(perfect predictions pass, garbage fails), plus for the Kaggle notebook
specifically: notebook JSON is valid nbformat 4 (11 cells), every code cell
parses, and the notebook's own data-generation + validation cells and its
stdlib scoring logic were executed locally from the notebook code itself
(240 train / 60 held-out, same category counts as `generate_data.py`;
score 1.0 on perfect predictions, 0.5 on half-garbage).
**Kaggle run (2026-11-04):** v3 ran end-to-end on a Kaggle GPU kernel
(~11 min, Tesla T4): BASE overall 0.0, LoRA overall 1.0 (json_valid 1.0,
category 1.0, product 1.0, field 1.0), per-product 1.0 across all five
products on the 60 held-out examples — notebook verdict PASS.
**Not run on this VM:** any real training or installation of
torch/transformers/peft — the notebook's training and generation cells
need a GPU kernel, and this VM has no torch installed. Honest caveat:
1.0 is on a synthetic held-out set, so it proves the pipeline and the
data fix, not generalisation to messy real policy text.

## Publication

Published to GitHub on 2026-11-06 at
https://github.com/Tejasv609/lora-policy-extraction (public), with
Tejasv's explicit approval. The v3 Kaggle kernel
(`TEJASV002/lora-policy-extraction`, version 3) ran and passed on
2026-11-04; its adapter is included in this repo at
`kaggle-output-v3/outputs/lora-policy/`. The v2 adapter is kept locally
only.
