# LoRA Fine-Tune: Structured Extraction from Loan-Policy Text

Fine-tunes Qwen2.5-0.5B-Instruct with LoRA to turn a short loan-policy sentence into exactly one JSON object — the policy category, the loan product, and the one fact that matters for that category. No prose, no retrieval, no tool calls: the output is structured data a downstream system can consume directly.

Example input:

> "The interest rate for the home loan is 9.5% per annum, fixed for the loan term."

Example output:

```json
{"category": "interest_rate", "product": "home loan", "interest_rate_percent": 9.5}
```

## Results

Scored on 60 held-out examples, greedy generation, exact match on all three fields:

| Model | Valid JSON | Category | Product | Field | Overall |
| --- | --- | --- | --- | --- | --- |
| Base Qwen2.5-0.5B-Instruct | 0.983 | 0.0 | 0.0 | 0.0 | 0.0 |
| + LoRA (this repo) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Per-product accuracy for the LoRA model is 1.0 for all five products (auto loan 8/8, education loan 13/13, home loan 11/11, personal loan 12/12, small business loan 16/16). The trained adapter is in `kaggle-output-v3/outputs/lora-policy/`.

## The task

One synthetic policy sentence in, one JSON object out. The five categories and the field each one extracts:

| Category | Field | Type |
| --- | --- | --- |
| eligibility | `min_credit_score` | int |
| interest_rate | `interest_rate_percent` | float |
| repayment | `max_tenure_months` | int |
| documentation | `required_documents` | list of strings |
| collateral | `collateral_type` | string |

The base model scores 0.0 overall because it does not reliably emit the exact structured format. Closing that gap — making a small model emit valid, exact JSON every time — is the point of the fine-tune.

## Method

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **LoRA:** r=16, alpha=32, dropout=0.05, applied to all attention and MLP projections (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) — 8,798,208 trainable parameters (1.75% of the model).
- **Training:** lr 2e-4, effective batch 16, 3 epochs, prompt tokens masked so the loss is only on the JSON completion.
- **Output:** a PEFT adapter directory only, not a merged model.

## Dataset

Fully synthetic, no real data. `generate_data.py` builds policy sentences from templates with a fictional institution ("Example Finance") and known gold labels, so every label is verifiable by construction. No real company is named and no scraped policy text is used — these are not real policies.

- 240 train / 60 held-out, seed 42
- 5 products: personal loan, home loan, auto loan, small business loan, education loan
- 5 categories, sampled across the templates above

```bash
python generate_data.py   # writes data/train.jsonl, data/heldout.jsonl
```

### A data bug worth knowing about

The first version of this dataset said only "this product" in every sentence, never naming the product. The product label was therefore unlearnable: 0 of 300 prompts named a product, and 49 training sentences mapped to multiple gold products. That run scored product 0.233 — essentially chance over five products — while category and field were both 1.0.

The fix was in the data, not the model: every template now names the product in the sentence itself ("Applicants for the home loan…", "The interest rate for the auto loan is…"). The data generator and the notebook both validate that every prompt names its gold product and that no policy text maps to more than one product. With that fixed and nothing else changed, product accuracy went from 0.233 to 1.0.

## Reproduce it

Training runs on a Kaggle GPU kernel, not on CPU. `lora-policy-extraction.ipynb` is self-contained: data generation, LoRA training, and base-vs-LoRA evaluation in one notebook (GPU enabled, internet enabled for the model download). `train.py`, `eval.py`, and `generate_data.py` are the same logic as standalone scripts for local reference.

```bash
# from this folder, with the Kaggle CLI set up
kaggle kernels push -p .
kaggle kernels status tejasv002/lora-policy-extraction
kaggle kernels output tejasv002/lora-policy-extraction --path ./kaggle-output
```

The run takes about 11 minutes on a Tesla T4. Success is the notebook's own final print: LoRA `overall` ≥ 0.80, `json_valid` == 1.00, and LoRA ≥ base + 0.20.

CPU smoke test (proves the code path, trains nothing useful):

```bash
python train.py --smoke
```

## Repository structure

```
.
├── lora-policy-extraction.ipynb   # self-contained Kaggle notebook: data, training, eval
├── generate_data.py               # builds data/train.jsonl and data/heldout.jsonl
├── train.py                       # standalone LoRA training script
├── eval.py                        # standalone scoring (JSON validity, category, product, field)
├── data/                          # generated train (240) and held-out (60) examples
├── kaggle-output-v3/outputs/lora-policy/   # trained v3 adapter + tokenizer
├── kernel-metadata.json           # Kaggle kernel config
└── requirements.txt
```

## Limitations

The 1.0 score is on a synthetic held-out set drawn from the same templates as the training data. It proves the fine-tuning pipeline works and that the data bug above was the real blocker — it does not prove the model would generalise to messy, real-world policy text. That is the honest next test for this adapter: paraphrased or hand-written policy sentences it has never seen in template form.
