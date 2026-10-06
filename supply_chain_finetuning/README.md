# Supply-Chain Anomaly Analyst

Fine-tune a small open-weight LLM with **QLoRA (4-bit NF4)** so it converts a raw
supply-chain event feed (JSON) into a structured anomaly assessment (JSON), then
**prove the improvement** with ROUGE-L (base vs fine-tuned), BERTScore, a blinded
LLM-as-judge, programmatic checks, and a manual hallucination audit — all on the
same held-out test set. Training runs on Google Colab free T4; data generation
and metric code run on any machine.

```
scenario matrix (540 tuples) ──▶ sample 170 ──▶ teacher (GLM-5.3-Flash, 7 gates)
        │                                             │
        │                                    data/raw.jsonl (150 accepted)
        │                                             │
        │                     ┌───────────────────────┼──────────────────────┐
        │                     ▼                       ▼                      ▼
        │              diversity report           chat JSONL          120/15/15 split
        │                                             │
        │                                             ▼
        │                     Colab T4: base baseline ──▶ QLoRA SFT ──▶ merge ──▶ HF Hub
        │                                             │
        │                                             ▼
        └────────────────────▶ ROUGE-L + BERTScore + blinded judge + ground-guard + manual audit
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<COLAB_NOTEBOOK_LINK_PLACEHOLDER>)

**Fine-tuned model:** <HUGGINGFACE_MODEL_LINK_PLACEHOLDER>

## Stage 1 — dataset (any machine, API-bound)

```bash
cp .env.example ../.env      # then fill TEACHER_API_KEY / JUDGE_API_KEY (never commit)
python scripts/generate_dataset.py        # resumable; stops at 150 accepted
python scripts/diversity_report.py        # charts -> data/diversity/, tables to stdout
python scripts/build_jsonl.py             # chat JSONL + exact 120/15/15 stratified split
pytest -q                                 # offline verification suite
```

Gates reject and regenerate bad samples (max 3 attempts per tuple): schema
validation, evidence-fields subset, class match, severity band, sentence/action
counts, near-duplicate filter (rapidfuzz), and a running family-share cap.

## Stage 2 — fine-tuning (Colab T4)

Open `notebooks/finetune_and_evaluate.ipynb` in Colab and run top to bottom:
keys come from Colab Secrets (`TEACHER_API_KEY`, `JUDGE_API_KEY`, `HF_TOKEN`);
the notebook verifies the chat template, generates the **base-model baseline
first**, trains with TRL `SFTTrainer` (assistant-token loss), monitors per-epoch
train/validation loss, merges the adapters, and pushes to the Hugging Face Hub.

## Stage 3 — evaluation (runs in the notebook; modules reusable anywhere)

| Metric | What it shows |
|---|---|
| ROUGE-L (full JSON + root cause only) | lexical alignment with gold, base vs tuned |
| BERTScore F1 | semantic alignment, robust to JSON key order |
| Blinded LLM-as-judge (6 dimensions, 0-30) | quality scores with arm labels hidden and gold withheld |
| Programmatic checks | % valid JSON, class exact-match, severity within ±1, evidence subset, ground-guard violations |
| Manual hallucination audit | human-labelled 15 tuned outputs → hallucination rate |

## Results

*(to be filled after the notebook run — no numbers are fabricated here)*

| Metric | Base (system prompt only) | Fine-tuned |
|---|---|---|
| ROUGE-L (JSON) | — | — |
| ROUGE-L (root cause) | — | — |
| BERTScore F1 | — | — |
| Judge total (0-30) | — | — |
| Valid JSON % | — | — |
| Class exact-match % | — | — |
| Hallucination rate (manual) | — | — |

## Repository layout

```
supply_chain_finetuning/
├── README.md  use_case.md  TRAINING_NOTES.md
├── requirements.txt (local)  requirements-colab.txt  .env.example
├── src/          config, schemas, prompts, llm_client, scenario_matrix
├── scripts/      generate_dataset.py, diversity_report.py, build_jsonl.py
├── evaluation/   normalize.py, metrics.py, judge.py, manual_audit.py
├── notebooks/    finetune_and_evaluate.ipynb   (Colab)
├── data/         teacher_system_prompt.txt, raw/train/valid/test.jsonl, diversity charts
├── outputs/      base/tuned generations, metrics, judge results, audit template
├── rag/          optional retrieval fallback (disabled by default)
└── tests/        offline pytest suite
```
