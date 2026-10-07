# Task 2 — Generative AI: Supply-Chain Anomaly Analyst, QLoRA Fine-Tuning (Revised Plan v2)

> Repo folder: `task2_genai/` · Max score 100 (+5 RAG bonus, +5 video) · Est. effort 6–7 h (incl. Colab)
> Prepared: 2026-10-06 · Supersedes `plan_supply_chain_finetuning.md`
> Repo-level rules (folder names, CITATIONS.md, REFLECTION.md, no keys, visible outputs) are in `plan_task1_financial.md` §0.

---

## 1. Use case and structured problem statement

**Use case:** convert a raw supply-chain SKU/warehouse-day event feed into a structured anomaly assessment. Domain-specific, objectively checkable, one of the assessment's suggested strong choices.

**Input:** one JSON object:
`product_category, region, order_volume_vs_forecast_pct, inventory_days_on_hand, on_time_delivery_pct, supplier_status, transit_days_normal, transit_days_observed, port_congestion_index, weather_event, labor_document_note, unit_cost_vs_last_quarter_pct, notes`.

**Output:** valid JSON
`{is_anomaly: bool, anomaly_class, severity: 1–5, root_cause: 2–4 sentences, evidence_fields: [input keys], corrective_actions: 3–5 items, confidence: 0–1}`.

**Taxonomy (10 classes + "none"):** demand spike · demand collapse · supplier shipment delay · port/logistics congestion · weather disruption · quality hold · labor strike · freight inflation · inventory obsolescence · warehouse capacity breach. No-anomaly cases use `is_anomaly=false`, `anomaly_class="none"`.

**Correct / partial / hallucinated**
- Correct: schema-valid, class supported by cited evidence fields, severity consistent with magnitudes, actions actionable and tied to the cause.
- Partially correct: schema-valid but one element off (class plausible-but-secondary, severity off by ≥ 2, weak causality).
- Hallucinated: invalid JSON, contradicts input values, cites an event/field not in the input, or class outside the taxonomy.

**Success criteria:** 150 examples, diversity metrics, val loss logged per epoch (decreasing), ROUGE-L base vs tuned on identical test set, BERTScore **and** LLM-judge, ≥ 10 manual labels with hallucination %, two-paragraph analysis, every hyperparameter justified, merged model on HF Hub.

---

## 2. Rubric traceability

| Criterion (pts) | Implemented by | Evidence |
|---|---|---|
| Use case quality (10) | §1 + `use_case.md` | problem statement, taxonomy |
| Dataset ≥ 100 + teacher prompt (5) | `scripts/generate_dataset.py` | `data/teacher_system_prompt.txt` + notebook appendix cell |
| Diversity metrics (10) | `scripts/diversity_report.py` | length histograms, keyword frequency, family×category matrix |
| JSONL chat + 80/10/10 split (5) | `scripts/build_jsonl.py` | `data/{train,valid,test}.jsonl`, sizes printed |
| QLoRA NF4 + PEFT (10) | Colab notebook | visible config + trainable-params print |
| Hyperparameter justification (15) | table §5.2 | notebook markdown table + `TRAINING_NOTES.md` |
| Loss monitoring (10) | `eval_strategy="epoch"` | loss table + curve |
| Merged model saved (5) | `merge_and_unload()` → HF Hub | public link in README |
| ROUGE-L comparison (8) | `evaluation/` | comparison table |
| Additional metric (7) | BERTScore + judge | two tables |
| Manual review ≥ 10 (7) | `manual_audit` | labelled matrix + rate % |
| Qualitative analysis (8) | notebook final markdown | two paragraphs with real excerpts |
| Bonus RAG (+5) | `rag/` (optional) | before/after example |

---

## 3. Layout and execution split

```
task2_genai/
├── README.md                  # Colab badge, HF link, results
├── use_case.md  TRAINING_NOTES.md
├── data/                      # teacher_system_prompt.txt, raw.jsonl, train/valid/test.jsonl, diversity charts
├── scripts/                   # generate_dataset.py, diversity_report.py, build_jsonl.py
├── notebook/task2_finetune_eval.ipynb      # ONE notebook: train + eval (Colab)
├── evaluation/                # metrics.py, judge.py, manual_audit.py
├── outputs/                   # base_generations.json, tuned_generations.json, metrics tables, audit
└── rag/                       # optional bonus
```

| Where | What |
|---|---|
| Local / any machine (API-bound, no GPU) | dataset generation, diversity report, JSONL build |
| **Colab T4** | base-model inference, training, merge, tuned-model inference, HF push |
| Anywhere (CPU + API) | ROUGE, BERTScore, judge, audit (also run inside the notebook so outputs are visible) |

v1 put "base ROUGE + judge runs" locally; a 3.8B model cannot run reasonably on a laptop CPU. **Both model arms are generated in Colab** and saved as JSON; scoring then reads those files.

---

## 4. Dataset engineering

### 4.1 Scenario matrix (diversity by construction)
`12 families × 5 product categories × 3 regions × 3 severity levels = 540 combinations`
(v1 stated ≈180, which is arithmetically wrong.) Families = 10 taxonomy classes + 2 no-anomaly controls (holiday peak, benign forecast correction).
Sample **170** tuples stratified by family (round-robin, seed 42, ≈ 14 per family) → over-generate to survive schema rejects → keep **150**.

### 4.2 Teacher ≠ student
Teacher = **GLM-5.3-Flash**; student = Phi-3-mini. Different models, so the stated trap is avoided.
- One call per scenario tuple returns `{input_feed, gold_assessment}`: an internally consistent feed plus a grounded assessment. `evidence_fields` must be keys that exist in the feed.
- Pydantic gate on gold; regenerate on failure (max 3). Python check: `evidence_fields ⊆ input keys`, `severity` within the tuple's band.
- Near-duplicate removal: `rapidfuzz.fuzz.token_set_ratio > 85` against accepted examples.
- Usage log → `logs/teacher_usage.csv`.
- The **full teacher system prompt** is saved verbatim to `data/teacher_system_prompt.txt` and printed in a notebook appendix cell.
- Honest limitation (goes in REFLECTION): gold labels are teacher-generated, not human-verified. Mitigated by programmatic consistency checks.

### 4.3 Diversity report (10 pts)
- Prompt-length histogram (tokens, student tokenizer) with min/median/max.
- Keyword/topic frequency: top 25 overall + distinct terms per family.
- Family × category matrix heatmap; check max family share ≤ 15% (uniform is 8.3%).
- Near-duplicate stats (pairs above threshold after filtering = 0).

### 4.4 Format + split (5 pts)
Chat JSONL: `{"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}`.
150 examples → **120 / 15 / 15** (exactly 80/10/10), stratified by family, seed 42; sizes printed. v1's 110/15/15 (79/11/11) was a rounded compromise; 150 removes it.
System turn = taxonomy + JSON spec (same system text used for the base-model baseline).

---

## 5. Fine-tuning

### 5.1 Student: `microsoft/Phi-3-mini-4k-instruct` (3.8B)
Why: ≠ teacher; fits T4 with 4-bit + LoRA; MIT licence allows a public HF push; decent JSON-following.
**Verification cell before training:** `tokenizer.apply_chat_template(messages, tokenize=False)` and confirm the system text appears in the rendered prompt. If the template drops the system role, fall back to `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) with `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. Record the choice in `TRAINING_NOTES.md`.
Rejected: Llama-2-7B (licence friction, VRAM), Mistral-7B (VRAM on T4 with margin), ≤1B (too weak for 2–4-sentence causal reasoning).

### 5.2 Hyperparameters (each with a written reason)
| Param | Value | Reason |
|---|---|---|
| Quantization | 4-bit NF4, double quant, compute dtype **fp16** | NF4 is the QLoRA paper's best quality/size trade; T4 has no bf16 |
| LoRA r | 16 | 3–4B model, ~120 examples: r=8 risks underfitting multi-sentence reasoning; r=32 adds capacity the data can't use |
| LoRA alpha | 32 | α/r = 2, standard scaling |
| target_modules | `qkv_proj, o_proj, gate_up_proj, down_proj` | **Phi-3 fuses Q/K/V into `qkv_proj` and gate/up into `gate_up_proj`**; v1's `q,k,v,o` names don't exist in this architecture and would fail. Adapting all linear layers follows the QLoRA paper's finding. OOM fallback: attention only (`qkv_proj, o_proj`) |
| lora_dropout | 0.05 | tiny data needs light regularization without destroying signal |
| Learning rate | 2e-4 | standard QLoRA range; short run needs a reasonably high LR |
| Scheduler | cosine, warmup_ratio 0.05 | smooth decay; short warm-up because total steps are few |
| Epochs | 3 | enough to show a val-loss trend without memorizing 120 examples |
| Per-device batch × grad accum | 2 × 4 = **effective 8** | fits T4; effective 8 gives ~15 steps/epoch (45 total). v1's effective 16 gave only ~7 steps/epoch — too few updates for per-epoch curves |
| Max sequence length | 1024 | verified from the token histogram: ≥ 95% of examples below ~750 tokens |
| Optimizer | `paged_adamw_32bit` | paging absorbs VRAM spikes |
| weight_decay / max_grad_norm | 0.01 / 0.3 | mild regularization; 0.3 is the QLoRA paper's clip |
| Precision & memory | `fp16=True`, gradient checkpointing, `use_cache=False` | required for the VRAM budget; cache off under checkpointing |
| Attention impl | `sdpa` (or `eager`) | FlashAttention-2 is unsupported on T4 |
| Seed | 42 | reproducibility |

Loss target: learn on **assistant tokens only** (use TRL's prompt/completion or completion-only option supported by the pinned version). Pin `transformers`, `trl`, `peft`, `bitsandbytes` in `requirements.txt` because TRL argument names change between releases (e.g. `max_seq_length` → `max_length`).

### 5.3 Loss monitoring
`eval_strategy="epoch"`, `save_strategy="epoch"`; table of train/val loss per epoch plus curve. Val loss decreasing is a graded requirement but not guaranteed on 15 validation examples, so pre-commit the response:
- val rises at epoch 3 → rerun at LR 1e-4 or 2 epochs;
- train/val gap large → dropout 0.1.
Every rerun and its reason is logged in `TRAINING_NOTES.md`. Report the final run honestly; never edit numbers.

### 5.4 OOM playbook (document real incidents)
1. max seq 1024 → 768 · 2. per-device batch 2 → 1 (accum 4 → 8) · 3. drop MLP targets · 4. restart the runtime (allocator state can't be freed) · 5. r 16 → 8 last.
Each incident: verbatim error, attempts, fix, lesson.

### 5.5 Merge and deliver
Reload base in fp16 (not 4-bit) → attach adapters → `merge_and_unload()` → `save_pretrained` + tokenizer → `push_to_hub` (public, with a short model card and training summary). Token via Colab Secrets. Save checkpoints to Google Drive per epoch so a session loss doesn't cost the run. Drive zip is the documented alternate.

---

## 6. Evaluation

### 6.1 Protocol
- Same 15 test prompts, same decoding (temperature 0, `max_new_tokens=512`) for both arms.
- **Base arm:** Phi-3-mini + the same system prompt, no adapters. Generate **before** training (or with adapters disabled) and save `base_generations.json`.
- **Tuned arm:** merged model → `tuned_generations.json`.
- Normalize before scoring: parse JSON, re-serialize with sorted keys. If parsing fails, score the raw text and count it as a schema failure.

### 6.2 Metrics
1. **ROUGE-L** (`rouge_score`, stemmer on) vs gold on normalized text; also on `root_cause` alone. Table base vs tuned.
2. **BERTScore F1** (semantic; tolerant to key-order and wording).
3. **LLM-as-judge** (structured JSON): `schema_adherence, class_correct, severity_calibration, root_cause_causality, action_quality, evidence_grounding`, each 0–5, plus `one_line_reason`.
   - Judge never sees gold; sees outputs in **randomized order with arm labels hidden**.
   - Prefer a judge model that is **not** the teacher (e.g., Groq `llama-3.3-70b-versatile`); if only GLM is available, disclose the self-preference risk.
   - Consistency gauge: re-judge 3 outputs, report per-dimension differences.
4. **Programmatic checks** (no LLM): JSON valid %, class exact match, severity within ±1, `evidence_fields ⊆ input keys`. Published beside judge scores.

### 6.3 Manual audit (graded, must be genuinely yours)
A small helper prints input, gold and tuned output side by side. **You** label all 15 tuned outputs `correct / partially_correct / hallucinated` with a one-line note. `hallucination_rate = hallucinated / reviewed × 100`. Add a programmatic ground-guard count (e.g., claims of "strike" when `labor_document_note` is empty). Do not let an LLM fill the labels; you are asked to defend them in the interview.

### 6.4 Qualitative analysis
Two paragraphs written after seeing real outputs: (1) where tuning helped, with two real before/after excerpts and numbers; (2) remaining failure modes (likely severity calibration and rare classes) with next steps: more data in weak families, a quality filter on teacher output, DPO for severity.

---

## 7. Bonus — RAG fallback (optional, 1 h, only if on schedule)
ChromaDB persistent store with ~6 short authored reference docs (field dictionary, taxonomy detail, playbooks, KPI thresholds). Trigger: model `confidence < 0.55` → retrieve top 3 → second-pass prompt that must cite chunk IDs → one visible before/after example. Skip if behind; the bonus is only +5.

---

## 8. Milestones
| # | Work | Time |
|---|---|---|
| M1 | teacher prompt + generation + gates | 1.5 h (API-bound; start early, run while building Task 3) |
| M2 | diversity report + JSONL split | 0.5 h |
| M3 | Colab: template check, base generations, QLoRA training | 1.5 h |
| M4 | merge, HF push, tuned generations | 0.5 h |
| M5 | metrics, judge, manual audit, analysis | 1.5 h |
| M6 | README, TRAINING_NOTES, CITATIONS | 0.5 h |

---

## 9. Risks
| Risk | Mitigation |
|---|---|
| Colab GPU unavailable / disconnect | start Colab early, Drive checkpoints, pinned versions |
| Template drops system role | verification cell + Qwen fallback |
| Val loss not monotonic | pre-committed rerun rules, honest reporting |
| Teacher inconsistency | Pydantic + programmatic gates + dedupe |
| Judge bias | different judge model, blinded order, programmatic metrics beside it |

## 10. Interview talking points
Scenario matrix over prompt-only diversity; teacher ≠ student and licence reasoning; fused-projection module names; why effective batch 8 for a small dataset; every hyperparameter tied to model/data size; three independent evaluation routes plus blinded judge; honest handling of teacher-label limits and non-monotonic loss.

## 11. Changes from v1
- Scenario space corrected to 540; 150 examples → exact 120/15/15.
- `target_modules` fixed for Phi-3 fused layers; effective batch 8; added attention implementation and template check.
- Base-model inference moved to Colab; judge blinded and preferably non-teacher; manual audit explicitly human.
- RAG demoted to optional; TRL version pinning added.

*AI-assisted plan; log it in `CITATIONS.md` and include the teacher system prompt in the README/appendix.*
