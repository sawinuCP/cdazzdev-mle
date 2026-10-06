# Training Notes — Supply-Chain Anomaly Analyst

Every hyperparameter carries a written reason tied to the model size (Phi-3-mini,
3.8B) and the data size (~120 training examples). Nothing is left at a silent
default; if a value changes during a run, the change and its reason are logged
in the incident log below.

## Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| Quantization | 4-bit **NF4** + double quant, compute dtype **fp16** | NF4 is the best quality/size trade in the QLoRA paper; T4 has no bf16, so fp16 compute |
| LoRA **r** | 16 | 3-4B model with ~120 examples: r=8 risks underfitting multi-sentence causal reasoning; r=32 adds capacity the data cannot use |
| LoRA **alpha** | 32 | alpha/r = 2, the standard scaling ratio, preserves the effective capacity of r=16 |
| **target_modules** | `qkv_proj, o_proj, gate_up_proj, down_proj` | **Phi-3 fuses Q/K/V into `qkv_proj` and gate/up into `gate_up_proj`** — the split `q_proj/k_proj/v_proj` names do not exist in this architecture and would silently train nothing. Adapting all linear layers follows the QLoRA paper's finding. OOM fallback: attention only (`qkv_proj, o_proj`) |
| **lora_dropout** | 0.05 | tiny-data regime: not zero (overfit risk), not 0.1 (signal loss) |
| bias | none | LoRA convention; bias tuning adds parameters without capacity benefits here |
| task_type | CAUSAL_LM | decoder-only language modelling |
| **learning_rate** | 2e-4 | standard QLoRA range; a short run on modest data warrants the upper-middle of 1e-4..3e-4 |
| **lr_scheduler_type** | cosine | smooth decay avoids abrupt end-of-run validation bumps at small step counts |
| **warmup_ratio** | 0.05 | short warm-up because total steps are few (~45) |
| **num_train_epochs** | 3 | enough to evidence a validation-loss trend without memorizing 120 examples |
| **per_device_train_batch_size x grad accum** | 2 x 4 = **effective 8** | fits T4-16GB; effective 8 gives ~15 steps/epoch (~45 total) — enough updates for per-epoch curves (effective 16 would give only ~7) |
| **max sequence length** | 1024 | justified by the measured token histogram: >= 95% of full chat examples fit within ~1024 tokens (see the diversity report) |
| **optim** | `paged_adamw_32bit` | paged optimizer absorbs VRAM spikes without a measured quality penalty |
| **weight_decay / max_grad_norm** | 0.01 / 0.3 | mild regularization; 0.3 is the QLoRA paper's clip value |
| **precision** | `fp16=True` | required on T4 (no bf16); paired with gradient checkpointing |
| **gradient checkpointing / use_cache** | True / False | mandatory for the VRAM budget; cache off under checkpointing |
| **attention implementation** | `sdpa` (or `eager`) | FlashAttention-2 is unsupported on T4 |
| **seed** | 42 | reproducibility |
| **eval/save/logging strategy** | epoch | per-epoch loss table is a required artifact |
| **report_to** | none | loss curves are rendered in-notebook; no external account needed |
| **loss scope** | assistant tokens only (prompt/completion masking) | the system+user prompt is fixed context; training only on the completion matches the deployment objective |

## Version pinning

TRL renames arguments between releases (`max_seq_length` -> `max_length`,
completion-only loss switches from dataset layout to a config flag). The
training cell therefore introspects `SFTConfig` fields at runtime and sets
whichever name the installed version supports. `requirements-colab.txt` pins
the compatible major lines (`transformers<4.47`, `trl<0.13`, `peft<0.14`).

## Pre-committed responses (decided BEFORE training)

- Validation loss rises at epoch 3 -> rerun at learning rate 1e-4 OR 2 epochs;
  both attempts are reported, never silently swapped.
- Train/validation gap large -> dropout 0.05 -> 0.1 rerun.
- Every rerun and its reason is recorded below; numbers are never edited.

## OOM escalation order

1. max sequence length 1024 -> 768
2. per-device batch 2 -> 1 (accumulation 4 -> 8, same effective batch)
3. drop MLP targets (`gate_up_proj`, `down_proj`) — attention only
4. restart the runtime (allocator state cannot be freed mid-graph)
5. LoRA r 16 -> 8 (last resort; also changes the capacity story)

## Incident log

*(real incidents only — add one row per occurrence)*

| # | Verbatim error | Attempted fixes | What worked | Lesson |
|---|---|---|---|---|
| — | — | — | — | — |
