# Router Agent Milestone 5: Evaluation & Analysis *(7 Nov)*

## Overview
- Consolidates router-agent evaluation work for Milestone 5, focusing on the three Vertex-tuned LoRA adapters (Llama 3.1 8B, Gemma 3 27B, Qwen 3 32B).
- Pulls the latest Hugging Face trainer artefacts and inspects the Milestone 3 dataset splits to quantify routing behaviour, surface bottlenecks, and guide model selection.
- Highlights error patterns (schema drift, rare routing templates, output length inflation) and outlines mitigation ideas for the next training cycle.

> Reproduce the numbers with `python Milestone-5/router-agent/collect_router_metrics.py`.  
> Outputs are stored in `Milestone-5/router-agent/router_eval_metrics.json`.

## Evaluation Setup
- **Data** – Vertex tuning JSONL splits under `Milestone-3/router-agent-scripts/data/vertex_tuning/` (train 6 962 · validation 818 · test 409).
- **Metrics aggregation** – `collect_router_metrics.py` fetches Hugging Face artefacts (`eval_results.json` or `trainer_state.json`) and summarises dataset statistics (route lengths, tool frequencies, difficulty mix, metrics-field schema variants).
- **Environment** – Python 3.11+, `huggingface_hub>=0.24`, no GPU required (downloads metadata only). Script converts `Counter` instances to plain dicts for reproducibility.

## Quantitative Metrics

| Model (LoRA) | Eval Loss ↓ | Perplexity ↓ | BLEU ↑ | Eval Runtime (s) ↓ | Samples/s ↑ | Steps/s ↑ |
|--------------|-------------|--------------|--------|--------------------|-------------|-----------|
| `router-gemma3-peft` | **0.608** | **1.837** | — | 15.43 | **53.02** | **3.37** |
| `router-qwen3-32b-peft` | 0.628 | 1.873 | — | 16.69 | 49.02 | 3.12 |
| `router-llama31-peft` | 0.676 | 1.972 | 0.400 | 67.93 | 12.04 | 1.52 |

Key takeaways:
- All three adapters maintain sub‑2 perplexity on the Vertex validation split, with Gemma 3 posting the lowest loss and highest throughput (53 samples/s vs. Llama’s 12).
- Llama 3.1’s BLEU ≈ 0.40 indicates partial but imperfect overlap with reference JSON plans—useful for relative comparisons but still noisy for structured data.
- Gemma/Qwen runs expose tight runtime variance across epochs (±0.03 s) signalling steady convergence; Gemma dropped from loss ≈ 1.99 at step 0 to 0.61 by epoch 3.

## Dataset Diagnostics (Test Split)

| Statistic | Value |
|-----------|-------|
| Samples | 409 |
| Avg route length | 3.06 steps |
| Route pattern coverage | 93.2 % `/general-search → /math → /code`; 5 x `/math → …` |
| Tool usage totals | `/math`: 433 · `/general-search`: 409 · `/code`: 409 |
| First-tool distribution | `/general-search`: 404 (98.8 %) · `/math`: 5 (1.2 %) |
| Difficulty mix | Advanced 274 · Intermediate 83 · Introductory 52 |
| Avg prompt tokens | 126.9 (word-split proxy) |

Additional schema findings from `router_eval_metrics.json`:
- Metrics block shapes vary: 371 samples keep `{primary, secondary}`, 27 add `*_guidance`, 6 include `*_computation`, and 4 introduce `computation_guidance`. These optional keys are frequent error spots in prior generations.
- All four-step plans appear exclusively in advanced problems, aligning with multi-pass reasoning tasks (e.g., verify math step after code execution).

## Error Analysis
- **Canonical-route bias** – 381/409 test prompts follow the exact `/general-search → /math → /code` template. Validation loss alone can mask whether models learn the rare `/math`‑first or four-step chains; historical inference logs show LoRAs occasionally revert to the dominant plan even when the gold label differs.  
  *Impact:* Misroutes on just the 1.2 % math-first cases can inflate future BLEU/perplexity due to the class imbalance.
- **Optional metrics fields** – 9.3 % of test rows demand extra guidance/computation strings. Earlier checkpoints often omitted these nested strings, triggering post-processing failures despite correct tool order. The absence of field-aware metrics means such schema drops are invisible to current eval dashboards.
- **Output length inflation** – Llama’s `eval_length_ratio=1.178` and `eval_gen_len≈710` vs. reference 606 indicate over-generation, usually via duplicated bullet lists or verbose rationales. Long outputs are more prone to JSON truncation when served under tight max-token limits.
- **Advanced-only edge cases** – Every four-step plan aligns with advanced topics (meta-learning, stochastic calculus). Without targeted sampling, the models have little signal on how to hand control back to `/math` after `/code`, leading to brittle behaviour on auditing / verification workflows.

## Limitations
- Metrics rely on Hugging Face trainer exports (validation split); no held-out test inference or structured scoring (JSON validity, tool-order F1) is automated yet.
- BLEU under-rewards semantically correct plans that paraphrase bullet text or reorder optional fields—interpret with caution.
- Evaluation currently assumes consistent prompt scaffolding; dataset drift (new tools, extra fields) would not be reflected until the script is updated.

## Implemented Improvements (Milestone 5)
- **Schema-aware scoring** – `schema_score.py` compares router predictions against ground-truth JSON, reporting per-field accuracy (route order, tool precision/recall, metrics-key retention) and surfacing length ratios + subset scores (math-first, four-step, guidance-heavy). Output example: `benchmarks/test_schema_eval.json`.
- **Hard-negative benchmark** – `generate_router_benchmark.py` mines the Milestone 3 splits for difficult archetypes (math-first, multi-pass math, optional metrics fields) and produces `benchmarks/deep_router_benchmark.jsonl` plus summary stats (`..._stats.json`) for regression testing or targeted fine-tuning.
- **Length monitoring hooks** – schema scoring now tracks `length_ratio`, `>1.1`, `<0.9` flags, enabling automated guards against over-long completions. Coupled with the benchmark subset, this gives us concrete pass/fail gates before rolling out routers with length-penalised generation configs.

### Benchmark Suites
- **Deep Router Benchmark**  
  ```bash
  python Milestone-5/router-agent/generate_router_benchmark.py \
    --source Milestone-3/router-agent-scripts/data/vertex_tuning/test.jsonl \
    --out Milestone-5/router-agent/benchmarks/deep_router_benchmark.jsonl \
    --stats Milestone-5/router-agent/benchmarks/deep_router_benchmark_stats.json
  ```
  Focuses on held-out test items; emphasises advanced, four-step, and metrics-rich prompts (291 rows).

- **Router Benchmark Hard (v1)**  
  ```bash
  python Milestone-5/router-agent/generate_router_benchmark.py \
    --source Milestone-3/router-agent-scripts/data/vertex_tuning/train.jsonl \
             Milestone-3/router-agent-scripts/data/vertex_tuning/validation.jsonl \
             Milestone-3/router-agent-scripts/data/vertex_tuning/test.jsonl \
    --categories math_first four_step math_backstop math_multi_pass metrics_guidance metrics_computation non_canonical_route \
    --limit-per-category 160 \
    --out Milestone-5/router-agent/benchmarks/router_benchmark_hard.jsonl \
    --stats Milestone-5/router-agent/benchmarks/router_benchmark_hard_stats.json
  ```
  Blends all splits to build a 322-example stress set covering non-canonical routes, math-first starts, multi-pass loops, and optional metrics fields. Sampling caps prevent any single archetype from dominating.

- **Scoring Example**
  ```bash
  python Milestone-5/router-agent/schema_score.py \
    --gold Milestone-5/router-agent/benchmarks/router_benchmark_hard.jsonl \
    --pred <router_predictions.jsonl> \
    --out Milestone-5/router-agent/benchmarks/router_benchmark_hard_eval.json
  ```
  Replace `<router_predictions.jsonl>` with model outputs (one JSON per line) to obtain subset metrics and length checks.

- **Automated Pass/Fail Check**
  ```bash
  python Milestone-5/router-agent/router_benchmark_runner.py \
    --gold Milestone-5/router-agent/benchmarks/router_benchmark_hard.jsonl \
    --pred <router_predictions.jsonl> \
    --thresholds Milestone-5/router-agent/router_benchmark_thresholds.json \
    --out Milestone-5/router-agent/benchmarks/router_benchmark_report.json
  ```
  The runner imports `schema_score`, evaluates every subset, and compares metrics against the thresholds registry so CI can gate deployments.

## Backlog / Next Steps
1. Wire schema-score metrics into CI so every router checkpoint uploads per-subset accuracy and length ratios.
2. Expand the benchmark with synthetic math-first prompts to further balance the canonical-route bias.
3. Experiment with decoder-side `length_penalty` and rationale compression during the next Vertex tuning cycle, using the new reports as acceptance criteria.

Artifacts leveraged in this milestone:
- `Milestone-5/router-agent/collect_router_metrics.py`
- `Milestone-5/router-agent/router_eval_metrics.json`
- `Milestone-5/router-agent/schema_score.py`
- `Milestone-5/router-agent/generate_router_benchmark.py`
- `Milestone-5/router-agent/benchmarks/deep_router_benchmark.jsonl`
- `Milestone-5/router-agent/benchmarks/deep_router_benchmark_stats.json`
- `Milestone-5/router-agent/benchmarks/router_benchmark_hard.jsonl`
- `Milestone-5/router-agent/benchmarks/router_benchmark_hard_stats.json`
- `Milestone-5/router-agent/FINAL_REPORT.md`
- `Milestone-5/router-agent/router_benchmark_thresholds.json`
- `Milestone-5/router-agent/router_benchmark_runner.py`
- `Milestone-3/router-agent-scripts/data/vertex_tuning/{train,validation,test}.jsonl`
