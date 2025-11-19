# Milestone 5 — Router Agent Final Report

## 1. Overview
- Evaluated three Vertex AI LoRA adapters (`router-gemma3-peft`, `router-qwen3-32b-peft`, `router-llama31-peft`) using the managed tuning metadata plus schema-aware checks.
- Authored a reusable evaluation toolkit:
  - `collect_router_metrics.py` — aggregates training metrics and dataset statistics.
  - `schema_score.py` — field-level scoring with length monitoring and subset breakdowns.
  - `router_benchmark_runner.py` — threshold gatekeeper for CI / deployment.
- Published two benchmark splits:
  - `benchmarks/deep_router_benchmark.jsonl` (291 eval items – held-out test subset).
  - `benchmarks/router_benchmark_hard.jsonl` (322 stress cases spanning math-first, four-step, guidance/computation variants).

## 2. Quantitative Metrics
| Model | Eval Loss ↓ | Perplexity ↓ | BLEU ↑ | Eval Runtime (s) ↓ | Samples/s ↑ | Steps/s ↑ |
|-------|-------------|--------------|--------|--------------------|-------------|-----------|
| `router-gemma3-peft` | **0.608** | **1.837** | — | 15.43 | **53.02** | **3.37** |
| `router-qwen3-32b-peft` | 0.628 | 1.873 | — | 16.69 | 49.02 | 3.12 |
| `router-llama31-peft` | 0.676 | 1.972 | 0.400 | 67.93 | 12.04 | 1.52 |

Dataset diagnostics (test split, 409 rows):
- Average route length: 3.06 steps.
- Canonical plan `/general-search → /math → /code`: 93.2 % of samples.
- Advanced difficulty share: 67.0 % (274/409); all four-step plans are advanced.
- Optional metrics fields: 9.3 % include guidance/computation strings.

## 3. Schema-Aware Benchmarking
- `schema_score.py` evaluates JSON validity, route order, metrics retention, todo alignment, handoff coverage, and length ratios.
- `router_benchmark_thresholds.json` defines minimum/maximum KPIs per subset (overall, math-first, four-step, guidance/computation, advanced).
- `router_benchmark_runner.py` combines both tools and returns pass/fail status plus detailed checks for CI.

Example CLI:
```bash
python schema_score.py \
  --gold Milestone-5/router-agent/benchmarks/router_benchmark_hard.jsonl \
  --pred <router_predictions.jsonl> \
  --out Milestone-5/router-agent/benchmarks/router_benchmark_eval.json

python router_benchmark_runner.py \
  --gold Milestone-5/router-agent/benchmarks/router_benchmark_hard.jsonl \
  --pred <router_predictions.jsonl> \
  --thresholds Milestone-5/router-agent/router_benchmark_thresholds.json \
  --out Milestone-5/router-agent/benchmarks/router_benchmark_report.json
```

## 4. Error Analysis Highlights
- Canonical-route bias risks mis-routing the 1.2 % math-first cases; monitor with the dedicated subset metrics.
- Optional metrics fields (guidance/computation) were historically brittle; thresholds enforce ≥90 % retention when present.
- Llama adapter over-generates (`length_ratio ≈ 1.18`) — length ratio thresholds catch regressions when tuning decoder settings.

## 5. Deliverables
- `collect_router_metrics.py` / `router_eval_metrics.json`
- `schema_score.py`, `router_benchmark_runner.py`, `router_benchmark_thresholds.json`
- Benchmark data in `benchmarks/*.jsonl`
- Documentation in `README.md` (evaluation narrative) and this final report.

## 6. Recommended Next Steps
1. Automate `router_benchmark_runner.py` in CI for every new checkpoint.
2. Generate synthetic math-first/four-step training data to reduce canonical bias.
3. Apply length penalties or rationale compression and compare before/after metrics using the benchmark thresholds.
