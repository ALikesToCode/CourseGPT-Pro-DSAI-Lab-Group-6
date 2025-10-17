Router Agent Vertex Tuning Pipeline
===================================

Overview
--------
- Converts the Week-2 router dataset into Vertex AI supervised fine-tuning format.
- Supports end-to-end tuning flow for open models (Llama 3.1 8B Instruct, Gemma 3 27B IT, Qwen 3 32B) using Vertex AI preview.
- Provides launch script to submit PEFT or full fine-tuning jobs, plus guidance for evaluation/deployment.

Dataset Preparation
-------------------
- Source dataset: `Milestone-2/router-agent-scripts/output.jsonl`.
- Conversion command (deterministic shuffle, split, and upload):
  ```bash
  python prepare_vertex_tuning_dataset.py \
      --input ../../Milestone-2/router-agent-scripts/output.jsonl \
      --output-dir data/vertex_tuning \
      --val-ratio 0.1 \
      --test-ratio 0.05 \
      --gcs-prefix gs://router-data-542496349667/router-dataset
  ```
- Resulting JSONL files (strict `prompt`/`completion` schema ready for Vertex tuning):
  - `gs://router-data-542496349667/router-dataset/train.jsonl` (6,962 rows)
  - `gs://router-data-542496349667/router-dataset/validation.jsonl` (818 rows)
  - `gs://router-data-542496349667/router-dataset/test.jsonl` (409 rows)
- Local copies reside under `data/vertex_tuning/` for quick inspection.
- `--limit` flag enables smoke subsets (e.g., `--limit 200`) that were used to validate the pipeline before running at full scale.

Router Data Schema & Validation
------------------------------
- Each record is converted to a Vertex-friendly JSON object with two keys:
  - `prompt`: deterministic system+user instruction that reminds the model to emit strict JSON.
  - `completion`: serialized router plan JSON containing downstream task directives.
- Fields preserved inside the completion payload (extracted from the week-2 dataset):
  - `route_plan`: ordered tool calls (e.g., `/math`, `/code`).
  - `route_rationale`: natural-language justification for the tool mix.
  - `expected_artifacts`, `thinking_outline`, `handoff_plan`, `todo_list`: structured checklists for downstream agents.
  - `difficulty`, `tags`, `acceptance_criteria`, `metrics`: metadata used for grading and evaluation.
- Quality checks performed by `prepare_vertex_tuning_dataset.py`:
  - Validates JSON per line and strips blank rows.
  - Ensures deterministic shuffling via `--seed`.
  - Allows dataset capping (`--limit`) to support smoke runs and few-shot experiments.
- Suggested additional validations before large jobs:
  - Run `python -m json.tool` on a sample file to confirm strict JSON.
  - Spot check 5–10 examples to ensure high-value prompts (no empty tool lists, etc.).
  - Compute basic stats (length distribution, tool frequency) to detect skew—see TODO in `prepare_vertex_tuning_dataset.py` for future automation.

Environment Setup
-----------------
- Create a local virtualenv and install dependencies:
  ```bash
  cd Milestone-3/router-agent-scripts
  python -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
  ```
- Ensure `gcloud auth login` and `gcloud config set project <PROJECT_ID>` are already run. Export optional defaults:
  ```bash
  export GOOGLE_CLOUD_PROJECT=<PROJECT_ID>
  export GOOGLE_CLOUD_REGION=us-central1
  ```

Helper Scripts
--------------
- `prepare_vertex_tuning_dataset.py`
  - Flags: `--input`, `--output-dir`, `--val-ratio`, `--test-ratio`, `--limit`, `--gcs-prefix`, `--seed`.
  - Emits three JSONL files and prints a summary block (counts, file paths, uploaded URIs).
  - Designed to be idempotent: re-running with the same seed regenerates the same splits.
- `launch_vertex_tuning.py`
  - Wraps `vertexai.preview.tuning.sft.preview_train` and auto-detects SDK capabilities (e.g., gracefully downgrades when `output_uri` unsupported).
  - Key flags: `--base-model`, `--train-uri`, `--validation-uri`, `--output-uri`, `--tuning-mode`, `--adapter-size`, `--display-name`, `--labels`, `--wait`.
  - Prints the Vertex job resource name so exact runs can be resumed/queried later.
- Suggested habit for reproducibility:
  - Capture the CLI commands (with timestamp) in `Milestone-3/router-agent-scripts/logs/` for grading transparency.
  - Store the `vertexai` package version (`pip show google-cloud-aiplatform`) alongside results.

Launch a Tuning Job
-------------------
- Submit a PEFT/LoRA job (recommended first run):
  ```bash
  python launch_vertex_tuning.py \
      --train-uri gs://router-data-542496349667/router-dataset/train.jsonl \
      --validation-uri gs://router-data-542496349667/router-dataset/validation.jsonl \
      --output-uri gs://router-data-542496349667/router-tuning/llama31-peft \
      --tuning-mode PEFT_ADAPTER \
      --adapter-size 16 \
      --epochs 3 \
      --display-name router-llama31-peft \
      --wait
  ```
- Switch to full fine-tuning by passing `--tuning-mode FULL --base-model meta/llama3_1@llama-3.1-8b-instruct`.
- Resume or continue training using `--custom-base` pointing at a prior checkpoints directory.
- Monitor jobs from the console (Vertex AI -> Tuning) or keep `--wait` to poll status from the CLI.

Evaluate & Deploy
-----------------
- When the job succeeds, artifacts reside in `<output-uri>/postprocess/node-0/checkpoints/final`.
- Deploy with Vertex endpoints:
  ```python
  from vertexai.preview import model_garden

  model = model_garden.CustomModel(
      gcs_uri="gs://router-data-542496349667/router-tuning/llama31-peft/postprocess/node-0/checkpoints/final",
  )

  endpoint = model.deploy(
      machine_type="g2-standard-12",
      accelerator_type="NVIDIA_L4",
      accelerator_count=1,
  )
  ```
- Route a handful of real router prompts through the endpoint before scaling the dataset run.
- Use `gcloud ai endpoints predict` or the Python SDK to script batch evaluations; remember to include `raw_response=True` for chat-completions style models.
- To download adapters for offline inference:
  ```bash
  gcloud storage cp -r gs://router-data-542496349667/router-tuning/llama31-peft/postprocess/node-0/checkpoints/final ./artifacts/llama31-peft
  ```
- Vertex console exposes training curves (loss vs steps) inside each tuning job; capture screenshots or CSV exports for the final report.
- For Qwen3-32B deployments, ensure the runtime exposes reasoning flags (e.g., vLLM `--enable-reasoning --reasoning-parser deepseek_r1`); Vertex managed endpoints may require a custom container to pass these arguments.

End-to-End Pipeline Verification
--------------------------------
- Smoke data run (200 examples) ensured JSONL schema compliance and GCS upload.
- Smoke tuning job confirmed IAM permissions, SDK compatibility, and checkpoint export location.
- Full dataset tuning kicked off after smoke validations succeeded (see job table below).

Active Tuning Jobs
------------------
| Vertex Job ID | Base model | Tuning mode | Adapter size | Output URI | Status | Eval highlights |
| --- | --- | --- | --- | --- | --- | --- |
| `projects/542496349667/locations/us-central1/tuningJobs/1491991597619871744` | `meta/llama3_1@llama-3.1-8b-instruct` | PEFT | 16 | `gs://router-data-542496349667/router-tuning/llama31-peft` | `JOB_STATE_SUCCEEDED` (2025-10-17 10:42:40 → 13:41:27 UTC) | BLEU ≈ 0.4004; eval loss ≈ 0.6758; perplexity ≈ 1.97; eval runtime ≈ 67.9 s |
| `projects/542496349667/locations/us-central1/tuningJobs/1108622679339958272` | `publishers/google/models/gemma3@gemma-3-27b-it` | PEFT | 16 | `gs://router-data-542496349667/router-tuning/gemma3-peft` | `JOB_STATE_SUCCEEDED` (2025-10-17 10:43:33 → 14:58:00 UTC) | Eval loss ≈ 0.6080; perplexity ≈ 1.84; eval runtime ≈ 15.4 s |
| `projects/542496349667/locations/us-central1/tuningJobs/2183294140421242880` | `qwen/qwen3@qwen3-32b` | PEFT | 16 | `gs://router-data-542496349667/router-tuning/qwen3-32b-peft` | `JOB_STATE_SUCCEEDED` (2025-10-17 10:42:40 → 14:23:06 UTC) | Eval loss ≈ 0.6277; perplexity ≈ 1.87; eval runtime ≈ 16.7 s |

Use the CLI to track progress:
```bash
gcloud ai tuning-jobs list --region=us-central1
gcloud ai tuning-jobs describe projects/542496349667/locations/us-central1/tuningJobs/<JOB_ID>
```

LoRA Configuration Notes
------------------------
- Default parameters:
  - `adapter_size` (rank) = 16; adjust via `--adapter-size` (supported values: 1, 4, 8, 16, 32).
  - Vertex automatically sets LoRA alpha and scaling; for custom tuning, override via `--learning-rate` or `--learning-rate-multiplier`.
  - Training epochs default to 3 (change with `--epochs`).
- Practical tips:
  - Lower ranks (4 or 8) reduce memory footprint for quick iterations; higher ranks (16+) typically improve fidelity on complex routing tasks.
  - Use labels (e.g., `--labels run=peft16,model=llama31`) to keep Vertex job dashboards organized.
  - Store LoRA weight exports located under `<output-uri>/postprocess/node-0/checkpoints/final/adapter_config.json` (metadata) and `adapter_model.bin`.

Qwen3-32B Integration (Dense Thinking Model)
--------------------------------------------
- **Vertex support:** run `gcloud ai models list --region=us-central1 | grep qwen3` to confirm that `qwen/qwen3@qwen3-32b` is available to your project—some tenants still require allowlisting.
- **Data preparation updates:**
  - For complex routing samples (multi-tool, long reasoning), wrap `route_rationale` + `thinking_outline` inside `<think>...</think>` blocks in the completion payload.
  - For simple tasks, emit an empty `<think></think>` to signal fast-path routing.
  - Update the prompt template to mention thinking mode, e.g. “Use `<think>` for multi-step reasoning and return strict JSON.”
  - Optional enhancement: add CLI flag (e.g., `--enable-thinking`) to `prepare_vertex_tuning_dataset.py` to automate tag insertion; until then, preprocess via a notebook or post-processing script—remember to introduce a `thinking` field in the completion JSON when doing so.
- **Smoke-test command (after adding thinking support):**
  ```bash
  python prepare_vertex_tuning_dataset.py \
      --input ../../Milestone-2/router-agent-scripts/output.jsonl \
      --output-dir data/vertex_tuning_smoke_qwen \
      --val-ratio 0.1 \
      --test-ratio 0.05 \
      --limit 200 \
      --seed 42 \
      --gcs-prefix gs://router-data-542496349667/router-dataset-smoke-qwen
  ```
  - Validate that roughly 70% of complex examples contain non-empty `<think>` content while simple routes remain empty.
- **Tuning command (dense 32B model):**
  ```bash
  python launch_vertex_tuning.py \
      --base-model qwen/qwen3@qwen3-32b \
      --train-uri gs://router-data-542496349667/router-dataset/train.jsonl \
      --validation-uri gs://router-data-542496349667/router-dataset/validation.jsonl \
      --output-uri gs://router-data-542496349667/router-tuning/qwen3-32b-peft \
      --tuning-mode PEFT_ADAPTER \
      --adapter-size 16 \
      --epochs 3 \
      --learning-rate-multiplier 0.7 \
      --display-name router-qwen3-32b-peft \
      --labels model=qwen3-32b,mode=thinking \
      --wait
  ```
- If Vertex AI reports `FAILED_PRECONDITION` or `PERMISSION_DENIED`, fall back to self-managed PEFT training on Vertex AI Workbench (install `transformers>=4.51` and `peft`, then follow the Hugging Face Qwen3-32B instructions).
- **Deployment tips:**
  - Use inference stacks that expose the `enable_thinking` toggle (e.g., vLLM ≥0.8.5 with `--enable-reasoning --reasoning-parser deepseek_r1`).
  - For production, disable thinking on low-complexity requests by injecting `/no_think` tokens or setting `enable_thinking=False`.
  - Quantize to 4- or 8-bit where feasible to cut inference cost; expect ~9–10× the cost of Llama 8B per token when left in FP16.

Model Architecture Justification
--------------------------------
- **Llama 3.1 8B Instruct** (Meta)
  - Architecture: 32 decoder-only transformer blocks, 4096 hidden size, grouped-query attention (32 query heads / 8 key-value heads), ~8B parameters, 128K-token context window, SwiGLU + RMSNorm stack.
  - Pros: instruction-tuned baseline with 128K context window, solid balance of quality vs cost, mature LoRA tooling, deployable on g2-standard-12 (NVIDIA L4).
  - Cons: smaller capacity than 27B+ models may limit complex reasoning; full fine-tuning requires larger accelerators.
- **Gemma 3 27B IT** (Google)
  - Architecture: ~48 transformer layers, grouped-query attention, SentencePiece tokenizer derived from Gemini/PaLM (≈260K vocab), 27B dense parameters with safety-tuned instruction head.
  - Pros: higher reasoning headroom, multilingual guardrails baked in, LoRA keeps GPU/VRAM requirements manageable.
  - Cons: inference footprint heavier than 8B models; preview-only tuning surface; LoRA adapters limit full-parameter updates.
- **Qwen3 32B Instruct** (Alibaba)
  - Architecture: 64-layer dense decoder-only transformer, grouped-query attention (64Q / 8KV heads), 32.8B total parameters (31.2B non-embedding active), 32K native context (extendable to 131,072 via YaRN), BBPE tokenizer (~151K vocab) optimized for code/math, native "thinking" mode (`enable_thinking=True`) emitting `<think>...</think>` blocks.
  - Pros: best-in-class agent/tool-calling pre-training among open models; native thinking mode aligns with `thinking_outline`/`route_rationale`; dense architecture simplifies LoRA tuning; multilingual coverage across 100+ locales.
  - Cons: higher inference cost and memory footprint than 8B/27B models (A100/H100 class GPUs recommended); thinking mode increases latency unless selectively disabled; managed Vertex tuning availability may require allowlisting—verify before scheduling jobs.

Model Comparison Snapshot
-------------------------
| Attribute | Llama 3.1 8B Instruct | Gemma 3 27B IT | Qwen3 32B Instruct |
| --- | --- | --- | --- |
| Architecture | 32-layer dense decoder, 4,096 hidden size, GQA (32Q/8KV) | ~48-layer dense decoder, grouped-query attention, PaLM-derived tokenizer (~256K vocab) | 64-layer dense decoder, GQA (64Q/8KV), native thinking mode |
| Active parameters | ~7.4B | ~25.6B | 31.2B |
| Native context | 128K tokens | 128K tokens | 32K tokens (131K with YaRN) |
| Tool/agent pre-training | Moderate | Strong | **Exceptional** |
| Thinking mode | Prompt-engineered | Prompt-engineered | Built-in (`enable_thinking`) |
| Recommended hardware | NVIDIA L4 / g2-standard-12 | L4 or A100 (preview tuning) | A100/H100 (quantize for L4) |
| Inference cost (relative) | Low | High | High (dense 32B; ~9–10× Llama 8B) |

Model Selection Recommendation
------------------------------
- **Best balance of accuracy vs. cost:** Gemma 3 27B IT delivered the lowest validation loss (≈0.608, perplexity ≈1.84) while maintaining strong multilingual guardrails. Choose Gemma when GPU budget allows A3/H100-class hardware and highest-quality routing decisions are required.
- **Cost-efficient production baseline:** Llama 3.1 8B Instruct achieved BLEU ≈0.40 with modest perplexity (≈1.97) and runs comfortably on g2-standard-12 (NVIDIA L4). Use Llama for latency-sensitive or budget-conscious deployments, and escalate to larger models only for difficult tickets.
- **Advanced reasoning or tool orchestration:** Qwen3 32B Instruct fine-tuned successfully (loss ≈0.628, perplexity ≈1.87) and offers native `<think>` reasoning blocks plus top-tier agent/tool training. Prefer Qwen for complex, multi-tool routing flows where explainable chain-of-thought outweighs higher inference cost.
- **Hybrid approach:** Route routine traffic to Llama (or a distilled variant) and hand off escalations to Gemma/Qwen based on desired reasoning depth (`/think` vs `/no_think` control tags).

Pipeline Verification
---------------------
- Stage: run `prepare_vertex_tuning_dataset.py --limit 200 --gcs-prefix .../smoke`.
- Tune: execute `launch_vertex_tuning.py` pointing to the smoke dataset to validate API IAM, dataset schema, checkpoint export.
- Evaluate: deploy the resulting small model, run regression prompts from Week-2 evaluation harness, and log deltas.
- Scale: once the smoke test passes, rerun without `--limit` to produce the full-tuning dataset and submit the production job.

Evaluation & Reporting Plan
---------------------------
- **Automated JSON validation:** use `jq` or a Python script to confirm every prediction is strict JSON; treat invalid JSON as failure. For Qwen3-32B, also check the presence/absence of `<think>...</think>` blocks based on scenario expectations (e.g., `jq 'select(.completion | test(\"<think>.*</think>\"))'`).
- **Routing accuracy metrics:**
  - `tool_recall`: percentage of ground-truth tools present in the generated `route_plan`.
  - `tool_precision`: percentage of predicted tools that exist in ground truth.
  - `exact_plan_match`: boolean; useful for leaderboard metrics.
  - `metadata_consistency`: track if difficulty/tags/acceptance criteria are preserved.
- **Evaluation workflow (sample):**
  1. Export tuned adapter to `gs://.../final`.
  2. Deploy to an endpoint as shown above (keep endpoint name for logging).
  3. Run `python evaluate_router.py --endpoint <ENDPOINT_ID> --test-file data/vertex_tuning/test.jsonl --max-examples 200`.
  4. Aggregated metrics are written to `reports/<timestamp>_metrics.json` for submission.
- **Human spot checks:** manually inspect 10 prompts focusing on failure modes (missing `/general-search`, wrong ordering, low-quality rationale).
- **Reporting template:** include dataset version, training command, Vertex job link, inference endpoint ID, metric table, and qualitative notes.
- **Thinking mode audit (Qwen3-32B):** track proportion of responses containing `<think>...</think>`; ensure thinking blocks are stripped from conversation history before the next turn to avoid drift.

Next Steps
----------
- Extend `evaluate_router.py` (future work) to automate regression checks versus previous router logic.
- Track tuning metrics via `job.tuning_data_statistics` and update documentation with observed gains or failure cases.

References
----------
- Meta AI, “Meta Llama 3.1 8B Instruct,” Hugging Face model card, 2025. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Google DeepMind, “Gemma 3 Technical Report,” 2025 release notes and model cards. https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/
- Alibaba Group, “Qwen3-32B Instruct,” Hugging Face model card & technical blog, 2025. https://huggingface.co/Qwen/Qwen3-32B and https://qwenlm.github.io/blog/qwen3/
