# Math Agent — Vertex Tuning README

This guide shows how to prepare the MathX subset, upload it to Google Cloud Storage (GCS), and launch Vertex AI supervised tuning jobs to produce PEFT adapters for math reasoning models.

## Prerequisites

- Google Cloud SDK (`gcloud`) installed and configured
- A GCP project with Vertex AI APIs enabled and a GCS bucket you can read/write
- Python 3.8+ and the repository scripts `prepare_vertex_tuning.py` and `launch_vertex_tuning.py`
- Recommended: a virtual environment (venv/conda) and service account or Application Default Credentials with permissions for Vertex AI and the target bucket

## 1. Set up your environment

Authenticate and configure `gcloud` and ADC (Application Default Credentials):

Bash / WSL / macOS:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud config set compute/region us-central1
export BUCKET_NAME="your-gcs-bucket-name"    # e.g. dsai-bucket
```

PowerShell (Windows):

```powershell
& gcloud auth login
& gcloud auth application-default login
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud config set compute/region us-central1
$env:BUCKET_NAME = "your-gcs-bucket-name"
```

Tip: If you run in WSL, prefer the Bash example. Ensure the service account or user account has Vertex AI and Storage permissions.

## 2. Prepare and upload data (one-time)

The included `prepare_vertex_tuning.py` downloads and formats a MathX subset and uploads two JSONL files (train/validation) to GCS.

Bash / WSL:

```bash
python prepare_vertex_tuning.py \
    --bucket "$BUCKET_NAME" \
    --gcs-prefix "mathx-dataset-v1"
```

PowerShell:

```powershell
python prepare_vertex_tuning.py --bucket $env:BUCKET_NAME --gcs-prefix "mathx-dataset-v1"
```

When complete the script will print the GCS URIs, for example:

- `gs://dsai-bucket/mathx-dataset-v1/train.jsonl`
- `gs://dsai-bucket/mathx-dataset-v1/validation.jsonl`

Use those exact URIs when launching tuning jobs.

## 3. Launch Vertex AI tuning jobs

Use `launch_vertex_tuning.py` to submit supervised tuning jobs. Below are tested example invocations—adjust URIs, display names, and hyperparameters for your experiments.

Notes:
- `--base-model` should use the Vertex / publisher identifier format: `publisher/model@version`
- `--output-uri` is a GCS prefix where Vertex will write adapter artifacts

Bash example (Qwen3 32B):

```bash
python launch_vertex_tuning.py \
    --base-model "qwen/qwen3@qwen3-32b" \
    --display-name "math-qwen3-32b-peft" \
    --train-uri "gs://$BUCKET_NAME/mathx-dataset-v1/train.jsonl" \
    --validation-uri "gs://$BUCKET_NAME/mathx-dataset-v1/validation.jsonl" \
    --output-uri "gs://$BUCKET_NAME/adapters/math-qwen3-32b" \
    --epochs 3 \
    --adapter-size 16
```

PowerShell equivalent:

```powershell
python launch_vertex_tuning.py --base-model "qwen/qwen3@qwen3-32b" --display-name "math-qwen3-32b-peft" --train-uri "gs://$env:BUCKET_NAME/mathx-dataset-v1/train.jsonl" --validation-uri "gs://$env:BUCKET_NAME/mathx-dataset-v1/validation.jsonl" --output-uri "gs://$env:BUCKET_NAME/adapters/math-qwen3-32b" --epochs 3 --adapter-size 16
```

Other tested model examples (Bash):

Gemma 3 (27B-instruction-tuned):

```bash
python launch_vertex_tuning.py \
    --base-model "google/gemma3@gemma-3-27b-it" \
    --display-name "math-gemma-3-27b-peft" \
    --train-uri "gs://$BUCKET_NAME/mathx-dataset-v1/train.jsonl" \
    --validation-uri "gs://$BUCKET_NAME/mathx-dataset-v1/validation.jsonl" \
    --output-uri "gs://$BUCKET_NAME/adapters/math-gemma-3-27b" \
    --epochs 3 \
    --adapter-size 16
```

Llama4 Scout (17B, instruct):

```bash
python launch_vertex_tuning.py \
    --base-model "meta/llama4@llama-4-scout-17b-16e-instruct" \
    --display-name "math-llama4-scout-17b-peft" \
    --train-uri "gs://$BUCKET_NAME/mathx-dataset-v1/train.jsonl" \
    --validation-uri "gs://$BUCKET_NAME/mathx-dataset-v1/validation.jsonl" \
    --output-uri "gs://$BUCKET_NAME/adapters/math-llama4-scout-17b" \
    --epochs 3 \
    --adapter-size 16
```

## Notes & tips

- Replace placeholder values (project id, bucket name, URIs) with your real values.
- The `prepare_vertex_tuning.py` script prints the final `train.jsonl` and `validation.jsonl` GCS URIs — use those when launching tuning jobs.
- Monitor job status and logs in the Google Cloud Console (Vertex AI > Training > Jobs).
-- Make sure Vertex AI (service account) or your ADC has read/write access to the bucket.

## 4. Evaluation & Benchmarks

After training, evaluate adapters on standard math benchmarks. Recommended datasets:

- GSM8K (basic multi-step word problems)
- MATH (competition problems, harder)
- SVAMP (robustness / paraphrases)
- MAWPS (word problems)
- MathQA / AQuA (multiple-choice / algebra)

Key metrics:
- Exact-match accuracy on final answers
- Step-by-step reasoning quality (manual or rubric-based)
- Robustness to phrasing and small perturbations

Quick evaluation snippet (conceptual):

```python
from datasets import load_dataset

# load a dataset (example)
gsm8k = load_dataset("gsm8k", "main", split="test")

for item in gsm8k:
    prompt = item["question"]
    # get adapter-loaded model and generate
    prediction = model.generate(prompt)
    # compute exact-match, log results
```

Datasets are available on Hugging Face (search by name) or via academic mirrors.

### Evaluation Metrics

- **Exact Match Accuracy**: Percentage of problems where final answer matches ground truth
- **Step-by-Step Quality**: Clarity and correctness of intermediate reasoning steps
- **Robustness**: Performance on problem variations and edge cases
- **Perplexity**: Language model confidence (lower is better)

### Running Evaluations

```python
# Example evaluation script structure
from datasets import load_dataset

# Load benchmark
gsm8k = load_dataset("gsm8k", "main", split="test")

# Evaluate your fine-tuned model
for problem in gsm8k:
    prediction = model.generate(problem['question'])
    # Compare prediction with problem['answer']
    # Calculate metrics
```

### Benchmark Access

- **GSM8K**: `https://huggingface.co/datasets/gsm8k`
- **MATH**: `https://huggingface.co/datasets/competition_math`
- **SVAMP**: `https://huggingface.co/datasets/ChilleD/SVAMP`
- **MathQA**: `https://huggingface.co/datasets/math_qa`

## 5. Model comparison (summary)

These are example, approximate observations from short LoRA runs (10K MathX subset, LoRA rank=16, epochs=3). Use them as guidance only.

| Model | Size | Typical training time* | Use-case |
|-------|------:|----------------------:|---------|
| Qwen3-32B | 32B | longer | Best for strongest reasoning when cost/compute is available |
| Gemma3-27B | 27B | medium | Strong instruction-following and multilingual support |
| Llama4-Scout-17B | 17B | shorter | Good balance for educational tasks and lower VRAM |

*Training time depends on machine type (TPU/GPU), batch size, and parallelism.

*Training time depends on machine type (TPU/GPU), batch size, and parallelism.

## 6. Fine-Tuning Results Comparison on Math Dataset

| **Aspect** | **LLaMA-4-Scout-17B-16E-Instruct** | **Gemma-3-27B-IT** | **Qwen-3-32B** |
|:------------|:----------------------------------:|:------------------:|:---------------:|
| **Initial Train Loss** | ~1.3 | ~0.85 | ~0.50 |
| **Final Train Loss** | ~0.55 | ~0.38 | ~0.33 |
| **Initial Validation (Eval) Loss** | ~0.73 | ~0.82 | ~0.50 |
| **Final Validation (Eval) Loss** | ~0.58 | ~0.41 | ~0.37 |
| **Convergence Trend** | Gradual, stable decline | Smooth and consistent | Smooth but with mild noise |
| **Training Stability** | Moderate | **Excellent** | Good (minor oscillations) |
| **Validation Gap (Overfitting)** | Slight gap | **Minimal gap** | Minimal gap |
| **Generalization Ability** | Medium | **High** | High |
| **Optimization Behavior** | Needs slight LR tuning | **Balanced and optimal** | Steady, slower convergence |
| **Overall Fine-Tuning Quality** | ⭐⭐☆ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Best Use Case** | When smaller VRAM or faster fine-tuning is needed | **For highest accuracy and stability on math reasoning** | For large-scale deployment with strong baseline |

---

## 🧠 Insights

- **Gemma-3-27B-IT** shows the **best trade-off** between training and evaluation loss, indicating strong generalization and effective learning from the math dataset.  
- **Qwen-3-32B** performs nearly as well, with slightly slower improvement due to its larger parameter count and strong base pretraining.  
- **LLaMA-4-Scout-17B** converges well but could benefit from **longer training or a lower learning rate**.

---

## 🏁 Final Ranking (Based on Loss Metrics)

1. 🥇 **Gemma-3-27B-IT** → Best overall performance  
2. 🥈 **Qwen-3-32B** → Strong, slightly noisier convergence  
3. 🥉 **LLaMA-4-Scout-17B** → Good baseline, needs minor tuning
