# CourseGPT Pro - Model Architecture & Training

This document provides detailed information about the custom fine-tuned models developed in Milestones 3 and 4.

---

## Table of Contents

1. [Model Architecture Overview](#1-model-architecture-overview)
2. [Router Agent Models](#2-router-agent-models)
3. [Math Agent Models](#3-math-agent-models)
4. [Code Agent Models](#4-code-agent-models)
5. [Training Datasets](#5-training-datasets)
6. [Training Methodology](#6-training-methodology)
7. [Evaluation Results](#7-evaluation-results)
8. [Model Deployment](#8-model-deployment)
9. [Integration Status](#9-integration-status)

---

## 1. Model Architecture Overview

CourseGPT Pro uses **specialized fine-tuned models** for each agent, trained via Parameter-Efficient Fine-Tuning (PEFT) methods:

### Model Summary Table

| Agent | Base Model | Parameters | Adapter Method | Trained In | Best Variant |
|-------|-----------|------------|----------------|-----------|--------------|
| **Router** | Llama 3.1 8B | 7.4B | LoRA | Milestone-3 | ✅ Gemma 3 27B |
| **Router** | Gemma 3 27B | 25.6B | LoRA | Milestone-3 | ⭐ **Best** |
| **Router** | Qwen3 32B | 31.2B | LoRA | Milestone-3 | ✅ Strong |
| **Math** | Gemma 3 4B | 4B | QLoRA | Milestone-3 | ✅ Good |
| **Math** | Gemma 3 27B | 25.6B | PEFT | Milestone-4 | ⭐ **Best** |
| **Math** | Qwen3 32B | 31.2B | PEFT | Milestone-4 | ✅ Strong |
| **Math** | Llama4 Scout 17B | 17B | PEFT | Milestone-4 | ⚠️ Weaker |
| **Code** | Qwen 0.6B | 0.6B | QLoRA | Milestone-3 | ⭐ **Completed** |
| **Code** | Llama 3.1 8B | 7.4B | LoRA | Milestone-3 | ⚠️ Incomplete |

**Legend:**
- ⭐ Best performing or production-ready
- ✅ Successfully trained with good results
- ⚠️ Issues or incomplete

---

## 2. Router Agent Models

### 2.1 Purpose

The Router Agent analyzes user queries and routes them to the appropriate specialized agent (Code, Math, or General).

### 2.2 Architecture Details

#### Llama 3.1 8B Instruct + LoRA

**Base Model:**
- Architecture: 32-layer decoder-only transformer
- Attention: Grouped Query Attention (GQA) with 32 query heads, 8 KV heads
- Hidden Size: 4096
- Vocabulary: 128,256 tokens
- Context Window: 128K tokens

**LoRA Configuration:**
- Rank: 16
- Alpha: 32
- Target Modules: All attention projections + MLP layers
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - `gate_proj`, `up_proj`, `down_proj`
- Dropout: 0.05
- Trainable Parameters: ~50M (~0.7% of total)

#### Gemma 3 27B IT + LoRA

**Base Model:**
- Architecture: ~48-layer decoder (PaLM-derived)
- Attention: Multi-Query Attention variant
- Parameters: ~25.6B active
- Tokenizer: SentencePiece (PaLM tokenizer)
- Context Window: 128K tokens

**LoRA Configuration:**
- Rank: 16
- Alpha: 32
- Target Modules: All linear projections
- Dropout: 0.05
- Trainable Parameters: ~65M (~0.25% of total)

#### Qwen3 32B + LoRA

**Base Model:**
- Architecture: 64-layer decoder with native thinking mode
- Attention: GQA (64 query heads, 8 KV heads)
- Parameters: 31.2B
- Context Window: 32K base (131K with YaRN extension)
- Special Features: Built-in `<think>` token support

**LoRA Configuration:**
- Rank: 16
- Alpha: 32
- Target Modules: All attention + MLP
- Dropout: 0.05
- Trainable Parameters: ~70M (~0.22% of total)

### 2.3 Training Configuration

All router models trained via **Google Vertex AI Supervised Fine-Tuning**:

```python
tuning_config = {
    "adapter_size": 16,  # LoRA rank
    "learning_rate_multiplier": 0.7,
    "epochs": 3,
    "warmup_ratio": 0.1,
}
```

**Data Split:**
- Training: 6,962 examples (85%)
- Validation: 818 examples (10%)
- Test: 409 examples (5%)

**Computational Requirements:**
- Platform: Vertex AI (managed GPU infrastructure)
- Training Time: 3-4 hours per model
- Cost: ~$15-25 per model (Vertex AI pricing)

### 2.4 Router Dataset Format

**Source:** Custom-generated routing traces (8,189 examples)

**Example:**
```json
{
  "prompt": "I need help debugging my Python code. It's raising a TypeError.",
  "completion": {
    "route_plan": ["/code"],
    "route_rationale": "Query mentions 'debugging' and 'Python code' with a specific error (TypeError), indicating a programming assistance request.",
    "thinking_outline": "User has a code error → needs debugging help → route to Code Agent",
    "handoff_plan": "Direct to code_agent with error context",
    "difficulty": "medium",
    "tags": ["debugging", "python", "error-handling"]
  }
}
```

### 2.5 Training Results

| Model | Initial Loss | Final Train Loss | Final Eval Loss | Perplexity | BLEU Score | Training Time |
|-------|--------------|------------------|-----------------|------------|------------|---------------|
| Llama 3.1 8B | 1.15 | 0.42 | **0.6758** | 1.97 | 0.4004 | ~3h |
| **Gemma 3 27B** | 1.08 | 0.38 | **0.6080** | **1.84** | - | ~4h |
| Qwen3 32B | 0.95 | 0.40 | **0.6277** | 1.87 | - | ~3.5h |

**Winner:** Gemma 3 27B (lowest eval loss and perplexity)

---

## 3. Math Agent Models

### 3.1 Purpose

The Math Agent solves mathematical problems with step-by-step explanations and LaTeX formatting.

### 3.2 Milestone-3 Model: Gemma 3 4B + QLoRA

**Base Model:**
- Architecture: Gemma 3 4B Instruct (compact instruction-tuned variant)
- Parameters: ~3.9B
- Context Window: 8K tokens

**QLoRA Configuration:**
- Method: 4-bit NF4 quantization + LoRA
- Quantization:
  - Type: NF4 (Normal Float 4-bit)
  - Compute dtype: BFloat16
  - Double quantization: Enabled
- LoRA:
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
  - Target Modules: All projections

**Training Setup:**
- Framework: Transformers + BitsAndBytes + PEFT
- Optimizer: AdamW (paged_adamw_8bit)
- Batch Size: 4 (effective batch size 16 with gradient accumulation)
- Learning Rate: 2e-4
- Scheduler: Cosine with warmup
- Gradient Checkpointing: Enabled
- Flash Attention: SDPA backend

**Results:**
```
Initial Training Loss: 2.1
Final Training Loss: 0.37
Evaluation Accuracy: 90.4%
Training Time: ~2.5 hours (RTX 4080)
```

### 3.3 Milestone-4 Models: Vertex AI PEFT

Three models tested on **MathX-5M** subset (10K examples):

#### Gemma 3 27B IT + PEFT (Best)

**Training Results:**
```
Initial Train Loss: 0.85
Final Train Loss: 0.38
Final Eval Loss: 0.41
Quality: ⭐⭐⭐ (Best)
```

**Strengths:**
- Most stable training curve
- Best balance of loss and perplexity
- Excellent LaTeX formatting in outputs

#### Qwen3 32B + PEFT (Strong Alternative)

**Training Results:**
```
Initial Train Loss: 0.50
Final Train Loss: 0.33
Final Eval Loss: 0.37
Quality: ⭐⭐⭐ (Excellent)
```

**Strengths:**
- Lowest final losses
- Strong reasoning with `<think>` tokens
- Good multi-step explanations

#### Llama4 Scout 17B + PEFT (Weaker)

**Training Results:**
```
Initial Train Loss: 1.3
Final Train Loss: 0.55
Final Eval Loss: 0.58
Quality: ⭐⭐☆ (Acceptable)
```

**Challenges:**
- Higher final losses
- Less consistent formatting
- Occasional reasoning errors

### 3.4 Math Dataset: MathX-5M

**Source:** `XenArcAI/MathX-5M`

**Statistics:**
- Total Size: ~4.32M problems
- Used for Tuning: 10K subset
- Split: 9K train / 1K validation

**Content Coverage:**
- Arithmetic & number theory
- Algebra & equations
- Geometry & trigonometry
- Calculus & derivatives
- Statistics & probability
- Word problems

**Difficulty Range:** Elementary → College

**Format Example:**
```json
{
  "question": "Solve for x: 3x + 7 = 22",
  "generated_solution": "<think>Need to isolate x</think>\n3x + 7 = 22\n3x = 22 - 7\n3x = 15\nx = 15/3\nx = 5",
  "difficulty": "elementary",
  "topic": "algebra"
}
```

---

## 4. Code Agent Models

### 4.1 Purpose

The Code Agent assists with programming questions, debugging, and code examples.

### 4.2 Qwen 0.6B + QLoRA (Production)

**Base Model:**
- Architecture: Qwen 0.6B (code-optimized compact model)
- Parameters: ~600M
- Context Window: 32K tokens
- Special Features: Code-pretrained, fast inference

**QLoRA Configuration:**
- Quantization: 4-bit NF4
- Compute dtype: BFloat16
- LoRA rank: 16
- Alpha: 32
- Target Modules: All linear projections

**Training Setup:**
- Dataset: `OpenCoder-LLM/opc-sft-stage2` (educational_instruct)
- Data Size: 119 parquet shards (~92M tokens)
- Template: Llama 3 chat format
- System Prompt: "You are an expert programming assistant"
- Batch Size: 4
- Learning Rate: 2e-4
- Hardware: RTX 4080

**Training Results:**
```
Initial Loss: 2.70
Final Loss: 0.40
Training Time: ~3h 45min
Model Size: ~400MB (quantized adapter)
```

**Output Quality:**
- ✅ Clean code generation
- ✅ Good explanations
- ✅ Proper syntax across languages
- ✅ Fast inference (<1s on GPU)

### 4.3 Code Dataset: OpenCoder SFT Stage 2

**Source:** `OpenCoder-LLM/opc-sft-stage2` (educational_instruct configuration)

**Format:**
```json
{
  "instruction": "Write a Python function to find the factorial of a number",
  "output": "Here's a clean implementation:\n\n```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n```",
  "code": "def factorial(n): ...",
  "entry_point": "factorial",
  "testcase": "assert factorial(5) == 120"
}
```

**Coverage:**
- Python, JavaScript, Java, C++, Go
- Algorithms & data structures
- Debugging patterns
- Best practices

---

## 5. Training Datasets

### Summary Table

| Agent | Dataset | Source | Size Used | Total Size | License |
|-------|---------|--------|-----------|------------|---------|
| Router | Custom Routing Traces | Self-generated | 8,189 | 8,189 | Custom |
| Math | MathX-5M | `XenArcAI/MathX-5M` | 10K | ~4.32M | MIT |
| Code | OpenCoder SFT Stage 2 | `OpenCoder-LLM/opc-sft-stage2` | 119 shards | ~92M tokens | Apache 2.0 |

### Data Preparation Pipeline

**Router Dataset:**
1. Generate synthetic queries with difficulty variations
2. Manually review and annotate routing decisions
3. Add metadata: tags, difficulty, acceptance criteria
4. Validate JSON schema compliance
5. Split: 85/10/5 train/val/test

**Math Dataset:**
1. Stream from Hugging Face Datasets
2. Apply Gemma chat template formatting
3. Filter by difficulty and topic coverage
4. Subsample to 10K for efficient tuning
5. Upload to Google Cloud Storage for Vertex AI

**Code Dataset:**
1. Load parquet shards incrementally
2. Apply Llama 3 chat template
3. Add system prompt: "You are an expert programming assistant"
4. Tokenize with truncation (max_length=2048)
5. Pack examples for efficient training

---

## 6. Training Methodology

### 6.1 Vertex AI Supervised Fine-Tuning (Router + Math)

**Platform:** Google Cloud Vertex AI

**Workflow:**
```
1. Prepare dataset (JSONL format)
   ↓
2. Upload to GCS bucket
   ↓
3. Launch tuning job via API/SDK
   ↓
4. Monitor training in console
   ↓
5. Download adapter from GCS
   ↓
6. Merge and publish to Hugging Face
```

**Configuration:**
```python
tuning_job = aiplatform.SupervisedTuningJob.create(
    base_model="meta/llama-3.1-8b-instruct",
    train_dataset=train_data_uri,
    validation_dataset=val_data_uri,
    tuned_model_output_uri=output_uri,
    adapter_config={
        "adapter_size": 16,  # LoRA rank
    },
    epochs=3,
    learning_rate_multiplier=0.7,
)
```

**Advantages:**
- Managed infrastructure (no GPU setup)
- Automatic checkpointing
- Built-in monitoring and logging
- Supports large models (up to 70B parameters)

**Cost:** ~$15-30 per tuning job (3 epochs on 8-32B models)

### 6.2 Local GPU Training (Code Agent)

**Platform:** Local RTX 4080 (16GB VRAM)

**Framework Stack:**
- `transformers` - Model loading and training
- `peft` - LoRA/QLoRA adapters
- `bitsandbytes` - 4-bit quantization
- `trl` - SFTTrainer for instruction tuning

**Memory Optimization:**
- 4-bit quantization: Reduces model footprint by ~75%
- Gradient checkpointing: Trades compute for memory
- Gradient accumulation: Simulate larger batch sizes
- Paged optimizers: Use CPU RAM overflow

**Example Code:**
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.6B",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=2048,
    args=training_args,
)
trainer.train()

# Save adapter
model.save_pretrained("./adapter")
```

### 6.3 Hyperparameter Selection

**Learning Rate:**
- Vertex AI: 0.7x multiplier on base LR (automatic)
- Local: 2e-4 (standard for LoRA)

**Batch Size:**
- Vertex AI: Automatic (depends on model size)
- Local: 4 (with grad accumulation = 16 effective)

**Epochs:**
- Router: 3 epochs (optimal for 8K examples)
- Math: 3 epochs (prevents overfitting)
- Code: 2 epochs (large dataset, early stopping)

**LoRA Rank:**
- Router: 16 (balance of capacity and efficiency)
- Math: 16
- Code: 16

**Alpha:**
- All models: 32 (2x rank, standard practice)

---

## 7. Evaluation Results

### 7.1 Router Agent Comparison

| Model | Eval Loss | Perplexity | Inference Speed | Memory | Recommendation |
|-------|-----------|------------|-----------------|--------|----------------|
| Llama 3.1 8B | 0.6758 | 1.97 | Fast | 8GB VRAM | Good baseline |
| **Gemma 3 27B** | **0.6080** | **1.84** | Medium | 24GB VRAM | **Production** |
| Qwen3 32B | 0.6277 | 1.87 | Slow | 32GB VRAM | Research/best accuracy |

**Decision:** Gemma 3 27B for production (best accuracy-speed tradeoff)

### 7.2 Math Agent Comparison

| Model (Milestone-4) | Train Loss | Eval Loss | Quality | LaTeX Support | Recommendation |
|---------------------|------------|-----------|---------|---------------|----------------|
| **Gemma 3 27B** | 0.38 | **0.41** | ⭐⭐⭐ | Excellent | **Production** |
| Qwen3 32B | 0.33 | **0.37** | ⭐⭐⭐ | Excellent | Alternative |
| Llama4 Scout 17B | 0.55 | 0.58 | ⭐⭐☆ | Good | Fallback |

**Decision:** Gemma 3 27B (best stability and formatting)

### 7.3 Code Agent

| Model | Status | Loss | Size | Speed | Recommendation |
|-------|--------|------|------|-------|----------------|
| **Qwen 0.6B** | ✅ Complete | 0.40 | 400MB | Very Fast | **Production** |
| Llama 3.1 8B | ⚠️ Incomplete | - | - | - | On hold |

**Decision:** Qwen 0.6B (compact, fast, production-ready)

---

## 8. Model Deployment

### 8.1 Hugging Face Repositories

**Published Models:**
```
CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft
CourseGPT-Pro-DSAI-Lab-Group-6/router-gemma3-peft
CourseGPT-Pro-DSAI-Lab-Group-6/router-qwen3-32b-peft
```

**Merged Models (for ZeroGPU):**
```
Alovestocode/router-llama31-merged
Alovestocode/router-gemma3-merged
Alovestocode/router-qwen3-32b-merged
```

### 8.2 Hugging Face Spaces

**Router Control Room:**
- URL: `Alovestocode/router-control-room-private`
- Backend: ZeroGPU (A100 on-demand)
- Interface: Gradio
- Features:
  - Model selector dropdown (Llama/Gemma/Qwen)
  - Streaming responses
  - Error handling with graceful fallbacks

**App Code:** `/Milestone-6/router-agent/hf_space/app.py`

### 8.3 Local Adapter Storage

**Router Adapters:**
```
/Milestone-3/router-agent-scripts/hf_artifacts/
├── llama31-peft/
├── gemma3-peft/
└── qwen3-32b-peft/
```

**Math Adapters:**
```
/Milestone-3/gemma3-4b-math-lora-adapter/final_adapter/
/Milestone-4/math-agent/adapters/
├── Qwen3-32B/adapter/
├── math-gemma-3-27b/adapter/
└── llama4-17b/adapter/
```

**Code Adapters:**
```
/Milestone-3/qwen_code_lora_adapter/final_adapter/
```

### 8.4 Loading Adapters in Code

**Example: Load Router Model**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    device_map="auto",
)

# Load adapter
model = PeftModel.from_pretrained(
    base_model,
    "CourseGPT-Pro-DSAI-Lab-Group-6/router-gemma3-peft",
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")

# Inference
inputs = tokenizer("Help me debug my code", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

---

## 9. Production Deployment

### 9.1 Model Loading

**Agent Implementation:**

Each agent loads its fine-tuned model using PEFT adapters:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    device_map="auto",
    load_in_8bit=True  # Optional quantization
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(
    base_model,
    "CourseGPT-Pro-DSAI-Lab-Group-6/router-gemma3-peft"
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")

# Create LangChain pipeline
pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipeline)
```

### 9.2 Production Configuration

**Recommended Deployment:**

**Router Agent:**
- Model: Gemma 3 27B + LoRA adapter
- Hosting: Hugging Face Inference API or dedicated GPU
- Memory: 24GB VRAM (16GB with quantization)
- Alternative: Qwen3 32B for complex reasoning tasks

**Math Agent:**
- Model: Gemma 3 27B + PEFT adapter
- Hosting: ZeroGPU Spaces or local GPU
- Memory: 24GB VRAM
- Alternative: Qwen3 32B for advanced problems

**Code Agent:**
- Model: Qwen 0.6B + QLoRA adapter
- Hosting: Local GPU or even CPU (small model)
- Memory: 2GB VRAM / 4GB RAM
- Latency: < 1 second per response

### 9.3 Performance Optimization

**Techniques Applied:**
- ✅ 8-bit quantization for reduced memory footprint
- ✅ Model caching to avoid reloading
- ✅ Batch processing for multiple requests
- ✅ Adapter-only loading (base models cached)
- ✅ Flash attention for faster inference

**Monitoring:**
- Response latency tracking
- Model memory usage
- Request throughput
- Error rates per agent

---

## Summary

✅ **Successfully Trained:** 8 model variants across 3 agent types
✅ **Best Models Identified:** Gemma 3 27B (Router, Math), Qwen 0.6B (Code)
✅ **Models Published:** Hugging Face repos with adapters
✅ **Production Ready:** Fine-tuned models deployed and serving requests

---

*Last Updated: 2025-01-19*
