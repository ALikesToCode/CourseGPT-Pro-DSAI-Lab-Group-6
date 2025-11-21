# Math Agent Architecture - Gemma-3-27B Fine-Tuning

## 📋 Executive Summary

This document presents the architecture and design decisions for the **Math Agent** component of CourseGPT-Pro. The agent is designed to solve complex mathematical problems with step-by-step reasoning, providing clear explanations and accurate solutions for educational purposes.

## 🎯 Architecture Overview

### Selected Architecture: **Gemma-3-27B with LoRA Fine-Tuning**

Our math agent leverages Google's Gemma-3-4B-IT model fine-tuned on the MathX-5M dataset using Low-Rank Adaptation (LoRA). This architecture was chosen after careful evaluation of multiple alternatives based on performance, resource efficiency, and educational applicability.

### Core Components

1. **Base Model**: `google/gemma-3-27b-it` (27 billion parameter instruction-tuned model)
2. **Training Dataset**: `XenArcAI/MathX-5M` (~4.32M mathematical problems with solutions)
3. **Fine-Tuning Method**: LoRA (Low-Rank Adaptation) for parameter-efficient training
4. **Optimization**: Mixed precision training (BF16/FP16) with gradient checkpointing

### Key Architecture Specifications

| Component              | Configuration              | Justification                                        |
| ---------------------- | -------------------------- | ---------------------------------------------------- |
| **Model Size**         | 27B parameters             | High-capacity model chosen for superior reasoning depth |
| **Fine-Tuning Method** | LoRA (Low-Rank Adaptation) | Parameter-efficient; only 0.5% of weights trained    |
| **LoRA Rank**          | r=16, α=32                 | Sufficient expressiveness for math reasoning tasks   |
| **Target Modules**     | All attention & FFN layers | Comprehensive adaptation across model architecture   |
| **Sequence Length**    | 2048 tokens                | Accommodates complex multi-step problem solving      |
| **Mixed Precision**    | BF16/FP16                  | Faster training with reduced memory usage            |

## 🏗️ Architecture Justification

### 1. Why Gemma-3-27B Over Alternatives?

#### **Evaluated Alternatives:**

| Model            | Size  | Pros                                    | Cons                                       | Why Not Selected                         |
| ---------------- | ----- | --------------------------------------- | ------------------------------------------ | ---------------------------------------- |
| **GPT-3.5/4**    | Large | Excellent performance                   | Proprietary, API costs, data privacy       | ❌ Not suitable for academic deployment   |
| **Llama-3.1-8B** | 8B    | Strong general reasoning                | Higher resource requirements               | ❌ Exceeds GPU constraints (>24GB VRAM)   |
| **Mistral-7B**   | 7B    | Good generalization                     | Less optimized for instruction following   | ❌ Suboptimal instruction adherence       |
| **Phi-3-Mini**   | 3.8B  | Efficient, fast                         | Limited mathematical reasoning depth       | ❌ Insufficient for complex math problems |
| **Gemma-2-2B**   | 2B    | Very lightweight                        | Reduced capability on multi-step reasoning | ❌ Too small for educational requirements |
| **Gemma-3-27B**   | 27B   | ✅ **High reasoning capacity & instruction-following** | Larger resource requirements | ✅ **SELECTED** — best performance trade-off |

#### **Selection Rationale:**

✅ **Performance**: Gemma-3-4B demonstrates strong mathematical reasoning capabilities while maintaining compact size  
✅ **Efficiency**: Fits within typical GPU constraints (12-16GB VRAM) with quantization  
✅ **Instruction Following**: Built-in instruction tuning ("IT" variant) provides superior prompt adherence  
✅ **Licensing**: Apache 2.0 license allows academic and commercial use without restrictions  
✅ **Architecture**: Transformer-based with proven attention mechanisms for sequential reasoning  
✅ **Community Support**: Google-backed with active development and documentation  

### 2. Why LoRA Over Alternative Training Methods?

#### **Evaluated Alternatives:**

| Method                 | Memory Efficiency    | Training Speed | Quality       | Why Not Selected                                    |
| ---------------------- | -------------------- | -------------- | ------------- | --------------------------------------------------- |
| **Full Fine-Tuning**   | Low (4×)             | Slow           | Highest       | ❌ Requires 64GB+ VRAM, trains all 4B parameters     |
| **Prompt Engineering** | N/A (no training)    | N/A            | Variable      | ❌ Limited domain adaptation, inconsistent           |
| **Adapter Layers**     | Medium (2×)          | Medium         | Good          | ❌ More parameters than LoRA, less efficient         |
| **LoRA**               | **High (efficient)** | **Fast**       | **Very Good** | ✅ **SELECTED** - Best efficiency/quality ratio      |
| **Prefix Tuning**      | High                 | Fast           | Moderate      | ❌ Lower quality on reasoning tasks                  |
| **BitFit**             | Very High            | Very Fast      | Lower         | ❌ Only tunes bias terms; insufficient for reasoning |

#### **LoRA Justification:**

✅ **Parameter Efficiency**: Only trains 0.5-1% of model parameters (LoRA adapters), reducing overfitting risk  
✅ **Memory Efficiency**: Significantly lower memory footprint compared to full fine-tuning  
✅ **Performance Preservation**: Maintains 95-99% of full fine-tuning quality with fraction of parameters  
✅ **Training Speed**: Faster convergence due to smaller parameter space  
✅ **Modularity**: LoRA adapters can be easily swapped or combined for different tasks  
✅ **Gradient Checkpointing**: Combined with memory optimization techniques for efficient training  
✅ **Practical Deployment**: Lightweight adapters (~50MB) enable easy distribution and updates  

### 3. Why MathX-5M Dataset Over Alternatives?

#### **Evaluated Alternatives:**

| Dataset        | Size      | Difficulty Range    | Solution Format                      | Why Not Selected                               |
| -------------- | --------- | ------------------- | ------------------------------------ | ---------------------------------------------- |
| **GSM8K**      | 8K        | Elementary-Middle   | Text only                            | ❌ Too small, limited complexity                |
| **MATH**       | 12K       | High school-College | LaTeX + Text                         | ❌ Too advanced, limited scale                  |
| **MetaMathQA** | 395K      | K-12                | Synthetic reasoning                  | ❌ Quality concerns with synthetic data         |
| **Orca-Math**  | 200K      | Middle-High school  | Step-by-step                         | ❌ Smaller scale, less diverse                  |
| **MathX-5M**   | **4.32M** | **K-12 to College** | **Step-by-step with `<think>` tags** | ✅ **SELECTED** - Comprehensive scale & quality |

#### **MathX-5M Justification:**

✅ **Scale**: 4.32M examples provide sufficient data for robust fine-tuning without overfitting  
✅ **Diversity**: Covers arithmetic, algebra, geometry, calculus, statistics, and word problems  
✅ **Solution Quality**: Each problem includes expected answer AND step-by-step reasoning process  
✅ **Educational Format**: `<think>` tags explicitly separate reasoning from final answer (pedagogical value)  
✅ **Difficulty Progression**: Ranges from elementary to college level, matching CourseGPT-Pro's target audience  
✅ **LaTeX Support**: Mathematical notation properly formatted for academic standards  

### 4. LoRA Configuration Justification

**Adapters / Fine-tuned Backbones:** LoRA adapters and evaluation checkpoints were produced for multiple backbones to compare trade-offs and deployment options: `Gemma-3-27B`, `Qwen3-32B`, and `Llama4-17n`.

#### **Target Modules Selection**

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                  "gate_proj", "up_proj", "down_proj"]      # Feed-forward layers
```

**Rationale:**
- **Attention Layers** (q/k/v/o_proj): Critical for capturing mathematical relationships and dependencies
- **Feed-Forward Layers** (gate/up/down_proj): Essential for complex reasoning and computation
- **Comprehensive Coverage**: Adapting both components ensures holistic model adaptation

#### **Hyperparameter Selection**

| Parameter         | Value    | Alternative Values | Justification                                                                                |
| ----------------- | -------- | ------------------ | -------------------------------------------------------------------------------------------- |
| **Rank (r)**      | 16       | 4, 8, 32, 64       | r=16 balances expressiveness and efficiency; r<16 underfits, r>16 offers diminishing returns |
| **Alpha (α)**     | 32       | 16, 64, 128        | α=2r is standard scaling; maintains stable gradients during training                         |
| **Dropout**       | 0.05     | 0.0, 0.1, 0.2      | Low dropout preserves learned patterns while preventing overfitting on math tasks            |
| **Learning Rate** | 2e-4     | 1e-4, 5e-4, 1e-3   | Conservative rate prevents catastrophic forgetting of base model knowledge                   |
| **Batch Size**    | 16 (2×8) | 8, 32, 64          | Optimal for GPU memory; gradient accumulation simulates larger batches                       |

### 5. Training Strategy Justification

### Memory Optimization Techniques

| Technique                       | Memory Saved                   | Quality Impact | Rationale                                    |
| ------------------------------- | ------------------------------ | -------------- | -------------------------------------------- |
| **LoRA Adapters**               | 99.5% parameters frozen        | Minimal (<5%)  | Only train small adapter matrices            |
| **Gradient Checkpointing**      | 40-50%                         | None           | Trades compute for memory                    |
| **Mixed Precision (BF16/FP16)** | 50% (activations)              | None           | Faster computation, lower memory             |
| **Gradient Accumulation**       | Enables larger effective batch | None           | Simulates larger batches without memory cost |

**Combined Effect**: These techniques enable efficient fine-tuning on GPUs with 16-24GB VRAM

#### **Data Loading Strategy**

```python
# Streaming mode instead of full dataset loading
streamed_dataset = load_dataset("XenArcAI/MathX-5M", split="train", streaming=True)
subset = list(islice(streamed_dataset, 10000))  # Materialize subset
```

**Rationale:**
- **Memory Efficiency**: Avoid loading 4.32M examples (>50GB) into RAM
- **Flexibility**: Easy to adjust training subset size based on compute budget
- **Reproducibility**: Deterministic subset selection ensures consistent experiments

### 6. Instruction Format Design

#### **Chat Template Structure**

```python
messages = [
    {"role": "system", "content": "You are an expert mathematics tutor..."},
    {"role": "user", "content": "<problem>"},
    {"role": "assistant", "content": "<step-by-step solution>"}
]
```

**Design Rationale:**
- **System Prompt**: Establishes expert persona and behavior expectations (step-by-step, clear reasoning)
- **User Role**: Presents problem in natural language, mimicking student queries
- **Assistant Role**: Demonstrates complete solution path with reasoning exposed
- **Consistency**: Matches Gemma's pre-training format, leveraging existing instruction-following capabilities

#### **Why Not Alternative Formats?**

| Alternative Format        | Issues                                  | Our Choice Advantage               |
| ------------------------- | --------------------------------------- | ---------------------------------- |
| Simple Q&A pairs          | No reasoning context, black-box answers | ❌ Lacks pedagogical value          |
| Code-style formatting     | Not natural for math explanations       | ❌ Poor readability for students    |
| Pure completion           | No role separation, unclear context     | ❌ Reduced instruction clarity      |
| **Chat-based (selected)** | None for instruction-tuned models       | ✅ Natural, contextual, educational |

## � Complete Architecture Pipeline

### End-to-End Training Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     1. DATA PREPARATION LAYER                        │
├─────────────────────────────────────────────────────────────────────┤
│  • Load MathX-5M in streaming mode (memory-efficient)               │
│  • Normalize column names (question → problem)                      │
│  • Materialize subset (10k examples for demo, scalable to full)     │
│  • Apply Gemma chat template formatting                             │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     2. MODEL INITIALIZATION LAYER                    │
├─────────────────────────────────────────────────────────────────────┤
│  • Load Gemma-3-27B-IT base model                                   │
│  • Apply 4-bit NF4 quantization (75% memory reduction)              │
│  • Enable gradient checkpointing (40% additional savings)           │
│  • Prepare for k-bit training (PEFT integration)                    │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    3. LORA ADAPTATION LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│  • Configure LoRA: r=16, α=32, dropout=0.05                         │
│  • Target modules: All attention + FFN layers (7 modules)           │
│  • Trainable parameters: ~0.5% of total (highly efficient)          │
│  • Task type: CAUSAL_LM (next-token prediction)                     │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      4. TRAINING EXECUTION LAYER                     │
├─────────────────────────────────────────────────────────────────────┤
│  • Optimizer: Paged AdamW 8-bit (memory-efficient)                  │
│  • Mixed precision: BF16/FP16 (hardware-dependent)                  │
│  • Batch size: 2 per device × 8 accumulation = 16 effective        │
│  • Learning rate: 2e-4 with cosine schedule + 3% warmup             │
│  • Max sequence length: 2048 tokens (multi-step reasoning)          │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     5. MODEL PERSISTENCE LAYER                       │
├─────────────────────────────────────────────────────────────────────┤
│  • Save LoRA adapters (lightweight, ~50MB)                          │
│  • Optional: Merge adapters with base model                         │
│  • Save tokenizer configuration                                     │
│  • Generate deployment artifacts                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Diagram

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   MathX-5M   │────▶│  Tokenizer   │────▶│  Formatted   │
│   Dataset    │     │  (Gemma-3)   │     │   Dataset    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Merged    │◀────│     LoRA     │◀────│  Gemma-3-27B │
│    Model     │     │   Adapters   │     │ (Quantized)  │
└──────────────┘     └──────────────┘     └──────────────┘
       │                                           ▲
       │                                           │
       ▼                                    ┌──────────────┐
┌──────────────┐                            │   Training   │
│  Inference   │                            │   Process    │
│   Engine     │                            └──────────────┘
└──────────────┘
```

## 🎯 Architecture Decision Summary

### Core Decision Matrix

| Decision Point         | Selected Option           | Key Rationale                                                |
| ---------------------- | ------------------------- | ------------------------------------------------------------ |
| **Base Model**         | Gemma-3-27B-IT            | Optimal performance for complex math reasoning |
| **Training Method**    | LoRA                      | 99.5% parameter reduction; efficient fine-tuning             |
| **Dataset**            | MathX-5M                  | Largest scale; step-by-step reasoning format                 |
| **LoRA Configuration** | r=16, α=32, 7 modules     | Comprehensive adaptation; proven for reasoning tasks         |
| **Mixed Precision**    | BF16/FP16                 | Faster training; reduced memory usage                        |
| **Sequence Length**    | 2048 tokens               | Accommodates multi-step solutions                            |
| **Batch Strategy**     | 2×8 gradient accumulation | Fits GPU constraints; stable training                        |
| **Optimizer**          | AdamW                     | Standard optimizer; proven for LLM training                  |

## 🚀 Getting Started (Quick Reference)

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (12GB+ VRAM recommended)
- Hugging Face account & token

### Installation

```bash
pip install -q -U transformers datasets accelerate peft trl bitsandbytes torch
```

### Environment Setup

```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

### Training Execution

Run cells sequentially in `math_agent_architecture_gemma_3_27b.ipynb`:
1. Install dependencies
2. Load & explore dataset
3. Configure model & training
4. Execute training
5. Save & test model

### Expected Outputs

- **LoRA Adapters**: `./gemma3-27b-math-lora-adapter/final_adapter/` (~50MB per-adapter)
- **Merged Model**: `./gemma3-27b-math-merged/` (optional merged checkpoint)
- **Training Logs**: Console output with loss, steps, time

## 📚 References & Related Work

### Primary Sources

- **Gemma Model**: [Google Gemma Model Card](https://huggingface.co/google/gemma-3-4b-it)
- **MathX-5M Dataset**: [XenArcAI MathX-5M on Hugging Face](https://huggingface.co/datasets/XenArcAI/MathX-5M)
- **QLoRA Paper**: [Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)](https://arxiv.org/abs/2305.14314)
- **LoRA Paper**: [Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)](https://arxiv.org/abs/2106.09685)

### Technical Libraries

- **PEFT (Parameter-Efficient Fine-Tuning)**: [Hugging Face PEFT](https://github.com/huggingface/peft)
- **TRL (Transformer Reinforcement Learning)**: [Hugging Face TRL](https://github.com/huggingface/trl)
- **BitsAndBytes**: [8-bit & 4-bit Quantization Library](https://github.com/TimDettmers/bitsandbytes)
- **Transformers**: [Hugging Face Transformers](https://github.com/huggingface/transformers)

### Comparative Studies

- **Math Reasoning Benchmarks**: GSM8K, MATH, MathQA
- **Alternative Models**: Llama-3, Mistral, Phi-3, GPT-4
- **Training Methods**: Full fine-tuning, Adapter layers, Prompt tuning, QLoRA

## 📄 License & Usage

### Model Licensing

- **Gemma Models**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms) - Apache 2.0 compatible
- **MathX-5M Dataset**: Check [dataset card](https://huggingface.co/datasets/XenArcAI/MathX-5M) for specific license
