# Math Agent - Gemma-3-4B Fine-Tuning

This directory contains the implementation for fine-tuning Google's Gemma-3-4B model on mathematical reasoning tasks using the MathX-5M dataset with QLoRA (Quantized Low-Rank Adaptation).

## 📋 Overview

The math agent is designed to solve mathematical problems step-by-step, providing clear reasoning and accurate solutions. The model is fine-tuned using efficient parameter-efficient techniques to achieve strong performance on mathematical reasoning tasks.

### Key Features

- **Model**: `google/gemma-3-4b-it` (4B parameter instruction-tuned model)
- **Dataset**: `XenArcAI/MathX-5M` (~4.32M mathematical problems with solutions)
- **Training Technique**: QLoRA with 4-bit quantization
- **LoRA Configuration**: 
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Dropout: 0.05

## 🗂️ Dataset: XenArcAI/MathX-5M

The MathX-5M dataset is a large-scale mathematical reasoning dataset containing approximately 4.32 million examples.

### Dataset Structure

Each example in the dataset contains:

| Column | Description |
|--------|-------------|
| `question` (renamed to `problem`) | Mathematical problem statement, often with LaTeX formatting |
| `expected_answer` | The final correct answer to the problem |
| `generated_solution` | Step-by-step reasoning process with `<think>` tags showing the solution path |

### Data Loading Strategy

The notebook uses streaming mode to efficiently load the dataset:

```python
# Load in streaming mode
streamed_dataset = load_dataset("XenArcAI/MathX-5M", split="train", streaming=True)

# Normalize column names
def unify_columns(ex):
    if "question" in ex:
        ex["problem"] = ex.pop("question")
    return ex

streamed_dataset = streamed_dataset.map(unify_columns)

# Materialize subset (10k examples for demo)
subset = list(islice(streamed_dataset, 10000))
dataset = Dataset.from_list(subset).select_columns([
    "problem", "generated_solution", "expected_answer"
])
```

### Example Data Point

**Problem:**
```
Solve: If 3x + 7 = 22, what is the value of x?
```

**Expected Answer:**
```
x = 5
```

**Generated Solution:**
```
<think>
Step 1: Start with the equation 3x + 7 = 22
Step 2: Subtract 7 from both sides: 3x = 15
Step 3: Divide both sides by 3: x = 5
</think>
Therefore, x = 5
```

## 📁 Files

- `math_agent_architecture_gemma_3_4b.ipynb` - Main training notebook with complete fine-tuning pipeline
- `README.md` - This documentation file

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- Hugging Face account and token (for accessing Gemma models)

### Installation

1. Install required packages:

```bash
pip install -q -U transformers datasets accelerate peft trl bitsandbytes
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

2. Set up your Hugging Face token:

```python
import os
os.environ["HF_TOKEN"] = "your_huggingface_token_here"
```

### Training Pipeline

The notebook follows this complete pipeline:

1. **Environment Setup** - Install dependencies and verify CUDA availability
2. **Configuration** - Set hyperparameters and paths
3. **Dataset Loading** - Load and preprocess MathX-5M dataset in streaming mode
4. **Tokenizer Setup** - Load Gemma-3 tokenizer and configure padding
5. **Data Formatting** - Format dataset using Gemma's chat template
6. **Quantization Config** - Set up 4-bit quantization with BitsAndBytes
7. **Model Loading** - Load Gemma-3-4B with quantization
8. **LoRA Configuration** - Configure Low-Rank Adaptation parameters
9. **Training Arguments** - Set training hyperparameters
10. **Trainer Initialization** - Initialize SFTTrainer with PEFT
11. **Training** - Fine-tune the model
12. **Save Adapters** - Save LoRA adapters
13. **Merge Model** - Merge adapters with base model (optional)
14. **Testing** - Test the fine-tuned model on sample problems
15. **Evaluation** - Evaluate performance

## ⚙️ Configuration

### Key Hyperparameters

```python
MODEL_NAME = "google/gemma-3-4b-it"
DATASET_NAME = "XenArcAI/MathX-5M"

# Training settings
MAX_SEQ_LENGTH = 2048
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 16
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 1

# LoRA settings
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
```

### Memory Optimization Techniques

- **4-bit Quantization**: NormalFloat 4-bit quantization with double quantization
- **Gradient Checkpointing**: Reduces memory usage during backpropagation
- **Paged AdamW 8-bit**: Memory-efficient optimizer
- **Mixed Precision**: BF16 or FP16 depending on GPU capability

## 📊 Training Details

### Instruction Format

The dataset is formatted using Gemma's chat template:

```python
messages = [
    {
        "role": "system",
        "content": "You are an expert mathematics tutor. Solve problems step-by-step, showing your reasoning clearly."
    },
    {
        "role": "user",
        "content": "<problem_text>"
    },
    {
        "role": "assistant",
        "content": "<solution_with_reasoning>"
    }
]
```

### Output Structure

After training, the following artifacts are generated:

```
./gemma3-4b-math-lora-adapter/
├── final_adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files

./gemma3-4b-math-merged/  (if merged)
├── model files
└── tokenizer files
```

## 🧪 Testing

Test the fine-tuned model on a sample problem:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load merged model
model = AutoModelForCausalLM.from_pretrained(
    "./gemma3-4b-math-merged",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("./gemma3-4b-math-merged")

# Test problem
test_problem = """Solve: A rectangle has a length that is 3 units longer 
than twice its width. If the perimeter is 54 units, find the dimensions."""

# Generate solution
messages = [
    {"role": "system", "content": "You are an expert mathematics tutor..."},
    {"role": "user", "content": test_problem}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📈 Performance Considerations

### GPU Memory Requirements

- **Training (with QLoRA)**: ~12-16GB VRAM
- **Inference (quantized)**: ~4-6GB VRAM
- **Inference (full precision)**: ~16GB VRAM

### Training Time Estimates

On a single GPU:
- **10k examples**: ~1-2 hours
- **100k examples**: ~10-15 hours
- **Full dataset (4.32M)**: ~3-5 days

### Optimization Tips

1. **Adjust batch size** based on GPU memory
2. **Enable packing** for shorter sequences to improve efficiency
3. **Use gradient accumulation** to simulate larger batch sizes
4. **Monitor GPU utilization** and adjust num_proc for data loading
5. **Use streaming mode** to avoid loading entire dataset into memory

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `PER_DEVICE_BATCH_SIZE`
   - Increase `GRADIENT_ACCUMULATION_STEPS`
   - Reduce `MAX_SEQ_LENGTH`

2. **Column Mismatch Error**
   - Ensure column names are normalized (question → problem)
   - Use the provided `unify_columns` function

3. **Tokenizer Not Defined in Multiprocessing**
   - Remove `num_proc` parameter from `dataset.map()`
   - Or use single-process mapping as shown in the notebook

4. **Slow Dataset Loading**
   - Use streaming mode for large datasets
   - Materialize only the subset you need for training

## 📚 References

- [Gemma Model Card](https://huggingface.co/google/gemma-3-4b-it)
- [MathX-5M Dataset](https://huggingface.co/datasets/XenArcAI/MathX-5M)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Library](https://github.com/huggingface/peft)
- [TRL Library](https://github.com/huggingface/trl)

## 🎯 Next Steps

1. **Evaluate** on validation/test sets
2. **Fine-tune hyperparameters** for better performance
3. **Train on larger subset** or full dataset
4. **Deploy** as a math reasoning agent
5. **Integrate** with CourseGPT-Pro system
6. **Benchmark** against other math reasoning models

## 📝 Notes

- The notebook uses **1% of the dataset (~10k examples)** for demonstration
- Remove `[:1%]` or adjust the `islice` parameter to train on more data
- Training on the full dataset requires significant GPU resources
- Consider using distributed training for large-scale experiments

## 📄 License

This project follows the licensing terms of the base models and datasets used:
- Gemma models: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- MathX-5M dataset: Check dataset card for license information
