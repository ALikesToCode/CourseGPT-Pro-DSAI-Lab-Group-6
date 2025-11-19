# Code Fine-Tuning Notebook

This notebook demonstrates fine-tuning the Meta-Llama-3.1-8B-Instruct model for code generation tasks using QLoRA (Quantized Low-Rank Adaptation) and the OpenCoder-LLM/opc-sft-stage2 dataset.

## Overview

The notebook performs the following steps:
1. Loads and explores the educational instruct dataset from OpenCoder-LLM.
2. Formats the dataset using the Llama 3.1 chat template.
3. Saves the formatted data to a JSONL file.
4. Loads the pre-trained Llama 3.1 model with 4-bit quantization.
5. Configures LoRA for efficient fine-tuning.
6. Trains the model using Supervised Fine-Tuning (SFT).
7. Saves the trained LoRA adapter.

## Dataset Overview

The notebook uses the **OpenCoder-LLM/opc-sft-stage2** dataset, specifically the `educational_instruct` subset.

### Dataset Details
- **Name**: OpenCoder-LLM/opc-sft-stage2
- **Subset Used**: educational_instruct
- **Size**: ~118K rows (educational_instruct subset)
- **Total Dataset Size**: 436,347 rows across all subsets
- **Format**: Parquet
- **License**: MIT
- **ArXiv Paper**: [2411.04905](https://arxiv.org/abs/2411.04905)

### Dataset Composition
The opc-sft-stage2 dataset consists of four parts:
- **educational_instruct**: Generated (instruction, code, test case) triples using algorithmic corpus as seed, validated through Python compiler
- **evol_instruct**: Based on MagicCoder-Evol-Instruct-110k
- **mceval_instruct**: Based on McEval-Instruct
- **package_instruct**: Generated from Python package documentation

### Data Features (educational_instruct)
- **instruction**: Programming problem or task description
- **output**: Complete code solution
- **test_cases**: Unit tests for validation (provides valuable signal for code RL)

### Usage in Fine-Tuning
The dataset provides high-quality instruction-response pairs specifically designed for code generation tasks, with built-in test cases that enable reinforcement learning from code execution feedback.

## Prerequisites

- Python 3.8+
- Hugging Face account with access to Meta-Llama models
- HF_TOKEN environment variable set with your Hugging Face token
- GPU with CUDA support (recommended: RTX 4080 or similar for BF16 support)
- At least 16GB VRAM for 4-bit quantized training

## Required Libraries

Install the dependencies using pip:

```bash
pip install transformers datasets accelerate peft trl bitsandbytes torch
```

## Setup

1. Clone or download the repository.
2. Set your Hugging Face token as an environment variable:
   ```bash
   export HF_TOKEN=your_hugging_face_token_here
   ```
3. Ensure you have access to the Meta-Llama-3.1-8B-Instruct model on Hugging Face.

## Running the Notebook

1. Open the `finetunening.ipynb` notebook in Jupyter or VS Code.
2. Run the cells in order:
   - Load and format the dataset
   - Configure model and tokenizer
   - Set up training arguments
   - Train the model
   - Save the adapter

## Configuration

Key parameters you can adjust:
- `DATASET_NAME`: Dataset to use (default: "OpenCoder-LLM/opc-sft-stage2")
- `MODEL_NAME`: Model to fine-tune (default: "meta-llama/Meta-Llama-3.1-8B-Instruct")
- `OUTPUT_DIR`: Directory to save outputs (default: "./llama31_code_lora_adapter")
- Training hyperparameters in `SFTConfig` (batch size, learning rate, epochs, etc.)

## Output

The notebook produces:
- `llama31_code_finetune_simple.jsonl`: Formatted training data
- `./llama31_code_lora_adapter/final_adapter/`: Trained LoRA adapter and tokenizer

## Notes

- The notebook uses 4-bit quantization to reduce memory usage.
- Gradient checkpointing is enabled for additional memory efficiency.
- Training is configured for a single epoch; adjust as needed for your use case.
- The model is fine-tuned specifically for code generation tasks using educational instruct data.

## Troubleshooting

- If you encounter CUDA out of memory errors, reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps`.
- Ensure your GPU supports BF16; otherwise, the notebook falls back to FP16.
- For Windows users, adjust `SAFE_WINDOWS_WORKERS` if you encounter multiprocessing issues.
