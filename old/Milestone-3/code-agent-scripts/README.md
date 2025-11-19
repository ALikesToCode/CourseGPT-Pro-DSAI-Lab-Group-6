# Fine-tuning Language Models for Code Generation

This work involves fine-tuning language models for code generation tasks.

## Models

The following models were used in this project:

- **Qwen/Qwen3-0.6B**: This model was successfully fine-tuned.
- **meta-llama/Meta-Llama-3.1-8B-Instruct**: An attempt was made to fine-tune this model.

## Dataset

The `OpenCoder-LLM/opc-sft-stage2` dataset was used for fine-tuning. This dataset is designed for instruction-tuning language models for code generation.

## Fine-tuning Process

The fine-tuning process was performed using the following techniques:

- **QLoRA**: For memory-efficient fine-tuning.
- **Flash Attention**: The training was performed using flash attention for improved performance.
- **BitsAndBytes**: For 4-bit quantization.

The fine-tuned Qwen-0.6B model adapter is saved in the `./qwen_code_lora_adapter` directory.
