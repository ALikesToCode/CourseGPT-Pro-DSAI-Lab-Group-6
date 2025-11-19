# Fine-Tuning Llama 3.1 for Code Generation

## Overview

This project fine-tunes the `meta-llama/Meta-Llama-3.1-8B-Instruct` model on a code instruction dataset. The goal is to create a model that can generate code snippets based on natural language instructions. The fine-tuning process uses Low-Rank Adaptation (LoRA) for efficient training.

## Dataset

The dataset used for fine-tuning is `OpenCoder-LLM/opc-sft-stage2` with the `educational_instruct` configuration. This dataset contains pairs of instructions and corresponding code snippets.

Each example in the dataset has the following structure:
- `instruction`: The natural language instruction.
- `output`: The corresponding code snippet.
- `code`: The code snippet.
- `entry_point`: The entry point of the code.
- `testcase`: A list of test cases for the code.

## Model

The base model for fine-tuning is `meta-llama/Meta-Llama-3.1-8B-Instruct`. This is a powerful large language model from Meta AI.

## Fine-tuning Process

The fine-tuning process consists of the following steps:

1.  **Load the dataset**: The `OpenCoder-LLM/opc-sft-stage2` dataset is loaded using the `datasets` library.

2.  **Format the data**: The dataset is formatted into a chat template suitable for fine-tuning Llama 3. The template is as follows:

    ```
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are an expert programming assistant.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>

    {instruction}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

    {full_response}<|eot_id|>
    ```

3.  **Save the formatted data**: The formatted dataset is saved as a JSONL file named `llama31_code_finetune_simple.jsonl`.

4.  **Load the pre-trained model**: The `meta-llama/Meta-Llama-3.1-8B-Instruct` model is loaded with 4-bit quantization to reduce memory usage.

5.  **Configure LoRA**: LoRA is configured to adapt the model for the code generation task. The LoRA configuration targets the `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` modules.

6.  **Train the model**: The model is fine-tuned using the `SFTTrainer` from the `trl` library.

7.  **Save the adapter**: The trained LoRA adapter is saved to the `./llama31_code_lora_adapter/final_adapter` directory.

## Usage

To run the fine-tuning process, you can execute the cells in the `finetunening.ipynb` notebook. Make sure you have the required libraries installed.
