# Code Agent — Evaluation with Ollama (Milestone 5)

This folder contains a Jupyter notebook (`eval_ollama.ipynb`) for evaluating a fine-tuned code generation model using Ollama.

The notebook performs the following steps:
1.  Loads a fine-tuned model from a local path using `peft` and `transformers`.
2.  Loads a dataset from Hugging Face (`nvidia/OpenCodeReasoning`).
3.  Generates code from the fine-tuned model based on the dataset.
4.  Uses an Ollama model to evaluate the generated code against a rubric.

## Prerequisites

- Python 3.8+ (Jupyter environment recommended)
- Ollama server running
- Install required packages:

```bash
pip install torch transformers peft bitsandbytes datasets ollama
```

## Usage

1.  **Configure the notebook:**
    - Open `eval_ollama.ipynb`.
    - Update the `lora_path` variable to point to the directory containing your fine-tuned model weights.
    - Specify the Ollama model you want to use for evaluation (e.g., `gpt-oss:20b`).

2.  **Run the cells:**
    - Execute the cells in the notebook sequentially to perform the evaluation.

The notebook will output the evaluation scores for the generated code.
