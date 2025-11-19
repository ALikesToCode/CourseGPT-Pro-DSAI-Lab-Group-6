# Preprocessing report for math agent

This readme contains only the informations about the files in this directory and how they work. For the full report please check the root milestone directory.


## Models 

Notebooks for these models exist individually, more than one models were tried because of hardware and performance limitations, the finals solution will only use one model for the math agent

- meta-llama/Llama-3.1-8B-Instruct
- meta-llama/Llama-3.2-3B-Instruct

## File information


- .gitignore - ignore any build files and models downloaded
- use_trained_model.sh - script used to run the fine tuned model
- Modelfile - config file for trained model
- requirements.txt - python libraries used to run the notebooks
- math_agent_preprocessing** - notebooks containing the preprocessing code

## Dataset used

Link - https://huggingface.co/datasets/XenArcAI/MathX-5M

### 1. Dataset Overview

* **Dataset Name:** `XenArcAI/MathX-5M`
* **Total Size:** ~2.43 GB
* **Number of Rows:** ~4.32 Million
* **Format:** Parquet
* **Task:** Text Generation, Question Answering
* **Description:** `MathX-5M` is a large-scale, high-quality corpus of mathematical reasoning problems. It contains 5 million examples of problems with detailed, step-by-step solutions, making it ideal for instruction-based fine-tuning. The dataset covers a wide range of mathematical domains, from basic arithmetic to advanced calculus.

### 2. Data Features

The dataset consists of three primary columns (features):

1.  **`problem`** (string):
    * **Content:** This column contains the mathematical problem statement. The problems are expressed in natural language and often include LaTeX formatting for mathematical notation.
    * **Example:** "Determine how many 1000 digit numbers \\( a \\) have the property that when any digit..."
    * **Observations:** The problems vary significantly in length and complexity, ranging from simple calculations to complex, multi-step proofs.

2.  **`expected_answer`** (string):
    * **Content:** This column holds the final, correct answer to the problem. The answers are typically concise and may also use LaTeX.
    * **Example:** `\[[32]]`
    * **Observations:** This feature provides the ground truth for evaluating the model's final output. The format is generally clean and direct.

3.  **`generated_solution`** (string):
    * **Content:** This is arguably the most valuable feature for fine-tuning. It contains a detailed, step-by-step thought process and derivation of the solution. It often starts with a `<think>` tag, outlining the reasoning path.
    * **Example:** `<think> Okay, so I need to figure out how many 100-digit numbers ... The problem is to compute the ... </think> To find the probability...`
    * **Observations:** This "chain-of-thought" or reasoning path is crucial for teaching the model *how* to solve problems, not just what the final answer is. The quality and detail of these solutions are key to the dataset's effectiveness.

### 3. Initial Findings & Implications for Fine-Tuning

* **Instructional Format:** The dataset's structure is perfectly suited for instruction fine-tuning. The combination of a problem, a reasoning process, and a final answer provides a clear and rich learning signal for the model.
* **LaTeX Formatting:** The prevalence of LaTeX means the tokenizer and model must be proficient at handling mathematical notation.
* **Chain-of-Thought:** The `generated_solution` column enables the model to learn complex reasoning. During fine-tuning, the prompt should be structured to encourage the model to generate a similar step-by-step thought process before arriving at the final answer.
* **Data Scale:** With over 4 million rows, the dataset is substantial. Even a small fraction of this data is sufficient for effective fine-tuning, especially when using techniques like LoRA.
* **Complexity Distribution:** The dataset claims a distribution of basic (30%), intermediate (30%), and advanced (40%) problems. This diversity is excellent for training a well-rounded model that can handle a variety of mathematical challenges.
