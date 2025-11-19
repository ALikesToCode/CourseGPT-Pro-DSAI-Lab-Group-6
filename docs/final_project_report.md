# CourseGPT-Pro: Final Project Report

**CourseGPT-Pro: A Multimodal, Tool-Augmented RAG System for Technical Homework Assistance**

**Team:** DSAI Lab Group 6
**Date:** November 19, 2025

---

## Abstract

Students working on technical assignments in engineering and computer science face a major challenge: course content is multimodal, featuring a complex combination of text, mathematical expressions (LaTeX), diagrams, tables, and code segments. Current AI helpers and naive Retrieval-Augmented Generation (RAG) models fall short, as they cannot interpret this structural complexity or perform the required logical and computational reasoning. This project proposes **CourseGPT-Pro**, an end-to-end system designed to overcome these limitations. By integrating a high-fidelity multimodal ingestion pipeline, a **hybrid retrieval system**, a **tool-augmented generative model orchestrated by a router and sub-agents**, and a final **verification layer**, CourseGPT-Pro aims to deliver accurate, reliable, and fully-cited answers to challenging technical problems.

---

## 1. Introduction

The main goal of CourseGPT-Pro is to build, train, and test a reliable homework bot that delivers verifiable, step-by-step assistance on technical university problems (e.g., IIT Madras curriculum). The system addresses the limitations of standard LLMs in handling domain-specific, multimodal technical content by employing a modular, agentic architecture.

### Project Objectives
1.  **High-Fidelity Multimodal Ingestion**: Parsing PDF documents into structured formats (text, LaTeX, tables, code).
2.  **Deterministic OCR for Grounding**: Establishing a canonical text layer for safety and verifiability.
3.  **Context-Aware Hybrid Retrieval**: Blending dense vector search with sparse lexical search and specialized math/code search.
4.  **Strong Tool-Augmented Reasoning**: Training models to reliably delegate tasks to symbolic solvers (SymPy) and code interpreters.
5.  **Verification & Trust**: Cross-checking answers against retrieved sources with exact citations.
6.  **Comprehensive Performance Assessment**: rigorous evaluation using automated metrics and judge models.

---

## 2. Literature Review (Milestone 1)

CourseGPT-Pro sits at the intersection of retrieval, multimodal perception, tool-augmented reasoning, and verifiability.

### 2.1 Retrieval-Augmented Generation & Hybrid Search
Modern RAG pipelines must optimize retriever quality and generator robustness. Research highlights the value of hybrid search—combining dense vector search for conceptual understanding with sparse lexical search (BM25) for precise keyword matching [1][7]. Our system adopts this hybrid approach, enhanced with reranking to guarantee faithful answers [5].

### 2.2 Multimodal Document Understanding
Document AI research distinguishes between OCR-dependent systems (LayoutLMv3) and OCR-free Transformers (Donut). We employ a dual strategy: deterministic OCR for symbol-level fidelity (critical for math/code) and multimodal encoders for layout reasoning and diagram grounding [12][15].

### 2.3 Tool-Augmented Language Models
Systems like Toolformer and ReAct demonstrate that LLMs can be taught to use external tools (calculators, APIs) to boost accuracy [24][28]. CourseGPT-Pro builds on this by using a **Router Agent** to dynamically dispatch tasks to specialized **Math** (SymPy-enabled) and **Code** (sandbox-enabled) agents.

### 2.4 Verifiability and Safety
Faithful tutoring requires page-level citations. Recent work on grounding (AGREE, G3) and secure execution environments informs our verification layer, ensuring every answer is cited and code is executed in isolated sandboxes [39][45].

---

## 3. Dataset and Methodology (Milestones 2–3)

Our methodology relies on specialized datasets and a modular agent architecture.

### 3.1 Dataset Preparation
We prepared three distinct datasets to train our specialized agents:

*   **Code Agent**: Sourced from `OpenCoder-LLM/opc-sft-stage2` (educational_instruct subset). Processed 118K instruction-code pairs, formatted with Llama 3.1 chat templates, and validated for SFT training.
*   **Math Agent**: Sourced from `XenArcAI/MathX-5M`. Processed ~4.32M problems (subset used for tuning), cleaning LaTeX formatting and structuring for chain-of-thought reasoning.
*   **Router Agent**: Generated synthetic training data using **Google Gemini 2.5 Pro**. Created 8,189 labeled routing traces containing user queries, route plans, rationales, and tool calls.

### 3.2 System Architecture
The system is orchestrated by a central **Router Agent** that delegates to sub-agents:

*   **Router Agent**: A lightweight orchestrator (fine-tuned Llama 3.1 8B / Gemma 3 27B / Qwen 3 32B) that analyzes queries and outputs a JSON execution plan.
*   **Math Agent**: A fine-tuned **Gemma 3 4B** model optimized for instruction adherence and step-by-step reasoning, capable of invoking SymPy.
*   **Code Agent**: A fine-tuned **Qwen 3 0.6B** model optimized for code generation, running in a secure sandbox.
*   **General Agent**: Handles retrieval from the knowledge base and general queries.

---

## 4. Model Development and Hyperparameter Tuning (Milestone 4)

We employed Parameter-Efficient Fine-Tuning (PEFT) techniques to adapt large language models to our specific domains.

### 4.1 Router Agent Tuning
*   **Base Models**: Llama 3.1 8B, Gemma 3 27B, Qwen 3 32B.
*   **Method**: Vertex AI PEFT/LoRA (Rank 16).
*   **Hyperparameters**: Learning rate multiplier 0.7, Epochs 3, Warm-up ratio 0.1.
*   **Outcome**: Gemma 3 27B adapter achieved the lowest validation loss (0.608) and best stability.

### 4.2 Math Agent Tuning
*   **Base Model**: Gemma 3 4B IT.
*   **Method**: LoRA (Rank 16, Alpha 32) on attention and MLP projections.
*   **Training**: Mixed BF16/FP16 precision, sequence length 2048.
*   **Outcome**: Training loss reduced from 2.1 to 0.37. Achieved 90.4% accuracy on curated evaluation set.

### 4.3 Code Agent Tuning
*   **Base Model**: Qwen 3 0.6B.
*   **Method**: QLoRA (Rank 16, Alpha 32) with 4-bit quantization.
*   **Training**: SFTTrainer with Flash Attention.
*   **Outcome**: Training loss reduced from 2.70 to ~0.40. The 0.6B model proved highly efficient for latency-sensitive code generation.

---

## 5. Evaluation & Analysis (Milestone 5)

We implemented a comprehensive evaluation framework covering all components.

### 5.1 Router Evaluation
*   **Metrics**: Schema validity (JSON structure), Tool selection precision/recall, Route rationale quality.
*   **Benchmarks**: A "Hard Benchmark" set of 322 adversarial samples.
*   **Results**: The Gemma 3 adapter demonstrated superior performance in adhering to the strict JSON schema and correctly routing complex, multi-step queries.

### 5.2 Math Agent Evaluation
*   **Benchmarks**: Evaluated against standard suites (GSM8K, MATH) and a custom subset of MathX.
*   **Method**: Vertex AI batch prediction with automated metric computation (exact match, numeric equality).
*   **Findings**: The specialized Gemma 3 4B model significantly outperformed baseline zero-shot performance on curriculum-aligned problems.

### 5.3 OCR & Ingestion
*   **Scaffolding**: Developed `ocr-module` for evaluating text extraction quality (CER/WER) and document type detection.

---

## 6. Deployment & Documentation (Milestone 6)

The final system is deployed as a modular, interactive application.

### 6.1 Deployment Architecture
*   **Frontend**: A **Streamlit UI** (`app/`) provides a chat interface, document management, and settings. It communicates with the backend services.
*   **Orchestration**: A **FastAPI** service (`api/`) hosts the LangGraph-based orchestration logic, managing the flow between Router, Math, and Code agents.
*   **Router Service**: The Router Agent is hosted on **Hugging Face Spaces** (or via ZeroGPU API) to provide low-latency routing decisions.
*   **Storage**: Cloudflare R2 is used for document storage, and Cloudflare AI Search (AutoRAG) powers the retrieval pipeline.

### 6.2 Documentation
Comprehensive documentation has been created under `docs/`, including:
*   **Technical Documentation**: Detailed setup, training, and deployment guides.
*   **User Guide**: Instructions for end-users on interacting with the bot.
*   **API Reference**: Specifications for the backend REST endpoints.

---

## 7. Conclusion and Future Work

CourseGPT-Pro successfully demonstrates a specialized, agentic approach to technical homework assistance. By decomposing problems into routing, reasoning, and coding tasks, we achieve higher reliability than monolithic models.

### Future Work
*   **Multimodal Expansion**: Deeper integration of vision models for interpreting diagrams directly within the reasoning loop.
*   **Feedback Loops**: Implementing reinforcement learning from human feedback (RLHF) based on user corrections.
*   **Scale**: Training larger specialist models (e.g., 70B+ parameter math agents) as compute resources allow.
*   **Live Retrieval**: Connecting the General Agent to live web search for real-time information.

---

## 8. References

1.  Zhao et al. "Retrieval-Augmented Generation for AI-Generated Content: A Survey." [arXiv:2402.19473]
2.  Gao et al. "Retrieval-Augmented Generation for Large Language Models: A Survey." [arXiv:2312.10997]
5.  NVIDIA. "Build an Enterprise-Scale Multimodal Document Retrieval Pipeline."
12. Huang et al. "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking." [arXiv:2204.08387]
24. Schick et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." [arXiv:2302.04761]
45. Google Research. "Effective Large Language Model Adaptation for Improved Grounding."
64. "MathScale: Scaling Instruction Tuning for Mathematical Reasoning." [arXiv:2403.02884]
141. "MathVista: Evaluating Mathematical Reasoning in Visual Contexts." [arXiv:2310.02255]

*(Full reference list available in `docs/technical_doc.md`)*
