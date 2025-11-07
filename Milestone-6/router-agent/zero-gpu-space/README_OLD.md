---
title: ZeroGPU-LLM-Inference
emoji: 🧠
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Streaming LLM chat with web search and debug
---

This Gradio app provides **token-streaming, chat-style inference** on a wide variety of Transformer models—leveraging ZeroGPU for free GPU acceleration on HF Spaces.

Key features:
- **Real-time DuckDuckGo web search** (background thread, configurable timeout) with results injected into the system prompt.
- **Prompt preview panel** for debugging and prompt-engineering insights—see exactly what’s sent to the model.
- **Thought vs. Answer streaming**: any `<think>…</think>` blocks emitted by the model are shown as separate “💭 Thought.”
- **Cancel button** to immediately stop generation.
- **Dynamic system prompt**: automatically inserts today’s date when you toggle web search.
- **Extensive model selection**: over 30 LLMs (from Phi-4 mini to Qwen3-14B, SmolLM2, Taiwan-ELM, Mistral, Meta-Llama, MiMo, Gemma, DeepSeek-R1, etc.).
- **Memory-safe design**: loads one model at a time, clears cache after each generation.
- **Customizable generation parameters**: max tokens, temperature, top-k, top-p, repetition penalty.
- **Web-search settings**: max results, max chars per result, search timeout.
- **Requirements pinned** to ensure reproducible deployment.

## 🔄 Supported Models

Use the dropdown to select any of these:

| Name                                  | Repo ID                                            |
| ------------------------------------- | -------------------------------------------------- |
| Taiwan-ELM-1_1B-Instruct              | liswei/Taiwan-ELM-1_1B-Instruct                    |
| Taiwan-ELM-270M-Instruct              | liswei/Taiwan-ELM-270M-Instruct                    |
| Qwen3-0.6B                            | Qwen/Qwen3-0.6B                                    |
| Qwen3-1.7B                            | Qwen/Qwen3-1.7B                                    |
| Qwen3-4B                              | Qwen/Qwen3-4B                                      |
| Qwen3-8B                              | Qwen/Qwen3-8B                                      |
| Qwen3-14B                             | Qwen/Qwen3-14B                                     |
| Gemma-3-4B-IT                         | unsloth/gemma-3-4b-it                              |
| SmolLM2-135M-Instruct-TaiwanChat      | Luigi/SmolLM2-135M-Instruct-TaiwanChat             |
| SmolLM2-135M-Instruct                 | HuggingFaceTB/SmolLM2-135M-Instruct                |
| SmolLM2-360M-Instruct-TaiwanChat      | Luigi/SmolLM2-360M-Instruct-TaiwanChat             |
| Llama-3.2-Taiwan-3B-Instruct          | lianghsun/Llama-3.2-Taiwan-3B-Instruct             |
| MiniCPM3-4B                           | openbmb/MiniCPM3-4B                                |
| Qwen2.5-3B-Instruct                   | Qwen/Qwen2.5-3B-Instruct                           |
| Qwen2.5-7B-Instruct                   | Qwen/Qwen2.5-7B-Instruct                           |
| Phi-4-mini-Reasoning                  | microsoft/Phi-4-mini-reasoning                     |
| Phi-4-mini-Instruct                   | microsoft/Phi-4-mini-instruct                      |
| Meta-Llama-3.1-8B-Instruct            | MaziyarPanahi/Meta-Llama-3.1-8B-Instruct            |
| DeepSeek-R1-Distill-Llama-8B          | unsloth/DeepSeek-R1-Distill-Llama-8B               |
| Mistral-7B-Instruct-v0.3              | MaziyarPanahi/Mistral-7B-Instruct-v0.3              |
| Qwen2.5-Coder-7B-Instruct             | Qwen/Qwen2.5-Coder-7B-Instruct                     |
| Qwen2.5-Omni-3B                       | Qwen/Qwen2.5-Omni-3B                               |
| MiMo-7B-RL                            | XiaomiMiMo/MiMo-7B-RL                              |

*(…and more can easily be added in `MODELS` in `app.py`.)*

## ⚙️ Generation & Search Parameters

- **Max Tokens**: 64–16384  
- **Temperature**: 0.1–2.0  
- **Top-K**: 1–100  
- **Top-P**: 0.1–1.0  
- **Repetition Penalty**: 1.0–2.0  

- **Enable Web Search**: on/off  
- **Max Results**: integer  
- **Max Chars/Result**: integer  
- **Search Timeout (s)**: 0.0–30.0  

## 🚀 How It Works

1. **User message** enters chat history.  
2. If search is enabled, a background DuckDuckGo thread fetches snippets.  
3. After up to *Search Timeout* seconds, snippets merge into the system prompt.  
4. The selected model pipeline is loaded (bf16→f16→f32 fallback) on ZeroGPU.  
5. Prompt is formatted—any `<think>…</think>` blocks will be streamed as separate “💭 Thought.”  
6. Tokens stream to the Chatbot UI. Press **Cancel** to stop mid-generation.