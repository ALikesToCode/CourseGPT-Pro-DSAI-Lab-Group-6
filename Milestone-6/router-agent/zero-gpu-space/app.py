import os
import time
import gc
import sys
import threading
from itertools import islice
from datetime import datetime
import re  # for parsing <think> blocks
import gradio as gr
import torch
from transformers import pipeline, TextIteratorStreamer, StoppingCriteria
from transformers import AutoTokenizer
from ddgs import DDGS
import spaces  # Import spaces early to enable ZeroGPU support
from torch.utils._pytree import tree_map

# Global event to signal cancellation from the UI thread to the generation thread
cancel_event = threading.Event()

access_token=os.environ['HF_TOKEN']

# Optional: Disable GPU visibility if you wish to force CPU usage
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ------------------------------
# Torch-Compatible Model Definitions with Adjusted Descriptions
# ------------------------------
MODELS = {
    # "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8": {
    #     "repo_id": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
    #     "description": "Sparse Mixture-of-Experts (MoE) causal language model with 80B total parameters and approximately 3B activated per inference step. Features include native 32,768-token context (extendable to 131,072 via YaRN), 16 query heads and 2 KV heads, head dimension of 256, and FP8 quantization for efficiency. Optimized for fast, stable instruction-following dialogue without 'thinking' traces, making it ideal for general chat and low-latency applications [[2]][[3]][[5]][[8]].",
    #     "params_b": 80.0
    # },
    # "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8": {
    #     "repo_id": "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    #     "description": "Sparse Mixture-of-Experts (MoE) causal language model with 80B total parameters and approximately 3B activated per inference step. Features include native 32,768-token context (extendable to 131,072 via YaRN), 16 query heads and 2 KV heads, head dimension of 256, and FP8 quantization. Specialized for complex reasoning, math, and coding tasks, this model outputs structured 'thinking' traces by default and is designed to be used with a reasoning parser [[10]][[11]][[14]][[18]].",
    #     "params_b": 80.0
    # },
    "Qwen3-32B-FP8": {
        "repo_id": "Qwen/Qwen3-32B-FP8",
        "description": "Dense causal language model with 32.8B total parameters (31.2B non-embedding), 64 layers, 64 query heads & 8 KV heads, native 32,768-token context (extendable to 131,072 via YaRN). Features seamless switching between thinking mode (for complex reasoning, math, coding) and non-thinking mode (for efficient dialogue), strong multilingual support (100+ languages), and leading open-source agent capabilities.",
        "params_b": 32.8
    },
    # ~30.5B total parameters (MoE: 3.3B activated)
    # "Qwen3-30B-A3B-Instruct-2507": {
    #     "repo_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    #     "description": "non-thinking-mode MoE model based on Qwen3-30B-A3B-Instruct-2507. Features 30.5B total parameters (3.3B activated), 128 experts (8 activated), 48 layers, and native 262,144-token context. Excels in instruction following, logical reasoning, multilingualism, coding, and long-context understanding. Supports only non-thinking mode (no <think> blocks). Quantized using AWQ (W4A16) with lm_head and gating layers preserved in higher precision.",
    #     "params_b": 30.5
    # },
    # "Qwen3-30B-A3B-Thinking-2507": {
    #     "repo_id": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    #     "description": "thinking-mode MoE model based on Qwen3-30B-A3B-Thinking-2507. Contains 30.5B total parameters (3.3B activated), 128 experts (8 activated), 48 layers, and 262,144-token native context. Optimized for deep reasoning in mathematics, science, coding, and agent tasks. Outputs include automatic reasoning delimiters (<think>...</think>). Quantized with AWQ (W4A16), preserving lm_head and expert gating layers.",
    #     "params_b": 30.5
    # },
    "gpt-oss-20b-BF16": {
        "repo_id": "unsloth/gpt-oss-20b-BF16",
        "description": "A 20B-parameter open-source GPT-style language model quantized to INT4 using AutoRound, with FP8 key-value cache for efficient inference. Optimized for performance and memory efficiency on Intel hardware while maintaining strong language generation capabilities.",
        "params_b": 20.0
    },
    "Qwen3-4B-Instruct-2507": {
        "repo_id": "Qwen/Qwen3-4B-Instruct-2507",
        "description": "Updated non-thinking instruct variant of Qwen3-4B with 4.0B parameters, featuring significant improvements in instruction following, logical reasoning, multilingualism, and 256K long-context understanding. Strong performance across knowledge, coding, alignment, and agent benchmarks.",
        "params_b": 4.0
    },    
    "Apriel-1.5-15b-Thinker": {
        "repo_id": "ServiceNow-AI/Apriel-1.5-15b-Thinker",
        "description": "Multimodal reasoning model with 15B parameters, trained via extensive mid-training on text and image data, and fine-tuned only on text (no image SFT). Achieves competitive performance on reasoning benchmarks like Artificial Analysis (score: 52), Tau2 Bench Telecom (68), and IFBench (62). Supports both text and image understanding, fits on a single GPU, and includes structured reasoning output with tool and function calling capabilities.",
        "params_b": 15.0
    },
    
    # 14.8B total parameters
    "Qwen3-14B": {
        "repo_id": "Qwen/Qwen3-14B",
        "description": "Dense causal language model with 14.8 B total parameters (13.2 B non-embedding), 40 layers, 40 query heads & 8 KV heads, 32 768-token context (131 072 via YaRN), enhanced human preference alignment & advanced agent integration.",
        "params_b": 14.8
    },
    "Qwen/Qwen3-14B-FP8": {
        "repo_id": "Qwen/Qwen3-14B-FP8",
        "description": "FP8-quantized version of Qwen3-14B for efficient inference.",
        "params_b": 14.8
    },

    # ~15B (commented out in original, but larger than 14B)
    # "Apriel-1.5-15b-Thinker": { ... },

    # 5B
    # "Apriel-5B-Instruct": {
    #     "repo_id": "ServiceNow-AI/Apriel-5B-Instruct",
    #     "description": "A 5B-parameter instruction-tuned model from ServiceNow’s Apriel series, optimized for enterprise tasks and general-purpose instruction following."
    # },

    # 4.3B
    "Phi-4-mini-Reasoning": {
        "repo_id": "microsoft/Phi-4-mini-reasoning",
        "description": "Phi-4-mini-Reasoning (4.3B parameters)",
        "params_b": 4.3
    },
    "Phi-4-mini-Instruct": {
        "repo_id": "microsoft/Phi-4-mini-instruct",
        "description": "Phi-4-mini-Instruct (4.3B parameters)",
        "params_b": 4.3
    },

    # 4.0B
    "Qwen3-4B": {
        "repo_id": "Qwen/Qwen3-4B",
        "description": "Dense causal language model with 4.0 B total parameters (3.6 B non-embedding), 36 layers, 32 query heads & 8 KV heads, native 32 768-token context (extendable to 131 072 via YaRN), balanced mid-range capacity & long-context reasoning.",
        "params_b": 4.0
    },

    "Gemma-3-4B-IT": {
        "repo_id": "unsloth/gemma-3-4b-it",
        "description": "Gemma-3-4B-IT",
        "params_b": 4.0
    },
    "MiniCPM3-4B": {
        "repo_id": "openbmb/MiniCPM3-4B",
        "description": "MiniCPM3-4B",
        "params_b": 4.0
    },
    "Gemma-3n-E4B": {
        "repo_id": "google/gemma-3n-E4B",
        "description": "Gemma 3n base model with effective 4 B parameters (≈3 GB VRAM)",
        "params_b": 4.0
    },
    "SmallThinker-4BA0.6B-Instruct": {
        "repo_id": "PowerInfer/SmallThinker-4BA0.6B-Instruct",
        "description": "SmallThinker 4 B backbone with 0.6 B activated parameters, instruction‑tuned",
        "params_b": 4.0
    },

    # ~3B
    # "AI21-Jamba-Reasoning-3B": {
    #     "repo_id": "ai21labs/AI21-Jamba-Reasoning-3B",
    #     "description": "A compact 3B hybrid Transformer–Mamba reasoning model with 256K context length, strong intelligence benchmark scores (61% MMLU-Pro, 52% IFBench), and efficient inference suitable for edge and datacenter use. Outperforms Gemma-3 4B and Llama-3.2 3B despite smaller size."
    # },
    "Qwen2.5-Taiwan-3B-Reason-GRPO": {
        "repo_id": "benchang1110/Qwen2.5-Taiwan-3B-Reason-GRPO",
        "description": "Qwen2.5-Taiwan model with 3 B parameters, Reason-GRPO fine-tuned",
        "params_b": 3.0
    },
    "Llama-3.2-Taiwan-3B-Instruct": {
        "repo_id": "lianghsun/Llama-3.2-Taiwan-3B-Instruct",
        "description": "Llama-3.2-Taiwan-3B-Instruct",
        "params_b": 3.0
    },
    "Qwen2.5-3B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Qwen2.5-3B-Instruct",
        "params_b": 3.0
    },
    "Qwen2.5-Omni-3B": {
        "repo_id": "Qwen/Qwen2.5-Omni-3B",
        "description": "Qwen2.5-Omni-3B",
        "params_b": 3.0
    },
    "Granite-4.0-Micro": {
        "repo_id": "ibm-granite/granite-4.0-micro",
        "description": "A 3B-parameter long-context instruct model from IBM, finetuned for enhanced instruction following and tool-calling. Supports 12 languages including English, Chinese, Arabic, and Japanese. Built on a dense Transformer with GQA, RoPE, SwiGLU, and 128K context length. Trained using SFT, RL alignment, and model merging techniques for enterprise applications.",
        "params_b": 3.0
    },

    # 2.6B
    "LFM2-2.6B": {
        "repo_id": "LiquidAI/LFM2-2.6B",
        "description": "The 2.6B parameter model in the LFM2 series, it outperforms models in the 3B+ class and features a hybrid architecture for faster inference.",
        "params_b": 2.6
    },

    # 1.7B
    "Qwen3-1.7B": {
        "repo_id": "Qwen/Qwen3-1.7B",
        "description": "Dense causal language model with 1.7 B total parameters (1.4 B non-embedding), 28 layers, 16 query heads & 8 KV heads, 32 768-token context, stronger reasoning vs. 0.6 B variant, dual-mode inference, instruction following across 100+ languages.",
        "params_b": 1.7
    },

    # ~2B (effective)
    "Gemma-3n-E2B": {
        "repo_id": "google/gemma-3n-E2B",
        "description": "Gemma 3n base model with effective 2 B parameters (≈2 GB VRAM)",
        "params_b": 2.0
    },

    # 1.5B
    "Nemotron-Research-Reasoning-Qwen-1.5B": {
        "repo_id": "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
        "description": "Nemotron-Research-Reasoning-Qwen-1.5B",
        "params_b": 1.5
    },
    "Falcon-H1-1.5B-Instruct": {
        "repo_id": "tiiuae/Falcon-H1-1.5B-Instruct",
        "description": "Falcon‑H1 model with 1.5 B parameters, instruction‑tuned",
        "params_b": 1.5
    },
    "Qwen2.5-Taiwan-1.5B-Instruct": {
        "repo_id": "benchang1110/Qwen2.5-Taiwan-1.5B-Instruct",
        "description": "Qwen2.5-Taiwan-1.5B-Instruct",
        "params_b": 1.5
    },

    # 1.2B
    "LFM2-1.2B": {
        "repo_id": "LiquidAI/LFM2-1.2B",
        "description": "A 1.2B parameter hybrid language model from Liquid AI, designed for efficient on-device and edge AI deployment, outperforming larger models like Llama-2-7b-hf in specific tasks.",
        "params_b": 1.2
    },

    # 1.1B
    "Taiwan-ELM-1_1B-Instruct": {
        "repo_id": "liswei/Taiwan-ELM-1_1B-Instruct",
        "description": "Taiwan-ELM-1_1B-Instruct",
        "params_b": 1.1
    },

    # 1B
    "Llama-3.2-Taiwan-1B": {
        "repo_id": "lianghsun/Llama-3.2-Taiwan-1B",
        "description": "Llama-3.2-Taiwan base model with 1 B parameters",
        "params_b": 1.0
    },

    # 700M
    "LFM2-700M": {
        "repo_id": "LiquidAI/LFM2-700M",
        "description": "A 700M parameter model from the LFM2 family, designed for high efficiency on edge devices with a hybrid architecture of multiplicative gates and short convolutions.",
        "params_b": 0.7
    },

    # 600M
    "Qwen3-0.6B": {
        "repo_id": "Qwen/Qwen3-0.6B",
        "description": "Dense causal language model with 0.6 B total parameters (0.44 B non-embedding), 28 transformer layers, 16 query heads & 8 KV heads, native 32 768-token context window, dual-mode generation, full multilingual & agentic capabilities.",
        "params_b": 0.6
    },
    "Qwen3-0.6B-Taiwan": {
        "repo_id": "ShengweiPeng/Qwen3-0.6B-Taiwan",
        "description": "Qwen3-Taiwan model with 0.6 B parameters",
        "params_b": 0.6
    },

    # 500M
    "Qwen2.5-0.5B-Taiwan-Instruct": {
        "repo_id": "ShengweiPeng/Qwen2.5-0.5B-Taiwan-Instruct",
        "description": "Qwen2.5-Taiwan model with 0.5 B parameters, instruction-tuned",
        "params_b": 0.5
    },

    # 360M
    "SmolLM2-360M-Instruct": {
        "repo_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "description": "Original SmolLM2‑360M Instruct",
        "params_b": 0.36
    },
    "SmolLM2-360M-Instruct-TaiwanChat": {
        "repo_id": "Luigi/SmolLM2-360M-Instruct-TaiwanChat",
        "description": "SmolLM2‑360M Instruct fine-tuned on TaiwanChat",
        "params_b": 0.36
    },

    # 350M
    "LFM2-350M": {
        "repo_id": "LiquidAI/LFM2-350M",
        "description": "A compact 350M parameter hybrid model optimized for edge and on-device applications, offering significantly faster training and inference speeds compared to models like Qwen3.",
        "params_b": 0.35
    },

    # 270M
    "parser_model_ner_gemma_v0.1": {
        "repo_id": "myfi/parser_model_ner_gemma_v0.1",
        "description": "A lightweight named‑entity‑like (NER) parser fine‑tuned from Google’s **Gemma‑3‑270M** model. The base Gemma‑3‑270M is a 270 M‑parameter, hyper‑efficient LLM designed for on‑device inference, supporting >140 languages, a 128 k‑token context window, and instruction‑following capabilities [2][7]. This variant is further trained on standard NER corpora (e.g., CoNLL‑2003, OntoNotes) to extract PERSON, ORG, LOC, and MISC entities with high precision while keeping the memory footprint low (≈240 MB VRAM in BF16 quantized form) [1]. It is released under the Apache‑2.0 license and can be used for fast, cost‑effective entity extraction in low‑resource environments.",
        "params_b": 0.27
    },
    "Gemma-3-Taiwan-270M-it": {
        "repo_id": "lianghsun/Gemma-3-Taiwan-270M-it",
        "description": "google/gemma-3-270m-it fintuned on Taiwan Chinese dataset",
        "params_b": 0.27
    },
    "gemma-3-270m-it": {
        "repo_id": "google/gemma-3-270m-it",
        "description": "Gemma‑3‑270M‑IT is a compact, 270‑million‑parameter language model fine‑tuned for Italian, offering fast and efficient on‑device text generation and comprehension in the Italian language.",
        "params_b": 0.27
    },
    "Taiwan-ELM-270M-Instruct": {
        "repo_id": "liswei/Taiwan-ELM-270M-Instruct",
        "description": "Taiwan-ELM-270M-Instruct",
        "params_b": 0.27
    },

    # 135M
    "SmolLM2-135M-multilingual-base": {
        "repo_id": "agentlans/SmolLM2-135M-multilingual-base",
        "description": "SmolLM2-135M-multilingual-base",
        "params_b": 0.135
    },
    "SmolLM-135M-Taiwan-Instruct-v1.0": {
        "repo_id": "benchang1110/SmolLM-135M-Taiwan-Instruct-v1.0",
        "description": "135-million-parameter F32 safetensors instruction-finetuned variant of SmolLM-135M-Taiwan, trained on the 416 k-example ChatTaiwan dataset for Traditional Chinese conversational and instruction-following tasks",
        "params_b": 0.135
    },
    "SmolLM2_135M_Grpo_Gsm8k": {
        "repo_id": "prithivMLmods/SmolLM2_135M_Grpo_Gsm8k",
        "description": "SmolLM2_135M_Grpo_Gsm8k",
        "params_b": 0.135
    },
    "SmolLM2-135M-Instruct": {
        "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "description": "Original SmolLM2‑135M Instruct",
        "params_b": 0.135
    },
    "SmolLM2-135M-Instruct-TaiwanChat": {
        "repo_id": "Luigi/SmolLM2-135M-Instruct-TaiwanChat",
        "description": "SmolLM2‑135M Instruct fine-tuned on TaiwanChat",
        "params_b": 0.135
    },
}

# Global cache for pipelines to avoid re-loading.
PIPELINES = {}

def load_pipeline(model_name):
    """
    Load and cache a transformers pipeline for text generation.
    Tries bfloat16, falls back to float16 or float32 if unsupported.
    """
    global PIPELINES
    if model_name in PIPELINES:
        return PIPELINES[model_name]
    repo = MODELS[model_name]["repo_id"]
    tokenizer = AutoTokenizer.from_pretrained(repo,
                token=access_token)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        try:
            pipe = pipeline(
                task="text-generation",
                model=repo,
                tokenizer=tokenizer,
                trust_remote_code=True,
                dtype=dtype, # Use `dtype` instead of deprecated `torch_dtype`
                device_map="auto",
                use_cache=True,      # Enable past-key-value caching
                token=access_token)
            PIPELINES[model_name] = pipe
            return pipe
        except Exception:
            continue
    # Final fallback
    pipe = pipeline(
        task="text-generation",
        model=repo,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
        use_cache=True
    )
    PIPELINES[model_name] = pipe
    return pipe


def retrieve_context(query, max_results=6, max_chars=50):
    """
    Retrieve search snippets from DuckDuckGo (runs in background).
    Returns a list of result strings.
    """
    try:
        with DDGS() as ddgs:
            return [f"{i+1}. {r.get('title','No Title')} - {r.get('body','')[:max_chars]}"
                    for i, r in enumerate(islice(ddgs.text(query, region="wt-wt", safesearch="off", timelimit="y"), max_results))]
    except Exception:
        return []

def format_conversation(history, system_prompt, tokenizer):
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "system", "content": system_prompt.strip()}] + history
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    else:
        # Fallback for base LMs without chat template
        prompt = system_prompt.strip() + "\n"
        for msg in history:
            if msg['role'] == 'user':
                prompt += "User: " + msg['content'].strip() + "\n"
            elif msg['role'] == 'assistant':
                prompt += "Assistant: " + msg['content'].strip() + "\n"
        if not prompt.strip().endswith("Assistant:"):
            prompt += "Assistant: "
        return prompt

def get_duration(user_msg, chat_history, system_prompt, enable_search, max_results, max_chars, model_name, max_tokens, temperature, top_k, top_p, repeat_penalty, search_timeout):
    # Get model size from the MODELS dict (more reliable than string parsing)
    model_size = MODELS[model_name].get("params_b", 4.0)  # Default to 4B if not found
    
    # Only use AOT for models >= 2B parameters
    use_aot = model_size >= 2
    
    # Adjusted for H200 performance: faster inference, quicker compilation
    base_duration = 20 if not use_aot else 40  # Reduced base times
    token_duration = max_tokens * 0.005  # ~200 tokens/second average on H200
    search_duration = 10 if enable_search else 0  # Reduced search time
    aot_compilation_buffer = 20 if use_aot else 0  # Faster compilation on H200
    
    return base_duration + token_duration + search_duration + aot_compilation_buffer

@spaces.GPU(duration=get_duration)
def chat_response(user_msg, chat_history, system_prompt,
                  enable_search, max_results, max_chars,
                  model_name, max_tokens, temperature,
                  top_k, top_p, repeat_penalty, search_timeout):
    """
    Generates streaming chat responses, optionally with background web search.
    This version includes cancellation support.
    """
    # Clear the cancellation event at the start of a new generation
    cancel_event.clear()
    
    history = list(chat_history or [])
    history.append({'role': 'user', 'content': user_msg})

    # Launch web search if enabled
    debug = ''
    search_results = []
    if enable_search:
        debug = 'Search task started.'
        thread_search = threading.Thread(
            target=lambda: search_results.extend(
                retrieve_context(user_msg, int(max_results), int(max_chars))
            )
        )
        thread_search.daemon = True
        thread_search.start()
    else:
        debug = 'Web search disabled.'

    try:
        cur_date = datetime.now().strftime('%Y-%m-%d')
        # merge any fetched search results into the system prompt
        if search_results:
            
            enriched = system_prompt.strip() + \
            f'''\n# The following contents are the search results related to the user's message:
            {search_results}
            In the search results I provide to you, each result is formatted as [webpage X begin]...[webpage X end], where X represents the numerical index of each article. Please cite the context at the end of the relevant sentence when appropriate. Use the citation format [citation:X] in the corresponding part of your answer. If a sentence is derived from multiple contexts, list all relevant citation numbers, such as [citation:3][citation:5]. Be sure not to cluster all citations at the end; instead, include them in the corresponding parts of the answer.
            When responding, please keep the following points in mind:
            - Today is {cur_date}.
            - Not all content in the search results is closely related to the user's question. You need to evaluate and filter the search results based on the question.
            - For listing-type questions (e.g., listing all flight information), try to limit the answer to 10 key points and inform the user that they can refer to the search sources for complete information. Prioritize providing the most complete and relevant items in the list. Avoid mentioning content not provided in the search results unless necessary.
            - For creative tasks (e.g., writing an essay), ensure that references are cited within the body of the text, such as [citation:3][citation:5], rather than only at the end of the text. You need to interpret and summarize the user's requirements, choose an appropriate format, fully utilize the search results, extract key information, and generate an answer that is insightful, creative, and professional. Extend the length of your response as much as possible, addressing each point in detail and from multiple perspectives, ensuring the content is rich and thorough.
            - If the response is lengthy, structure it well and summarize it in paragraphs. If a point-by-point format is needed, try to limit it to 5 points and merge related content.
            - For objective Q&A, if the answer is very brief, you may add one or two related sentences to enrich the content.
            - Choose an appropriate and visually appealing format for your response based on the user's requirements and the content of the answer, ensuring strong readability.
            - Your answer should synthesize information from multiple relevant webpages and avoid repeatedly citing the same webpage.
            - Unless the user requests otherwise, your response should be in the same language as the user's question.
            # The user's message is:
            '''
        else:
            enriched = system_prompt

        # wait up to 1s for snippets, then replace debug with them
        if enable_search:
            thread_search.join(timeout=float(search_timeout))
            if search_results:
                debug = "### Search results merged into prompt\n\n" + "\n".join(
                    f"- {r}" for r in search_results
                )
            else:
                debug = "*No web search results found.*"

        # merge fetched snippets into the system prompt
        if search_results:
            enriched = system_prompt.strip() + \
            f'''\n# The following contents are the search results related to the user's message:
            {search_results}
            In the search results I provide to you, each result is formatted as [webpage X begin]...[webpage X end], where X represents the numerical index of each article. Please cite the context at the end of the relevant sentence when appropriate. Use the citation format [citation:X] in the corresponding part of your answer. If a sentence is derived from multiple contexts, list all relevant citation numbers, such as [citation:3][citation:5]. Be sure not to cluster all citations at the end; instead, include them in the corresponding parts of the answer.
            When responding, please keep the following points in mind:
            - Today is {cur_date}.
            - Not all content in the search results is closely related to the user's question. You need to evaluate and filter the search results based on the question.
            - For listing-type questions (e.g., listing all flight information), try to limit the answer to 10 key points and inform the user that they can refer to the search sources for complete information. Prioritize providing the most complete and relevant items in the list. Avoid mentioning content not provided in the search results unless necessary.
            - For creative tasks (e.g., writing an essay), ensure that references are cited within the body of the text, such as [citation:3][citation:5], rather than only at the end of the text. You need to interpret and summarize the user's requirements, choose an appropriate format, fully utilize the search results, extract key information, and generate an answer that is insightful, creative, and professional. Extend the length of your response as much as possible, addressing each point in detail and from multiple perspectives, ensuring the content is rich and thorough.
            - If the response is lengthy, structure it well and summarize it in paragraphs. If a point-by-point format is needed, try to limit it to 5 points and merge related content.
            - For objective Q&A, if the answer is very brief, you may add one or two related sentences to enrich the content.
            - Choose an appropriate and visually appealing format for your response based on the user's requirements and the content of the answer, ensuring strong readability.
            - Your answer should synthesize information from multiple relevant webpages and avoid repeatedly citing the same webpage.
            - Unless the user requests otherwise, your response should be in the same language as the user's question.
            # The user's message is:
            '''
        else:
            enriched = system_prompt

        pipe = load_pipeline(model_name)

        prompt = format_conversation(history, enriched, pipe.tokenizer)
        prompt_debug = f"\n\n--- Prompt Preview ---\n```\n{prompt}\n```"
        streamer = TextIteratorStreamer(pipe.tokenizer,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        gen_thread = threading.Thread(
            target=pipe,
            args=(prompt,),
            kwargs={
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'repetition_penalty': repeat_penalty,
                'streamer': streamer,
                'return_full_text': False,
            }
        )
        gen_thread.start()

        # Buffers for thought vs answer
        thought_buf = ''
        answer_buf = ''
        in_thought = False
        assistant_message_started = False

        # First yield contains the user message
        yield history, debug

        # Stream tokens
        for chunk in streamer:
            # Check for cancellation signal
            if cancel_event.is_set():
                if assistant_message_started and history and history[-1]['role'] == 'assistant':
                    history[-1]['content'] += " [Generation Canceled]"
                yield history, debug
                break
            
            text = chunk

            # Detect start of thinking
            if not in_thought and '<think>' in text:
                in_thought = True
                history.append({'role': 'assistant', 'content': '', 'metadata': {'title': '💭 Thought'}})
                assistant_message_started = True
                after = text.split('<think>', 1)[1]
                thought_buf += after
                if '</think>' in thought_buf:
                    before, after2 = thought_buf.split('</think>', 1)
                    history[-1]['content'] = before.strip()
                    in_thought = False
                    answer_buf = after2
                    history.append({'role': 'assistant', 'content': answer_buf})
                else:
                    history[-1]['content'] = thought_buf
                yield history, debug
                continue

            if in_thought:
                thought_buf += text
                if '</think>' in thought_buf:
                    before, after2 = thought_buf.split('</think>', 1)
                    history[-1]['content'] = before.strip()
                    in_thought = False
                    answer_buf = after2
                    history.append({'role': 'assistant', 'content': answer_buf})
                else:
                    history[-1]['content'] = thought_buf
                yield history, debug
                continue

            # Stream answer
            if not assistant_message_started:
                history.append({'role': 'assistant', 'content': ''})
                assistant_message_started = True

            answer_buf += text
            history[-1]['content'] = answer_buf.strip()
            yield history, debug

        gen_thread.join()
        yield history, debug + prompt_debug
    except GeneratorExit:
        # Handle cancellation gracefully
        print("Chat response cancelled.")
        # Don't yield anything - let the cancellation propagate
        return
    except Exception as e:
        history.append({'role': 'assistant', 'content': f"Error: {e}"})
        yield history, debug
    finally:
        gc.collect()


def update_default_prompt(enable_search):
    return f"You are a helpful assistant."

def update_duration_estimate(model_name, enable_search, max_results, max_chars, max_tokens, search_timeout):
    """Calculate and format the estimated GPU duration for current settings."""
    try:
        dummy_msg, dummy_history, dummy_system_prompt = "", [], ""
        duration = get_duration(dummy_msg, dummy_history, dummy_system_prompt, 
                              enable_search, max_results, max_chars, model_name, 
                              max_tokens, 0.7, 40, 0.9, 1.2, search_timeout)
        model_size = MODELS[model_name].get("params_b", 4.0)
        return (f"⏱️ **Estimated GPU Time: {duration:.1f} seconds**\n\n"
                f"📊 **Model Size:** {model_size:.1f}B parameters\n"
                f"🔍 **Web Search:** {'Enabled' if enable_search else 'Disabled'}")
    except Exception as e:
        return f"⚠️ Error calculating estimate: {e}"

# ------------------------------
# Gradio UI
# ------------------------------
with gr.Blocks(
    title="LLM Inference with ZeroGPU",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        radius_size="lg",
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
    ),
    css="""
        .duration-estimate { background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-left: 4px solid #667eea; padding: 12px; border-radius: 8px; margin: 16px 0; }
        .chatbot { border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        button.primary { font-weight: 600; }
        .gradio-accordion { margin-bottom: 12px; }
    """
) as demo:
    # Header
    gr.Markdown("""
    # 🧠 ZeroGPU LLM Inference
    ### Powered by Hugging Face ZeroGPU with Web Search Integration
    """)
    
    with gr.Row():
        # Left Panel - Configuration
        with gr.Column(scale=3):
            # Core Settings (Always Visible)
            with gr.Group():
                gr.Markdown("### ⚙️ Core Settings")
                model_dd = gr.Dropdown(
                    label="🤖 Model",
                    choices=list(MODELS.keys()),
                    value="Qwen3-1.7B",
                    info="Select the language model to use"
                )
                search_chk = gr.Checkbox(
                    label="🔍 Enable Web Search",
                    value=False,
                    info="Augment responses with real-time web data"
                )
                sys_prompt = gr.Textbox(
                    label="📝 System Prompt",
                    lines=3,
                    value=update_default_prompt(search_chk.value),
                    placeholder="Define the assistant's behavior and personality..."
                )
            
            # Duration Estimate
            duration_display = gr.Markdown(
                value=update_duration_estimate("Qwen3-1.7B", False, 4, 50, 1024, 5.0),
                elem_classes="duration-estimate"
            )
            
            # Advanced Settings (Collapsible)
            with gr.Accordion("🎛️ Advanced Generation Parameters", open=False):
                max_tok = gr.Slider(
                    64, 16384, value=1024, step=32,
                    label="Max Tokens",
                    info="Maximum length of generated response"
                )
                temp = gr.Slider(
                    0.1, 2.0, value=0.7, step=0.1,
                    label="Temperature",
                    info="Higher = more creative, Lower = more focused"
                )
                with gr.Row():
                    k = gr.Slider(
                        1, 100, value=40, step=1,
                        label="Top-K",
                        info="Number of top tokens to consider"
                    )
                    p = gr.Slider(
                        0.1, 1.0, value=0.9, step=0.05,
                        label="Top-P",
                        info="Nucleus sampling threshold"
                    )
                rp = gr.Slider(
                    1.0, 2.0, value=1.2, step=0.1,
                    label="Repetition Penalty",
                    info="Penalize repeated tokens"
                )
            
            # Web Search Settings (Collapsible)
            with gr.Accordion("🌐 Web Search Settings", open=False, visible=False) as search_settings:
                mr = gr.Number(
                    value=4, precision=0,
                    label="Max Results",
                    info="Number of search results to retrieve"
                )
                mc = gr.Number(
                    value=50, precision=0,
                    label="Max Chars/Result",
                    info="Character limit per search result"
                )
                st = gr.Slider(
                    minimum=0.0, maximum=30.0, step=0.5, value=5.0,
                    label="Search Timeout (s)",
                    info="Maximum time to wait for search results"
                )
            
            # Actions
            with gr.Row():
                clr = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)
        
        # Right Panel - Chat Interface
        with gr.Column(scale=7):
            chat = gr.Chatbot(
                type="messages",
                height=600,
                label="💬 Conversation",
                show_copy_button=True,
                avatar_images=(None, "🤖"),
                bubble_full_width=False
            )
            
            # Input Area
            with gr.Row():
                txt = gr.Textbox(
                    placeholder="💭 Type your message here... (Press Enter to send)",
                    scale=9,
                    container=False,
                    show_label=False,
                    lines=1,
                    max_lines=5
                )
                with gr.Column(scale=1, min_width=120):
                    submit_btn = gr.Button("📤 Send", variant="primary", size="lg")
                    cancel_btn = gr.Button("⏹️ Stop", variant="stop", visible=False, size="lg")
            
            # Example Prompts
            gr.Examples(
                examples=[
                    ["Explain quantum computing in simple terms"],
                    ["Write a Python function to calculate fibonacci numbers"],
                    ["What are the latest developments in AI? (Enable web search)"],
                    ["Tell me a creative story about a time traveler"],
                    ["Help me debug this code: def add(a,b): return a+b+1"]
                ],
                inputs=txt,
                label="💡 Example Prompts"
            )
            
            # Debug/Status Info (Collapsible)
            with gr.Accordion("🔍 Debug Info", open=False):
                dbg = gr.Markdown()
    
    # Footer
    gr.Markdown("""
    ---
    💡 **Tips:** 
    - Use **Advanced Parameters** to fine-tune creativity and response length
    - Enable **Web Search** for real-time, up-to-date information
    - Try different **models** for various tasks (reasoning, coding, general chat)
    - Click the **Copy** button on responses to save them to your clipboard
    """, elem_classes="footer")

    # --- Event Listeners ---

    # Group all inputs for cleaner event handling
    chat_inputs = [txt, chat, sys_prompt, search_chk, mr, mc, model_dd, max_tok, temp, k, p, rp, st]
    # Group all UI components that can be updated.
    ui_components = [chat, dbg, txt, submit_btn, cancel_btn]

    def submit_and_manage_ui(user_msg, chat_history, *args):
        """
        Orchestrator function that manages UI state and calls the backend chat function.
        It uses a try...finally block to ensure the UI is always reset.
        """
        if not user_msg.strip():
            # If the message is empty, do nothing.
            # We yield an empty dict to avoid any state changes.
            yield {}
            return

        # 1. Update UI to "generating" state.
        #    Crucially, we do NOT update the `chat` component here, as the backend
        #    will provide the correctly formatted history in the first response chunk.
        yield {
            txt: gr.update(value="", interactive=False),
            submit_btn: gr.update(interactive=False),
            cancel_btn: gr.update(visible=True),
        }

        cancelled = False
        try:
            # 2. Call the backend and stream updates
            backend_args = [user_msg, chat_history] + list(args)
            for response_chunk in chat_response(*backend_args):
                yield {
                    chat: response_chunk[0],
                    dbg: response_chunk[1],
                }
        except GeneratorExit:
            # Mark as cancelled and re-raise to prevent "generator ignored GeneratorExit"
            cancelled = True
            print("Generation cancelled by user.")
            raise
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            # If an error happens, add it to the chat history to inform the user.
            error_history = (chat_history or []) + [
                {'role': 'user', 'content': user_msg},
                {'role': 'assistant', 'content': f"**An error occurred:** {str(e)}"}
            ]
            yield {chat: error_history}
        finally:
            # Only reset UI if not cancelled (to avoid "generator ignored GeneratorExit")
            if not cancelled:
                print("Resetting UI state.")
                yield {
                    txt: gr.update(interactive=True),
                    submit_btn: gr.update(interactive=True),
                    cancel_btn: gr.update(visible=False),
                }

    def set_cancel_flag():
        """Called by the cancel button, sets the global event."""
        cancel_event.set()
        print("Cancellation signal sent.")
    
    def reset_ui_after_cancel():
        """Reset UI components after cancellation."""
        cancel_event.clear()  # Clear the flag for next generation
        print("UI reset after cancellation.")
        return {
            txt: gr.update(interactive=True),
            submit_btn: gr.update(interactive=True),
            cancel_btn: gr.update(visible=False),
        }

    # Event for submitting text via Enter key or Submit button
    submit_event = txt.submit(
        fn=submit_and_manage_ui,
        inputs=chat_inputs,
        outputs=ui_components,
    )
    submit_btn.click(
        fn=submit_and_manage_ui,
        inputs=chat_inputs,
        outputs=ui_components,
    )

    # Event for the "Cancel" button.
    # It sets the cancel flag, cancels the submit event, then resets the UI.
    cancel_btn.click(
        fn=set_cancel_flag,
        cancels=[submit_event]
    ).then(
        fn=reset_ui_after_cancel,
        outputs=ui_components
    )

    # Listeners for updating the duration estimate
    duration_inputs = [model_dd, search_chk, mr, mc, max_tok, st]
    for component in duration_inputs:
        component.change(fn=update_duration_estimate, inputs=duration_inputs, outputs=duration_display)

    # Toggle web search settings visibility
    def toggle_search_settings(enabled):
        return gr.update(visible=enabled)
    
    search_chk.change(
        fn=lambda enabled: (update_default_prompt(enabled), gr.update(visible=enabled)),
        inputs=search_chk,
        outputs=[sys_prompt, search_settings]
    )
    
    # Clear chat action
    clr.click(fn=lambda: ([], "", ""), outputs=[chat, txt, dbg])
    
    demo.launch()