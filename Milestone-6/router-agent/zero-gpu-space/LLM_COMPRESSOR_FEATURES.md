# LLM Compressor & vLLM Advanced Features

This document outlines advanced features from LLM Compressor and vLLM that can be leveraged for better performance and optimization.

## LLM Compressor Features

### 1. Quantization Modifiers

LLM Compressor supports multiple quantization methods beyond AWQ:

#### AWQModifier (Activation-aware Weight Quantization)
```python
from llmcompressor.modifiers.awq import AWQModifier

AWQModifier(
    w_bit=4,              # Weight bits (4 or 8)
    q_group_size=128,     # Quantization group size
    zero_point=True,      # Use zero-point quantization
    version="GEMM"        # Kernel version: "GEMM" or "GEMV"
)
```

#### GPTQModifier (GPTQ Quantization)
```python
from llmcompressor.modifiers.quantization import GPTQModifier

GPTQModifier(
    w_bit=4,              # Weight bits
    q_group_size=128,     # Group size
    desc_act=False,       # Whether to use activation order
    sym=True              # Symmetric quantization
)
```

#### INT8Modifier (8-bit Quantization)
```python
from llmcompressor.modifiers.quantization import INT8Modifier

INT8Modifier(
    w_bit=8,
    q_group_size=128
)
```

### 2. Pruning Modifiers

#### MagnitudePruningModifier
```python
from llmcompressor.modifiers.pruning import MagnitudePruningModifier

MagnitudePruningModifier(
    sparsity=0.5,         # 50% sparsity
    structured=False      # Unstructured pruning
)
```

### 3. Combined Modifiers

You can combine multiple modifiers for maximum compression:

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.pruning import MagnitudePruningModifier

oneshot(
    model="Alovestocode/router-qwen3-32b-merged",
    output_dir="./router-qwen3-compressed",
    modifiers=[
        AWQModifier(w_bit=4, q_group_size=128),
        MagnitudePruningModifier(sparsity=0.1)  # 10% pruning + AWQ
    ]
)
```

## vLLM Advanced Features

### 1. FP8 Quantization (Latest)

vLLM supports FP8 quantization for even better performance:

```python
from vllm import LLM

llm = LLM(
    model="Alovestocode/router-qwen3-32b-merged",
    quantization="fp8",           # FP8 quantization
    dtype="float8_e5m2",          # FP8 format
    gpu_memory_utilization=0.95
)
```

**Benefits:**
- ~2x faster than AWQ
- Lower memory usage
- Better quality retention

### 2. FP8 KV Cache

Reduce KV cache memory usage with FP8:

```python
llm = LLM(
    model="Alovestocode/router-qwen3-32b-merged",
    quantization="awq",
    kv_cache_dtype="fp8",         # FP8 KV cache
    gpu_memory_utilization=0.90
)
```

**Benefits:**
- 50% reduction in KV cache memory
- Enables longer context windows
- Minimal quality impact

### 3. Chunked Prefill (Already Implemented)

```python
enable_chunked_prefill=True  # ✅ Already in our config
```

**Benefits:**
- Better handling of long prompts
- Reduced memory spikes
- Improved throughput

### 4. Prefix Caching (Already Implemented)

```python
enable_prefix_caching=True  # ✅ Already in our config
```

**Benefits:**
- Faster time-to-first-token (TTFT)
- Reuses common prefixes
- Better for repeated prompts

### 5. Continuous Batching (Already Implemented)

```python
max_num_seqs=256  # ✅ Already in our config
```

**Benefits:**
- Dynamic batching
- Better GPU utilization
- Lower latency

### 6. Tensor Parallelism

For multi-GPU setups:

```python
llm = LLM(
    model="Alovestocode/router-qwen3-32b-merged",
    tensor_parallel_size=2,      # Use 2 GPUs
    pipeline_parallel_size=1      # Pipeline parallelism
)
```

### 7. Speculative Decoding

For faster inference with draft models:

```python
llm = LLM(
    model="Alovestocode/router-qwen3-32b-merged",
    speculative_model="small-draft-model",  # Draft model
    num_speculative_tokens=5                # Tokens to speculate
)
```

### 8. SGLang Backend

For even better performance with structured outputs:

```python
llm = LLM(
    model="Alovestocode/router-qwen3-32b-merged",
    enable_lora=True,              # LoRA support
    max_lora_rank=16
)
```

## Recommended Optimizations for Our Use Case

### Current Setup (Good)
- ✅ AWQ 4-bit quantization
- ✅ Continuous batching (max_num_seqs=256)
- ✅ Prefix caching
- ✅ Chunked prefill
- ✅ FlashAttention-2

### Additional Optimizations to Consider

#### 1. FP8 KV Cache (High Impact)
```python
llm_kwargs = {
    "model": repo,
    "quantization": "awq",
    "kv_cache_dtype": "fp8",      # Add this
    "gpu_memory_utilization": 0.95,  # Can increase with FP8 KV
    # ... rest of config
}
```

**Impact:** 50% KV cache memory reduction, longer contexts

#### 2. FP8 Quantization (If Available)
```python
llm_kwargs = {
    "model": repo,
    "quantization": "fp8",        # Instead of AWQ
    "dtype": "float8_e5m2",
    # ... rest of config
}
```

**Impact:** ~2x faster inference, better quality

#### 3. Optimized Sampling Parameters
```python
sampling_params = SamplingParams(
    temperature=0.2,
    top_p=0.9,
    max_tokens=20000,
    stop=["<|end_of_plan|>"],
    skip_special_tokens=False,   # Keep special tokens for parsing
    spaces_between_special_tokens=False
)
```

#### 4. Model Warmup with Real Prompts
```python
def warm_vllm_model(llm, tokenizer):
    """Warm up with actual router prompts."""
    warmup_prompts = [
        "You are the Router Agent. Test task: solve 2x+3=7",
        "You are the Router Agent. Test task: implement binary search",
    ]
    for prompt in warmup_prompts:
        outputs = llm.generate(
            [prompt],
            SamplingParams(max_tokens=10, temperature=0)
        )
```

## Implementation Priority

1. **High Priority:**
   - FP8 KV cache (easy, high impact)
   - Optimized sampling parameters (easy)

2. **Medium Priority:**
   - FP8 quantization (if models support it)
   - Better warmup strategy

3. **Low Priority:**
   - Tensor parallelism (requires multi-GPU)
   - Speculative decoding (requires draft model)

## References

- [vLLM Quantization Docs](https://docs.vllm.ai/en/latest/features/quantization/)
- [LLM Compressor Docs](https://docs.vllm.ai/projects/llm-compressor/)
- [vLLM Performance Guide](https://docs.vllm.ai/en/latest/performance/)
- [FP8 Quantization Paper](https://arxiv.org/abs/2309.06180)

