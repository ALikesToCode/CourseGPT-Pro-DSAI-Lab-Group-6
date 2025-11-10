# AWQ Quantization Guide for Router Models

This guide explains how to quantize the CourseGPT-Pro router models to AWQ (Activation-aware Weight Quantization) format for efficient inference.

## Models to Quantize

- [router-gemma3-merged](https://huggingface.co/Alovestocode/router-gemma3-merged) (27B, BF16)
- [router-qwen3-32b-merged](https://huggingface.co/Alovestocode/router-qwen3-32b-merged) (33B, BF16)

## Quick Start: Google Colab

1. **Open the Colab notebook**: `quantize_to_awq_colab.ipynb`
2. **Set runtime to GPU**: Runtime → Change runtime type → GPU (A100 recommended)
3. **Add your HF token**: Replace `your_hf_token_here` in cell 2
4. **Run all cells**: The notebook will quantize both models

## Requirements

- **GPU**: A100 (40GB+) or H100 recommended for 27B-33B models
- **Time**: ~30-60 minutes per model
- **Disk Space**: ~20-30GB per quantized model
- **HF Token**: With write access to `Alovestocode` namespace

## AWQ Configuration

The notebook uses optimized AWQ settings:

```python
AWQ_CONFIG = {
    "w_bit": 4,              # 4-bit quantization
    "q_group_size": 128,     # Group size for quantization
    "zero_point": True,      # Use zero-point quantization
    "version": "GEMM",       # GEMM kernel (better for longer contexts)
}
```

## Output Repositories

By default, quantized models are saved to:
- `Alovestocode/router-gemma3-merged-awq`
- `Alovestocode/router-qwen3-32b-merged-awq`

You can modify the `output_repo` in the configuration to upload to existing repos or different names.

## Verification

After quantization, the notebook includes a verification step that:
1. Loads the AWQ model
2. Tests generation
3. Checks parameter count
4. Verifies model integrity

## Usage After Quantization

Update your `app.py` to use the AWQ models:

```python
MODELS = {
    "Router-Gemma3-27B-AWQ": {
        "repo_id": "Alovestocode/router-gemma3-merged-awq",
        "quantization": "awq"
    },
    "Router-Qwen3-32B-AWQ": {
        "repo_id": "Alovestocode/router-qwen3-32b-merged-awq",
        "quantization": "awq"
    }
}
```

## Alternative: Using llm-compressor (vLLM)

If you prefer using llm-compressor (vLLM's quantization tool):

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# Quantize model
oneshot(
    model="Alovestocode/router-gemma3-merged",
    output_dir="./router-gemma3-awq",
    modifiers=[AWQModifier(w_bit=4, q_group_size=128)]
)
```

However, AutoAWQ (used in the notebook) is more mature and widely tested.

## Troubleshooting

### Out of Memory
- Use a larger GPU (A100 80GB or H100)
- Reduce `calibration_dataset_size` to 64 or 32
- Quantize models one at a time

### Slow Quantization
- Ensure you're using a high-end GPU (A100/H100)
- Check that CUDA is properly configured
- Consider using multiple GPUs with `tensor_parallel_size`

### Upload Failures
- Verify HF token has write access
- Check repository exists or can be created
- Ensure sufficient disk space

## References

- [AutoAWQ Documentation](https://github.com/casper-hansen/AutoAWQ)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [vLLM AWQ Support](https://docs.vllm.ai/en/latest/features/quantization/awq.html)

