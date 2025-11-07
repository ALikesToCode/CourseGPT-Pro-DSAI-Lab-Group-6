# Licensing & References

## 1. Codebase
- The CourseGPT-Pro repository (including `Milestone-6/router-agent/**`) is intended to be released under the **MIT License**. Add the canonical MIT text at the repo root before public launch.
- MIT boilerplate (for quick reference):

```
MIT License

Copyright (c) 2025 CourseGPT-Pro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. Dataset & Data References
| Dataset / Asset | Location | License | Notes |
| --- | --- | --- | --- |
| Router Agent Training Dataset | `https://huggingface.co/datasets/Alovestocode/Router-agent-data` | MIT | Synthetic Gemini 2.5 Pro traces. Include the provided citation in reports. |
| Vertex tuning splits | `Milestone-3/router-agent-scripts/data/vertex_tuning/*.jsonl` | MIT (inherits from above) | Do not redistribute raw student data—already synthetic. |
| Hard benchmark | `Milestone-5/router-agent/benchmarks/router_benchmark_hard.jsonl` | MIT | Generated from the same source dataset. |

## 3. Model Checkpoints
| Model | Source | License / Terms | Usage Notes |
| --- | --- | --- | --- |
| `meta-llama/Llama-3.1-8B-Instruct` | Meta | Llama 3 Community License | Requires agreeing to Meta’s terms; no use in regulated categories without review. |
| `google/gemma-3-27b-pt` | Google | Gemma Terms of Use | Attribute Google and avoid prohibited content; see `https://ai.google.dev/gemma/terms`. |
| `Qwen/Qwen3-32B` | Alibaba | Qwen Community License | Allows commercial use with attribution. |
| `CourseGPT-Pro-DSAI-Lab-Group-6/router-*-peft` | Hugging Face org | MIT (inherits) | Vertex-tuned LoRA adapters. |
| `Alovestocode/router-*-merged` | Hugging Face user | MIT (inherits) | GGUF/merged checkpoints consumed by ZeroGPU.

Ensure HF Spaces referencing these models include the correct license metadata in `space_config.json` or the repo settings.

## 4. Third-Party Libraries
| Library | License | Usage |
| --- | --- | --- |
| Gradio | Apache 2.0 | UI/UX layer for Milestone 6 demo. |
| FastAPI | MIT | ZeroGPU REST API. |
| Transformers / Accelerate | Apache 2.0 | Model loading + inference loops. |
| bitsandbytes | MIT | 8-bit/4-bit quantised loading on ZeroGPU. |
| huggingface_hub | Apache 2.0 | Model + dataset downloads, Space uploads. |
| google-generativeai | Proprietary API TOS | Provides Gemini fallback for agent stubs. |

Verify compatibility with MIT before shipping binaries; Apache 2.0 and MIT are compatible.

## 5. Citations & Attribution
- **Dataset**: include the citation snippet from the dataset card in the final PDF.
- **Models**: mention Meta, Google, and Alibaba/Qwen licenses plus any restrictions in the README/API docs.
- **Toolchains**: cite Vertex AI (training), Hugging Face Spaces (deployment), and Gemini 2.5 Pro (data generation & fallback) wherever results are published.

## 6. Pending Actions
- [ ] Add a repo-level `LICENSE` file (MIT) before making the GitHub project public.
- [ ] Store copies of external licenses referenced above under `docs/licenses/` if offline review is required.
- [ ] Confirm that any future datasets (e.g., MathX-5M subsets) have compatible licenses and update this table accordingly.
