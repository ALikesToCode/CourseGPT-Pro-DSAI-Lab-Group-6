---
title: Router Agent Training Dataset
datasets:
- Alovestocode/Router-agent-data
tags:
- router
- tool-use
- synthetic
- large-language-models
license: mit
---

# Router Agent Training Dataset

## Overview

This dataset contains synthetic router-training examples produced with the Gemini 2.5 Pro dataset generator from the CourseGPT-Pro project. Each JSON line includes:

- `user_query` and `task_summary`
- A detailed `route_plan`, `route_rationale`, and `handoff_plan`
- Acceptance criteria, metrics, compute budget, reproducibility contract, and citation policy
- A TODO checklist with verification steps and router QA closure
- Difficulty tier, topical tags, quality score, and evaluation guidance

These records are designed to train or benchmark an LLM router that coordinates math, coding, and general-search agents.

## Files

```
output.jsonl    # one JSON object per line
```

## Usage

```python
from datasets import load_dataset

ds = load_dataset("Alovestocode/Router-agent-data")
print(ds["train"][0])
```

Each record is self-contained and includes metadata fields to automate grading and orchestration.

## Generation Notes

- Generated with `Milestone-2/router-agent-scripts/gemini_router_dataset.py`
- Sequential ID repair ensured `router_XXXX` naming matches file order
- Gemini API concurrency was throttled to respect per-minute quotas
- Themes and tags rotate through a freshness window to encourage diversity

## Citation

If you find this dataset useful in academic or industrial projects, please cite both the CourseGPT-Pro repository and the Gemini 2.5 Pro API:

```
@misc{RouterAgentDataset2025,
  title        = {Router Agent Training Dataset},
  author       = {CourseGPT-Pro Team},
  howpublished = {\url{https://huggingface.co/datasets/Alovestocode/Router-agent-data}},
  year         = {2025}
}
```

## License

The dataset is distributed under the MIT License. Refer to the CourseGPT-Pro repository for the license text.

