# CourseGPT Pro - Licenses & Attribution

This document provides information about licenses for code, models, datasets, and dependencies used in this project.

---

## Table of Contents

1. [Project License](#1-project-license)
2. [Base Model Licenses](#2-base-model-licenses)
3. [Dataset Licenses](#3-dataset-licenses)
4. [Dependencies & Libraries](#4-dependencies--libraries)
5. [Cloud Services](#5-cloud-services)
6. [Third-Party Attributions](#6-third-party-attributions)
7. [User-Generated Content](#7-user-generated-content)

---

## 1. Project License

### CourseGPT Pro Codebase

**License:** [To be determined by project maintainers]

**Recommended:** MIT License or Apache 2.0

**Example MIT License:**
```
MIT License

Copyright (c) 2025 CourseGPT Pro DSAI Lab Group 6

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

---

## 2. Base Model Licenses

### Llama 3.1 8B Instruct

**License:** Llama 3.1 Community License Agreement
**Provider:** Meta AI
**URL:** https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

**Key Terms:**
- ✅ Allowed: Research and commercial use
- ✅ Allowed: Fine-tuning and distribution of fine-tuned models
- ✅ Allowed: Hosting as a service
- ❌ Restricted: Cannot use to improve other LLMs without Meta's permission
- ❌ Restricted: Must comply with Meta's Acceptable Use Policy

**Citation:**
```bibtex
@misc{llama31,
  title={Llama 3.1},
  author={Meta AI},
  year={2024},
  url={https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct}
}
```

---

### Gemma 3 (4B, 27B)

**License:** Gemma Terms of Use
**Provider:** Google DeepMind
**URL:** https://ai.google.dev/gemma/terms

**Key Terms:**
- ✅ Allowed: Research and commercial use
- ✅ Allowed: Fine-tuning and modification
- ✅ Allowed: Redistribution of fine-tuned models
- ❌ Restricted: Must comply with Google's Prohibited Use Policy
- ⚠️ Notice: Attribution to Google required

**Citation:**
```bibtex
@misc{gemma3,
  title={Gemma 3 Model Family},
  author={Google DeepMind},
  year={2024},
  url={https://ai.google.dev/gemma}
}
```

---

### Qwen 3 (0.6B, 32B)

**License:** Apache License 2.0 (with additional terms)
**Provider:** Alibaba Cloud
**URL:** https://huggingface.co/Qwen

**Key Terms:**
- ✅ Allowed: Commercial and non-commercial use
- ✅ Allowed: Modification and distribution
- ✅ Allowed: Fine-tuning and derivative works
- ⚠️ Notice: Must include original license and copyright notice
- ⚠️ Notice: Must state changes if modified

**Citation:**
```bibtex
@misc{qwen3,
  title={Qwen3 Language Models},
  author={Qwen Team, Alibaba Cloud},
  year={2024},
  url={https://huggingface.co/Qwen}
}
```

---

### Llama 4 Scout 17B

**License:** Llama 4 Community License
**Provider:** Meta AI
**URL:** [To be determined - model-specific page]

**Key Terms:** Similar to Llama 3.1 (see above)

---

### Google Gemini 2.5 Flash (API)

**License:** Google Cloud Terms of Service
**Provider:** Google AI
**URL:** https://ai.google.dev/terms

**Key Terms:**
- ✅ Allowed: API usage per service terms
- ❌ Restricted: Cannot extract or replicate model
- ⚠️ Notice: Subject to quota limits and pricing
- ⚠️ Notice: Data sent to API may be used to improve services (opt-out available)

**Pricing:** Pay-per-use (see https://ai.google.dev/pricing)

---

## 3. Dataset Licenses

### MathX-5M

**License:** MIT License
**Source:** `XenArcAI/MathX-5M`
**Hugging Face:** https://huggingface.co/datasets/XenArcAI/MathX-5M

**Key Terms:**
- ✅ Free to use for any purpose
- ✅ No attribution required (but appreciated)
- ⚠️ Provided "as is" without warranty

**Citation:**
```bibtex
@misc{mathx5m,
  title={MathX-5M: A Large-Scale Mathematical Problem Dataset},
  author={XenArcAI},
  year={2024},
  url={https://huggingface.co/datasets/XenArcAI/MathX-5M}
}
```

---

### OpenCoder SFT Stage 2

**License:** Apache License 2.0
**Source:** `OpenCoder-LLM/opc-sft-stage2`
**Hugging Face:** https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2

**Key Terms:**
- ✅ Commercial and non-commercial use allowed
- ✅ Modification and redistribution allowed
- ⚠️ Must include original license notice

**Citation:**
```bibtex
@misc{opencoder_sft,
  title={OpenCoder SFT Stage 2 Dataset},
  author={OpenCoder Team},
  year={2024},
  url={https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2}
}
```

---

### Custom Router Dataset

**License:** Custom (owned by project)
**Source:** Self-generated
**Size:** 8,189 examples

**Key Terms:**
- Owned by project creators
- Can be licensed under MIT or Apache 2.0 if publicly released
- Consider sharing on Hugging Face for community benefit

---

## 4. Dependencies & Libraries

### Python Libraries

| Library | License | URL |
|---------|---------|-----|
| FastAPI | MIT | https://fastapi.tiangolo.com/ |
| Uvicorn | BSD-3-Clause | https://www.uvicorn.org/ |
| LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangGraph | MIT | https://github.com/langchain-ai/langgraph |
| Transformers | Apache 2.0 | https://huggingface.co/docs/transformers |
| PEFT | Apache 2.0 | https://huggingface.co/docs/peft |
| BitsAndBytes | MIT | https://github.com/TimDettmers/bitsandbytes |
| TRL | Apache 2.0 | https://huggingface.co/docs/trl |
| boto3 | Apache 2.0 | https://aws.amazon.com/sdk-for-python/ |
| httpx | BSD-3-Clause | https://www.python-httpx.org/ |
| pypdf | BSD-3-Clause | https://pypdf.readthedocs.io/ |
| python-dotenv | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| pytest | MIT | https://pytest.org/ |
| python-multipart | Apache 2.0 | https://github.com/andrew-d/python-multipart |

**Installation:**
```bash
pip install -r requirements.txt
```

All dependencies are permissively licensed (MIT, BSD, Apache 2.0) and allow commercial use.

---

## 5. Cloud Services

### Cloudflare R2

**Service:** Object storage (S3-compatible)
**Terms:** https://www.cloudflare.com/service-specific-terms-application-services/
**Pricing:** https://www.cloudflare.com/plans/developer-platform/r2-pricing/

**Data Ownership:**
- You retain all rights to your uploaded data
- Cloudflare processes data only per your instructions
- GDPR-compliant

---

### Cloudflare AI Search (AutoRAG)

**Service:** Managed RAG pipeline
**Terms:** https://www.cloudflare.com/service-specific-terms-application-services/
**Pricing:** Based on usage (contact Cloudflare)

**Data Processing:**
- Documents indexed for search
- Embeddings generated automatically
- Data stored in Cloudflare infrastructure

---

### Google Vertex AI

**Service:** Managed ML platform for model training
**Terms:** https://cloud.google.com/terms/service-terms
**Pricing:** https://cloud.google.com/vertex-ai/pricing

**Model Ownership:**
- You own fine-tuned model adapters
- Google provides infrastructure and base models
- Training data processed in Google Cloud

---

### Google Gemini API

**Service:** LLM API access
**Terms:** https://ai.google.dev/terms
**Pricing:** https://ai.google.dev/pricing

**Data Usage:**
- Queries sent to Gemini API
- May be used to improve services (opt-out: set `user_id` for GDPR compliance)

---

## 6. Third-Party Attributions

### Hugging Face

**Platform:** https://huggingface.co/
**Services Used:**
- Model hosting (fine-tuned adapters)
- Dataset hosting (MathX-5M, OpenCoder)
- ZeroGPU Spaces (deployment)

**Acknowledgment:**
```
This project uses models and datasets hosted on Hugging Face.
We thank the Hugging Face team for providing this valuable infrastructure.
```

---

### LangChain / LangGraph

**Framework:** https://langchain.com/
**License:** MIT

**Acknowledgment:**
```
This project uses LangChain and LangGraph for LLM orchestration and multi-agent systems.
```

---

## 7. User-Generated Content

### Uploaded Documents

**Ownership:** Users retain all rights to uploaded content

**Processing:**
- Documents processed in-memory during chat sessions
- Text extracted for RAG indexing (if opted in)
- Not stored permanently by default

**Privacy:**
- User data isolated by `user_id`
- RAG results filtered per user
- Temporary URLs expire after specified duration

**GDPR Compliance:**
- Users can request data deletion
- Right to access stored data
- Right to opt-out of data processing

---

### Conversation Data

**Storage:** In-memory (not persistent by default)

**Recommendations for Production:**
- Implement data retention policies
- Provide user data export functionality
- Allow conversation history deletion

---

## 8. Compliance Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **Commercial Use** | ✅ Allowed | All components permit commercial use |
| **Redistribution** | ✅ Allowed | With proper attribution and license inclusion |
| **Modification** | ✅ Allowed | All components allow modification |
| **Patent Grant** | ✅ Included | Apache 2.0 components include patent grant |
| **GDPR Compliance** | ⚠️ Partial | Implement user data controls for full compliance |
| **Attribution** | ⚠️ Required | Must credit model providers and datasets |
| **Ethical Use** | ⚠️ Required | Must comply with model providers' acceptable use policies |

---

## 9. Recommended Attributions

**In Documentation:**
```
CourseGPT Pro uses the following open-source technologies:

Models:
- Llama 3.1 by Meta AI
- Gemma 3 by Google DeepMind
- Qwen 3 by Alibaba Cloud

Datasets:
- MathX-5M by XenArcAI
- OpenCoder SFT Stage 2 by OpenCoder Team

Frameworks:
- FastAPI by Sebastián Ramírez
- LangChain by LangChain AI
- Hugging Face Transformers
```

**In UI (Footer or About Page):**
```
Powered by Llama 3.1, Gemma 3, and Qwen 3 models.
Training datasets: MathX-5M (MIT), OpenCoder (Apache 2.0).
Infrastructure: Cloudflare R2, Google Vertex AI.
```

---

## 10. License Compliance Checklist

### For Distribution

- [ ] Include original licenses for all dependencies
- [ ] Add attribution for base models
- [ ] Credit dataset providers
- [ ] Include copy of project license (MIT/Apache 2.0)
- [ ] Document any modifications to third-party code
- [ ] Remove or replace any proprietary components

### For Commercial Deployment

- [ ] Review all model licenses for commercial terms
- [ ] Comply with acceptable use policies
- [ ] Set up proper data privacy controls
- [ ] Implement user consent mechanisms
- [ ] Provide terms of service to end users

### For Open Source Release

- [ ] Choose project license (MIT recommended)
- [ ] Add LICENSE file to repository root
- [ ] Include third-party licenses in LICENSES directory
- [ ] Update README with attribution section
- [ ] Document any patent or trademark considerations

---

## 11. Contact & Legal

For licensing questions or permissions:
- **Project Maintainers:** [Your contact information]
- **Institution:** DSAI Lab Group 6
- **Legal Questions:** [Legal contact if applicable]

---

## 12. Disclaimer

```
THE SOFTWARE AND MODELS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

IN NO EVENT SHALL THE AUTHORS, MODEL PROVIDERS, OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE OR MODELS.

Users are responsible for ensuring compliance with all applicable laws and
regulations in their jurisdiction.
```

---

*Last Updated: 2025-01-19*
*License Documentation Version: 1.0*
