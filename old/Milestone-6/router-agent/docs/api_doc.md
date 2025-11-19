# Router Agent API Documentation

## Base URLs
| Component | Base URL (sample) | Auth | Notes |
| --- | --- | --- | --- |
| Gradio UI Space | `https://huggingface.co/spaces/Alovestocode/router-control-room-private` | Optional HF login (private) | Interactive demo + REST endpoints (Gradio `/run/predict`). Replace with the org-hosted slug when public. |
| ZeroGPU FastAPI Backend | `https://Alovestocode-router-router-zero.hf.space` | Requires `HF_TOKEN` if repo private | Hosts merged checkpoints; UI can point `HF_ROUTER_API` here. |

## ZeroGPU REST API
### `GET /health`
Healthcheck (root `/` simply redirects to the Gradio UI).

**Response**
```json
{
  "status": "ok",
  "model": "Alovestocode/router-gemma3-merged",
  "strategy": "8bit"
}
```

### `POST /v1/generate`
Executes the router model.

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `prompt` | string | ✔ | — | Full router prompt (system text + user query). |
| `max_new_tokens` | int | ✖ | 600 | Generation cap (64–1024 safe). |
| `temperature` | float | ✖ | 0.2 | Softmax sampling temperature. |
| `top_p` | float | ✖ | 0.9 | Nucleus sampling.

**Request**
```bash
curl -X POST https://Alovestocode-router-router-zero.hf.space/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "<system>...<user>...",
        "max_new_tokens": 600,
        "temperature": 0.2,
        "top_p": 0.9
      }'
```

**Response**
```json
{
  "text": "{\n  \"route_plan\": [ ... ],\n  ...
}"
```
The `text` field always contains raw JSON (string). Clients should run `json.loads` after trimming any leading thinking blocks.

### `GET /console`
Returns the lightweight HTML console that calls `/v1/generate` without the full Gradio UI.

## Gradio Space REST Hooks
The UI exposes Gradio’s standard `/run/predict` endpoint for automation.

### `POST https://<space>.hf.space/run/predict`
Payload is a JSON object with a `data` list that mirrors the UI controls. The first entry is the user prompt; later entries are dropdowns and toggles.

```bash
curl -X POST https://Alovestocode-router-control-room-private.hf.space/run/predict \
  -H "Content-Type: application/json" \
  -d '{
        "data": [
          "Summarise the differences between AdaGrad and Adam and produce PyTorch code.",
          "Gemma 3 27B Router Adapter",
          600,
          0.2,
          0.9
        ]
      }'
```

**Response skeleton**
```json
{
  "data": [
    {
      "status": "success",
      "plan": { ... validated router JSON ... },
      "raw_text": "...",
      "backend": "Using base model `google/gemma-3-27b-pt` with adapter `router-gemma3-peft`"
    },
    {
      "benchmark_report": {
        "overall": {"json_valid": 0.99, ...},
        "math_first": {...}
      }
    }
  ]
}
```
Indices and exact shapes depend on the UI blocks, so prefer using the Space interactively unless you control both ends. The bundled `test_router_models.py` script demonstrates how to call the backend programmatically without Gradio.

## Authentication
- **HF Spaces (UI)**: attach `Authorization: Bearer <HF_TOKEN>` headers if the Space is private.
- **ZeroGPU API**: private Spaces automatically enforce Hugging Face auth; use `huggingface_hub.InferenceClient` with `token=HF_TOKEN` or pass `-H "Authorization: Bearer <token>"` when curling.

## Versioning & Deprecation
- API version is encoded in the path (`/v1/generate`). Increment when schema or prompt framing changes.
- Document new optional fields (e.g., `primary_guidance`, `primary_computation`) in `docs/user_guide.md` and update tests/benchmarks.
- Backwards-incompatible changes require bumping benchmarks + thresholds; capture the decision in `docs/final_project_report_outline.md`.
