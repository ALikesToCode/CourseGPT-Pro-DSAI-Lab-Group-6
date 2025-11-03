# OCR Service – Milestone 6 Scaffold (Deployment)

This directory is reserved for the teammate handling OCR deployment. It complements, but is separate from, the router/math/code/general agents.

## Target Capabilities
- Accept images, PDFs, DOCX files, or ZIP archives containing multiple pages.
- Detect whether each page actually contains text.
- Extract recognised text (plain text and optional structured formats).
- Return actionable metadata (confidence scores, language detection, detected layout type).

## Provided Files
- `handler.py` — Template `OCRService` class with the expected entry points.
- `README.md` — This deployment guide (update with implementation details, dependencies, and operational notes).

## Integration Notes
- The Hugging Face Space (see `Milestone-6/router-agent/hf_space/app.py`) can hook into this service to provide OCR utilities alongside the router planner.
- Use environment variables (e.g., `OCR_MODEL_NAME`, `OCR_API_KEY`) to configure third-party services.
- For heavy dependencies (e.g., `pytesseract`, `pdfplumber`), list them in a dedicated requirements file or include instructions for container builds.

## Suggested TODOs
- [ ] Implement text detection + extraction in `handler.py`.
- [ ] Add batching/streaming support for large PDFs.
- [ ] Expose a CLI or REST wrapper for local testing.
- [ ] Produce deployment documentation covering resource requirements and latency benchmarks.
