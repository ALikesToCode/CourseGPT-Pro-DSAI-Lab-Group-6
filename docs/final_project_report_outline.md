# Final Project Report Outline (Router Agent Focus)
Use this as the backbone for the PDF / DOCX submission. Each section references the milestone artefacts needed to populate it.

1. **Title Page**
   - Project name, team members, date, affiliation.
2. **Abstract**
   - 200-word summary of CourseGPT-Pro with emphasis on router orchestration breakthroughs.
3. **Introduction**
   - Motivation + problem statement. Reference `docs/overview.md` for key framing.
4. **Literature Review (Milestone 1)**
   - Summarise relevant RAG, tool-use, and agentic systems. Cite the sources already catalogued in `README.md`.
5. **Dataset & Methodology (Milestones 2–3)**
   - Recount the Gemini-generated dataset, preprocessing pipeline, and Vertex tuning setup (`technical_doc.md §2`). Include table of split sizes and schema fields.
6. **Model Development & Hyperparameter Tuning (Milestone 4)**
   - Describe adapter choices, Vertex job settings, and ablations (LoRA rank, LR multipliers). Pull numbers from `technical_doc.md §4`.
7. **Evaluation & Analysis (Milestone 5)**
   - Present quantitative metrics (loss, perplexity, BLEU) plus schema-score insights and benchmark subset performance. Include failure analysis (canonical route bias, metrics-field drift).
8. **Deployment & Documentation (Milestone 6)**
   - Detail the Gradio Space, ZeroGPU backend, CI hooks, and benchmarking harness. Link to `docs/technical_doc.md` + `docs/api_doc.md` for reproducibility specifics.
9. **User Experience & Accessibility**
   - Summarise findings from `docs/user_guide.md`, screenshots, and any demo video links.
10. **Conclusion & Future Work**
    - Highlight achievements (live router, schema-validated plans) and list short-term backlog items (math-first data augmentation, automated CI gating, agent plug-ins).
11. **References**
    - Combine literature citations + dataset/model licenses (see `docs/licenses.md`).
12. **Appendix**
    - Include sample router plan JSON, benchmark threshold tables, and deployment logs / commands.

> Tip: Keep the PDF narrative high-level and reference these Markdown files for deep technical detail. Embed the architecture figure from `assets/image1.png` and add a table summarising deployment URLs + hardware tiers.
