# CourseGPT-Pro

**CourseGPT-Pro: A Multimodal, Tool-Augmented RAG System for Technical Homework Assistance**

CourseGPT-Pro is an end-to-end system designed to assist students with technical assignments by integrating multimodal ingestion, hybrid retrieval, and tool-augmented reasoning.

## Project Structure

- `app/`: Streamlit-based frontend user interface.
- `api/`: FastAPI backend service for orchestration and RAG.
- `docs/`: Comprehensive documentation (Technical, User Guide, API).
- `src/`: Core model and pipeline code.
- `notebooks/`: Experimental and training notebooks.
- `data/`: Dataset storage and processing scripts.
- `models/`: Saved model weights and adapters.
- `old/`: Archive of Milestones 1-6.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API Service

```bash
cd api
uvicorn main:app --reload
```

### 3. Run the Frontend App

```bash
cd app
streamlit run src/main.py
```

## Documentation

For detailed information, please refer to the [Documentation](docs/overview.md).

- [Technical Documentation](docs/technical_doc.md)
- [User Guide](docs/user_guide.md)
- [API Documentation](docs/api_doc.md)
