# General-Search Agent (Milestone 6 Scaffold)

The **General Agent** (a.k.a. General-Search) handles open-ended research, retrieval, and summarisation tasks delegated by the router.

## Responsibilities
- Perform tool-augmented information gathering (web search, vector DB lookup, document QA).
- Return concise summaries with properly formatted citations and metadata.
- Conform to the shared `AgentHandler` interface so the router can treat it like any other tool.

## Implementation Plan
1. Implement the `GeneralSearchAgent` in `handler.py` using your preferred search stack (Google Custom Search, Tavily, SerpAPI, etc.).
2. Normalise results into `AgentResult` with `content`, `citations`, and any collected artifact links.
3. Expose configuration via environment variables (`GENERAL_AGENT_API_KEY`, `SEARCH_PROVIDER`, etc.).
4. Integrate with the Hugging Face Space and router runtime to service `/general-search(...)` calls.

Import example:
```python
from Milestone-6.general-agent.handler import GeneralSearchAgent
```

Add documentation describing rate limits, caching strategy, and quality assurance once the implementation is complete.
