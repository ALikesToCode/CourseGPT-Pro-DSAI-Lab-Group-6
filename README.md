# CourseGPT: A LangGraph-Based Student Helper Chatbot

## Title Page

**Project Title:** CourseGPT — A LangGraph-Based Student Helper Chatbot  
**Student/Team Details:** _[Add Your Name / Team Name]_  
**Institution/Department Details:** _[University / Department Information]_  
**Date:** _[Submission Date]_  

---

![CourseGPT Cover](assets/coursegpt_cover.png)


## 1. Abstract

CourseGPT is an intelligent educational assistant designed using a LangGraph-based agentic architecture to support students with subject-specific tasks in mathematics, programming, and general academic queries. Unlike generic chatbots, which often struggle with precise reasoning and specialized tasks, CourseGPT uses a multi-agent workflow to route queries to domain-specialized agents. The system integrates FastAPI for backend API services and Streamlit for an interactive web-based frontend. Experimental evaluation indicates improved intent classification accuracy, better performance on math and coding tasks, and acceptable latency, demonstrating CourseGPT’s effectiveness as a practical student helper chatbot.

---

## 2. Introduction

### 2.1. Project Overview

Large Language Models (LLMs) have transformed how students access educational assistance by enabling natural language interaction with powerful models. However, single-model conversational systems often lack domain specialization and structured workflows. CourseGPT addresses this gap by designing a multi-agent, LangGraph-based educational assistant tailored for common student needs such as solving math problems, writing and debugging code, and answering general academic questions.

### 2.2. Problem Statement

Generic chatbots are typically optimized for broad conversational ability rather than deep, structured reasoning. This leads to several issues in academic contexts:

- Inconsistent accuracy for mathematical calculations and proofs.
- Hallucinated or incorrect code in programming tasks.
- Lack of clear separation between types of queries (e.g., mathematical vs. conceptual vs. coding).
- No explicit routing mechanism to decide which “expert” logic should handle a query.

These limitations motivate the need for a system that can intelligently route user queries to specialized agents.

### 2.3. Objectives

The main objectives of this project are:

- To design and implement a **multi-agent educational assistant** using LangGraph.
- To build specialized agents for:
  - Mathematical problem solving.
  - Programming/code-related assistance.
  - General academic and conceptual queries.
- To develop a **routing mechanism** that classifies user intent and dispatches queries to the appropriate agent.
- To provide a user-friendly interface and scalable backend suitable for real student usage.

### 2.4. Scope of the Project

The initial scope of CourseGPT is limited to three major categories of tasks:

- **Math:** Algebra, basic calculus, numeric problem-solving, and step-wise explanations.
- **Programming:** Code generation, error analysis, debugging, and conceptual explanations (initially focusing on Python; extensible to other languages).
- **General Queries:** Explanations of concepts, definitions, summarization, and general-purpose Q&A.

Out-of-scope items for the current iteration include advanced domain-specific tools (e.g., physics simulation engines), full exam-generation systems, and deep integration with institutional learning management systems.

---

## 3. Literature Review (Milestone 1)

### 3.1. Evolution of Educational Chatbots

Educational chatbots have evolved significantly over the last few decades:

- **Rule-Based Systems:** Early systems such as ELIZA depended on pattern matching and hand-crafted rules. They lacked true understanding and were difficult to scale.
- **Traditional NLP & ML-Based Systems:** With statistical methods, chatbots gained limited contextual understanding but still struggled with complex reasoning.
- **LLM-Based Systems:** Modern transformers and LLMs (e.g., GPT-style models) provide robust language understanding and generation. They support open-ended dialogue and can be adapted to educational tasks with prompt engineering and fine-tuning.

The latest research focuses on combining LLMs with tool use, retrieval systems, and multi-agent orchestration for more reliable, task-specific behavior.

### 3.2. Agentic Workflows vs. RAG

Two prominent architectural paradigms are:

- **Single-Chain or Monolithic LLM Workflows:** A single LLM processes input and returns output. This is simple but not modular, and expertise is not clearly separated.
- **Retrieval-Augmented Generation (RAG):** The LLM is augmented with document retrieval. This improves factual correctness but does not inherently provide task specialization (e.g., math vs. code reasoning).
- **Agentic / Multi-Agent Workflows:** Multiple LLM-based agents, each with specific roles (e.g., router, math expert, coder), are coordinated via an orchestration framework (like LangGraph). This allows:
  - Role specialization.
  - Flexible routing and conditional logic.
  - Better modularity and maintainability.

CourseGPT adopts the agentic design, leveraging LangGraph to finely control how queries move through different agents.

### 3.3. Review of Technologies

- **LangGraph & LangChain:**
  - LangChain provides tools, chains, and utilities to work with LLMs.
  - LangGraph extends this by enabling graph-based workflows, where agents (nodes) are connected by edges with conditional logic.
- **FastAPI (Backend):**
  - An asynchronous web framework in Python.
  - Ideal for high-performance JSON APIs.
  - Supports async endpoints, which is important for LLM inference calls.
- **Streamlit (Frontend):**
  - A Python-based rapid prototyping framework for web apps.
  - Allows quick development of interactive UIs.
  - Well-suited for building chat-like interfaces and visualizing results without complex frontend code.

These technologies align well with the project’s needs: modular backend orchestration, fast API endpoints, and a simple interactive frontend.

---

## 4. Dataset and Methodology (Milestones 2–3)

### 4.1. System Architecture

The overall system architecture is organized into three main layers:

- **Frontend (Streamlit):**
  - Presents a chat interface where students can type questions.
  - Handles session management for ongoing conversations.
- **Backend (FastAPI):**
  - Exposes HTTP endpoints for processing messages.
  - Receives user input from the frontend and forwards it to the agent layer.
  - Returns structured responses (text, code blocks, explanations) to the frontend.
- **Agent Layer (LangGraph):**
  - Implements a graph of agents:
    - Router Agent.
    - Math Agent.
    - Programming Agent.
    - General Agent.
  - The router decides which agent handles the user’s query.
  - Agents can pass state/context as needed.

**Tech Stack Selection:**

- **Language:** Python (due to ecosystem support for LLM tooling).
- **Orchestration:** LangGraph on top of LangChain.
- **Backend Framework:** FastAPI for asynchronous REST APIs.
- **Frontend Framework:** Streamlit for rapid UI development.

### 4.2. Agentic Workflow Design (The Methodology)

#### 4.2.1. The Logic Core: Understanding the Conditional Edge (Routing Logic)

At the heart of LangGraph in CourseGPT lies the routing logic:

- The **Router Agent** inspects the user query.
- Based on content and intent, it selects one of the downstream agents:
  - Math Agent.
  - Programming Agent.
  - General Agent.
- This decision is typically encoded as a **conditional edge** in LangGraph, which routes the state to different nodes depending on the router’s output.

For example, if the user asks, “Solve 2x + 3 = 7”, the router classifies it as a math query and forwards it to the Math Agent. If the user asks, “Why is my Python function returning None?”, it is routed to the Programming Agent.

#### 4.2.2. Intent Classification

The intent classification leverages both heuristic patterns and LLM reasoning:

- **Keyword-based Hints:**
  - Presence of terms such as “integral”, “solve for x”, “limit”, “equation” biases toward Math.
  - Terms like “Python”, “compile error”, “stack trace”, “function”, “class” bias toward Programming.
- **LLM-Assisted Classification:**
  - A lightweight LLM call (router prompt) analyzes the query and outputs a label (MATH / CODE / GENERAL).
- The router’s prompt explicitly instructs the model:
  - To classify queries into one of the three categories.
  - To be conservative in ambiguous cases and route to General when unsure.

Agentic workflow diagram

![Agent Graph](assets/agent_graph.png)


#### 4.2.3. Inter-Agent Communication: State Management

LangGraph maintains a **shared state** that can be passed across nodes:

- State includes:
  - User query.
  - Conversation history (where needed).
  - Intermediate results.
- This allows:
  - Multi-turn conversations: the same agent or different agents can refer back to previous answers.
  - Potential future expansions where one agent’s output becomes another agent’s input (e.g., a General Agent drafting a question that the Math Agent then solves).

Architecture diagram

![System Architecture](assets/architecture_diagram.png)


### 4.3. Data Handling

Data handling focuses on how user queries and prompts are processed:

- **Input Processing:**
  - Sanitizing user text.
  - Detecting code blocks.
  - Stripping extra whitespace, etc.
- **Prompt Structuring:**
  - Each agent uses a dedicated system prompt designed for its role.
  - Templates may include:
    - “You are a math expert. Provide step-by-step solutions.”
    - “You are a programming tutor. Generate correct and well-commented code.”
- **Context Window Management:**
  - Only the most relevant parts of conversation history are kept for each agent.
  - Long conversations are summarized when exceeding token limits.
  - This maintains LLM efficiency while preserving needed context.

---

## 5. Model Development and Hyperparameter Tuning (Milestone 4)

### 5.1. The Agent Ecosystem (Model Configuration)

#### 5.1.1. Router Agent

- **Role:** Determine which specialized agent should handle the request.
- **Prompt Characteristics:**
  - Explicit instructions: classify strictly into MATH / PROGRAMMING / GENERAL.
  - Encourage deterministic output format (e.g., JSON or fixed labels).
- **Hyperparameters:**
  - Low temperature (e.g., 0.0–0.2) for consistent routing decisions.
  - Limited maximum tokens since responses are short.

#### 5.1.2. Math Agent

- **Role:** Solve mathematical questions and provide step-by-step explanations.
- **Model Behavior:**
  - Encouraged to show intermediate steps and reasoning.
  - Avoid hallucination in numeric results.
- **Hyperparameters:**
  - Temperature very low (0.0–0.2) to reduce variability.
  - Top-P also relatively low to maintain deterministic reasoning.
- **Tools (Future-Ready):**
  - Ability to integrate with a calculator or symbolic math library.

#### 5.1.3. Programming Agent

- **Role:** Support programming-related tasks, such as code writing, debugging, and explaining snippets.
- **Model Behavior:**
  - Generate syntactically correct and logically coherent code.
  - Explain errors and propose fixes.
- **Hyperparameters:**
  - Temperature very low (0.0) to minimize randomness.
  - Sufficient max tokens to return full code blocks and explanations.

#### 5.1.4. General Agent

- **Role:** Address conceptual, theoretical, and general academic questions.
- **Model Behavior:**
  - More conversational and explanatory.
  - Capable of summarization and high-level reasoning.
- **Hyperparameters:**
  - Moderately higher temperature (e.g., 0.7) to allow more diverse, natural language generation.
  - Balanced Top-P to retain coherence.

### 5.2. Building the Graph

The LangGraph workflow is built as follows:

- **Nodes:**
  - Router Node.
  - Math Node.
  - Programming Node.
  - General Node.
- **Edges:**
  - From Router to each specialized node, conditioned on the router’s classification.
  - Optionally, back to Router or to an end/response node for future refinements.
- **Compilation:**
  - The graph is defined using LangGraph’s API.
  - Once defined, it is compiled into an executable graph that can be invoked by the backend.
- The compiled graph is then integrated into the FastAPI backend, which calls it asynchronously for each request.

### 5.3. Tuning and Optimization

Several rounds of tuning were applied:

- **Prompt Refinement:**
  - Clarified agent roles.
  - Added examples of correct and incorrect behavior in the router prompt.
  - Added detailed instructions for step-by-step reasoning in the Math Agent prompt.
- **Hyperparameter Adjustments:**
  - Lowered temperature and Top-P for Math and Programming to avoid hallucination and randomness.
  - Slightly higher temperature for General Agent for more natural language.
- **Performance vs. Cost Trade-Offs:**
  - Constrained max tokens for the router to reduce cost.
  - Allowed larger context for code and math explanations where needed.

---

## 6. Evaluation & Analysis (Milestone 5)

### 6.1. Testing Strategy

The system was evaluated using both automated tests and manual inspection:

- **Unit Testing (Agent-Level):**
  - Test cases for router decisions (labeled dataset of queries).
  - Math Agent: verification against known correct solutions.
  - Programming Agent: sample problems with expected outputs or correct code patterns.
- **Integration Testing (End-to-End):**
  - Full conversations executed through the frontend.
  - Verifying:
    - Routing correctness.
    - Response coherence.
    - Latency and stability.

Evaluation results (example)

![Evaluation Results](assets/evaluation_results.png)



### 6.3. Comparative Analysis

The CourseGPT agentic approach was compared against a baseline:

- **Baseline:** Single LLM prompt for all queries (no explicit routing, no specialized agents).
- **Findings:**
  - Baseline model sometimes mixed math reasoning with conversational fluff.
  - Code answers were less consistent and sometimes lacked proper structure.
  - CourseGPT provided more reliable math and programming results due to specialization.
  - The agent-based architecture also offered better modularity and made debugging and improvements easier (e.g., only improve the Math Agent without changing others).

---

## 7. Deployment & Documentation (Milestone 6)

### 7.1. Backend Infrastructure (FastAPI)

The backend is implemented with FastAPI:

- **API Endpoints:**
  - `/chat` (POST): receives user message and session context.
  - Internally calls the LangGraph graph execution.
  - Returns a structured JSON response (message text, code, metadata).
- **Request/Response Logic:**
  - Request includes user query and optional conversation history.
  - Backend wraps this into the state for the graph.
  - The response body is formatted in a consistent schema for easy rendering in the frontend.
- **Asynchronous Execution:**
  - Uses `async` endpoints for non-blocking I/O.
  - Supports concurrent user sessions.

### 7.2. Frontend Implementation (Streamlit)

The Streamlit app provides:

- **User Interface Design:**
  - A chat-style interface for sending and receiving messages.
  - Clear separation of user messages and bot responses.
  - Syntax highlighting for code snippets.
- **Session Management:**
  - Streamlit’s `session_state` is used to persist chat history per user session.
  - Each new query is appended to the history and displayed in chronological order.
- **State Persistence:**
  - The frontend sends the conversation history to the backend when needed, allowing consistent multi-turn conversations.

### 7.3. Integration: Connecting Streamlit UI to FastAPI Backend

Integration details:

- Streamlit UI issues HTTP POST requests to FastAPI endpoints using Python libraries such as `requests` or `httpx`.
- FastAPI processes the query via LangGraph and returns the response.
- The Streamlit app parses the response, formats it (e.g., markdown, code blocks), and displays it in the interface.
- Basic error handling is implemented to show fallback messages when backend errors occur.

### 7.4. User Manual / Usage Documentation

The user documentation includes:

- **Getting Started:**
  - How to open the web app.
  - How to begin a conversation.
- **Usage Guidelines:**
  - Example queries for math, code, and general questions.
  - How to phrase questions for best results.
- **Troubleshooting:**
  - What to do if the system is slow or unresponsive.
  - How to handle unclear or incorrect answers.
- **Technical Documentation (for Developers):**
  - Setup instructions (Python environment, dependencies).
  - Configuration of API keys and model settings.
  - Instructions for extending the system (e.g., adding new agents).

---

## 8. Conclusion and Future Work

### 8.1. Summary of Achievements

This project successfully:

- Designed and implemented a **multi-agent LLM-based educational assistant** using LangGraph.
- Built specialized agents for Math, Programming, and General queries.
- Implemented a Router Agent to classify user intent and route queries appropriately.
- Created a full-stack solution with a FastAPI backend and Streamlit frontend.
- Evaluated the system and observed improved reliability and modularity compared to a single-prompt baseline.

### 8.2. Limitations

Despite its success, CourseGPT has several limitations:

- **Limited Subject Coverage:** Currently focused on math, programming, and general queries; lacks dedicated agents for other disciplines.
- **Context Window Constraints:** Long conversations may require summarization, and older context can be lost.
- **Static Tools:** The current implementation may not fully exploit external tools (e.g., live web search, code execution in a secure sandbox).
- **Model and Infrastructure Dependency:** Performance depends on the underlying LLM and available compute resources.

### 8.3. Future Enhancements

Potential future improvements include:

- **Memory Persistence:**
  - Integrate a database or vector store (e.g., PostgreSQL, Chroma, or other) to store user-specific interactions.
  - Enable long-term personalization and recall of previous sessions.
- **External Tools Integration:**
  - Add web search capabilities for up-to-date factual information.
  - Integrate a Python REPL or containerized environment for safe code execution and verification.
- **More Subject-Specific Agents:**
  - Agents for Physics, Chemistry, History, Biology, and others.
  - Specialized prompts and tools (e.g., equation solvers, graphing tools).
- **Advanced Analytics:**
  - Track usage patterns for further tuning.
  - Provide instructors with anonymized aggregated insights (if integrated into learning platforms).

---

## 9. References and Appendix

### References

- LangChain Documentation – LLM Orchestration and Tools.  
- LangGraph Documentation – Graph-based Agent Workflows.  
- FastAPI Documentation – Modern, Fast (High-Performance) Web Framework for Building APIs with Python.  
- Streamlit Documentation – The fastest way to build data apps in Python.  
- Research papers and articles on:
  - LLM-based tutoring systems.
  - Agentic AI workflows.
  - Retrieval-Augmented Generation (RAG).

_(Replace with specific citation formats such as IEEE/APA as required by the institution.)_
