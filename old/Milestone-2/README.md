# Milestone 2: Dataset Preparation [October 10]

This milestone focuses on dataset preparation for the CourseGPT-Pro system, involving the collection, identification, and preprocessing of suitable datasets for training specialized agents in STEM education contexts. The work includes preparing data for a code generation agent, a math reasoning agent, and a router agent that orchestrates between them.

## Objectives

- **Collect or Identify Suitable Datasets**: Source high-quality, domain-specific datasets for each agent type
- **Prepare and Preprocess Data**: Clean, format, and structure data for training and validation pipelines

## Overview of Dataset Preparation Components

### 1. Code Agent Dataset Preparation (`code-agent-scripts/`)

**Purpose**: Prepare dataset for fine-tuning a code generation model to assist with programming assignments and algorithmic challenges.

**Dataset Collection & Preparation**:
- **Dataset Identified**: OpenCoder-LLM/opc-sft-stage2 (educational_instruct subset)
- **Data Processing**: Format instruction-response pairs using Llama 3.1 chat template
- **Preprocessing Steps**: Convert to JSONL format, validate code solutions, prepare for SFT training

**Main Files**:
- `finetunening.ipynb`: Data loading, formatting, and preprocessing pipeline
- `README.md`: Detailed dataset overview and preprocessing instructions

**Data Characteristics**: 118K instruction-code pairs with test cases for validation

### 2. Math Agent Dataset Preparation (`math-agent-scripts/`)

**Purpose**: Prepare dataset for training a math reasoning agent capable of solving complex problems with step-by-step explanations.

**Dataset Collection & Preparation**:
- **Dataset Identified**: XenArcAI/MathX-5M
- **Data Processing**: Format math problems with chain-of-thought solutions
- **Preprocessing Steps**: Clean LaTeX formatting, structure for instruction tuning, prepare validation splits

**Main Files**:
- `math_agent_preprocessing.ipynb`: Data loading, formatting, and preprocessing pipeline
- `use_trained_model.sh`: Script for testing preprocessed data
- `Modelfile`: Configuration for data processing
- `requirements.txt`: Python dependencies
- `README.md`: Dataset overview and preprocessing details

**Data Characteristics**: 4.32M math problems with detailed solutions (30% basic, 30% intermediate, 40% advanced)

### 3. Router Agent Dataset Preparation (`router-agent-scripts/`)

**Purpose**: Generate synthetic training data for a router agent that dispatches queries to appropriate specialized agents.

**Dataset Collection & Preparation**:
- **Dataset Generation**: Synthetic data created using Google Gemini 2.5 Pro
- **Data Processing**: Generate structured JSONL with routing decisions, thinking outlines, and verification criteria
- **Preprocessing Steps**: Quality validation, difficulty stratification, format for training

**Main Files**:
- `gemini_router_dataset.py`: CLI tool for generating and preprocessing router datasets
- `requirements.txt`: Dependencies including google-genai SDK
- `README.md`: Comprehensive documentation with data generation examples

**Data Characteristics**: Custom-generated JSONL with routing logic, acceptance criteria, and orchestration plans

## Dataset Preparation Workflow

The dataset preparation follows a systematic approach for each agent:

1. **Dataset Identification**: Research and select appropriate datasets based on agent requirements
2. **Data Collection**: Download and organize raw datasets from sources like Hugging Face
3. **Data Exploration**: Analyze dataset structure, quality, and characteristics
4. **Preprocessing**: Clean, format, and transform data for training pipelines
5. **Validation**: Create training and validation splits, ensure data quality
6. **Documentation**: Record preprocessing steps and dataset characteristics

## Integration and Data Flow

The prepared datasets feed into the CourseGPT-Pro system's agent training pipeline:

1. **Router Agent Data**: Synthetic routing examples for query dispatching logic
2. **Math Agent Data**: Formatted math problems with step-by-step solutions
3. **Code Agent Data**: Instruction-code pairs with test cases for programming tasks
4. **General Search Agent**: Web/RAG data sources (prepared separately)

## Common Prerequisites

- Python 3.8+ environment
- Access to dataset repositories (Hugging Face, etc.)
- API keys for synthetic data generation (Google Gemini)
- Sufficient storage for large datasets (several GB)

## Getting Started with Dataset Preparation

1. Review the objectives and select target datasets for each agent
2. Set up development environment and install dependencies
3. Download and explore raw datasets
4. Run preprocessing notebooks/scripts in order:
   - Code agent: Load OpenCoder dataset and format for instruction tuning
   - Math agent: Process MathX-5M dataset with LaTeX cleaning and formatting
   - Router agent: Generate synthetic routing data using Gemini API
5. Validate preprocessing outputs and create train/validation splits
6. Document preprocessing steps and data characteristics

## Data Quality Assurance

- All datasets undergo quality checks and filtering
- Preprocessing includes format validation and error handling
- Training/validation splits ensure representative data distribution
- Documentation includes data statistics and preprocessing decisions

## Notes

- Datasets are selected for their relevance to STEM education contexts
- Preprocessing focuses on creating high-quality training signals
- Data preparation enables efficient model training and evaluation
- Prepared datasets serve as foundation for subsequent agent fine-tuning

For detailed preprocessing instructions and dataset specifications, refer to the individual README.md files in each subdirectory.
