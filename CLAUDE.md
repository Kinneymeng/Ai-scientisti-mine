# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The AI Scientist is a system for fully automated scientific discovery using LLMs. It enables LLMs to independently perform research: generate ideas, run experiments, write papers, and conduct peer review.

**Safety Note:** This codebase autonomously executes LLM-generated code. Always use containerization (see `experimental/Dockerfile`) and restrict web access.

## Core Architecture

### Pipeline Flow
`launch_scientist.py` orchestrates the full pipeline:
1. **Idea Generation** (`ai_scientist/generate_ideas.py`) - Generates ideas, checks novelty via Semantic Scholar/OpenAlex
2. **Experiment Execution** (`ai_scientist/perform_experiments.py`) - Uses Aider to modify code, runs experiments
3. **Paper Writeup** (`ai_scientist/perform_writeup.py`) - Generates LaTeX paper from results
4. **Peer Review** (`ai_scientist/perform_review.py`) - Automated review with reflection and ensembling

### Experiment Constraints
- `MAX_ITERS`: 4 iterations max per experiment (code modification cycles)
- `MAX_RUNS`: 5 total experiment runs allowed
- 2-hour timeout per run
- `NUM_REFLECTIONS`: 3 (for idea generation)

### LLM Interface
`ai_scientist/llm.py` provides unified access to models with backoff/retry logic:
- **Anthropic**: Claude 3/3.5 (direct, Bedrock, Vertex AI)
- **OpenAI**: GPT-4o, GPT-4.1, o1, o3-mini series
- **DeepSeek**: deepseek-chat, deepseek-coder, deepseek-reasoner
- **Google**: Gemini 1.5/2.0/2.5 models
- **OpenRouter**: Llama 3.1-405b

Aider (`aider-chat` package) handles AI-assisted code modification during experiments.

### Template System (`templates/`)
Each template provides a research domain with standardized structure:
- `experiment.py --out_dir <dir>` - Main experiment runner
- `plot.py` - Creates visualizations from `run_*` directories
- `prompt.json` - Contains `system` and `task_description` fields
- `seed_ideas.json` - Example ideas with `Name`, `Title`, `Experiment`, `Interestingness`, `Feasibility`, `Novelty`
- `latex/template.tex` - Paper formatting template (ICLR 2024 format)

**Output format** (`final_info.json`):
```json
{"metric_name": {"means": 0.85, "stds": 0.02}, ...}
```

Official templates: `nanoGPT`, `nanoGPT_lite`, `2d_diffusion`, `grokking`
Community templates: `earthquake-prediction`, `mobilenetV3`, `seir`, `sketch_rnn`, `tensorf`, `MACE`, `probes`

## Development Commands

### Environment Setup
```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist
pip install -r requirements.txt
# Linux: sudo apt-get install texlive-full
```

### Template Preparation (required before experiments)
```bash
# NanoGPT - prepare data first
python data/enwik8/prepare.py
python data/shakespeare_char/prepare.py
python data/text8/prepare.py
cd templates/nanoGPT && python experiment.py --out_dir run_0 && python plot.py

# 2D Diffusion
git clone https://github.com/gregversteeg/NPEET.git && cd NPEET && pip install .
pip install scikit-learn
cd templates/2d_diffusion && python experiment.py --out_dir run_0 && python plot.py

# Grokking
pip install einops
cd templates/grokking && python experiment.py --out_dir run_0 && python plot.py
```

### Running AI Scientist
```bash
# Basic run
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment nanoGPT_lite --num-ideas 2

# Parallel execution (--parallel N sets number of workers)
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment nanoGPT_lite --parallel 4 --gpus "0,1,2,3"

# Skip phases for faster iteration
python launch_scientist.py --skip-idea-generation --experiment nanoGPT_lite  # use existing ideas.json
python launch_scientist.py --skip-novelty-check --experiment nanoGPT_lite    # faster, may have duplicates

# Alternative literature search engine
python launch_scientist.py --engine openalex --experiment nanoGPT_lite

# Enable improvement based on reviews
python launch_scientist.py --improvement --experiment nanoGPT_lite
```

### Paper Review
```python
from ai_scientist.perform_review import load_paper, perform_review
import openai
client = openai.OpenAI()
paper_txt = load_paper('report.pdf')
review = perform_review(paper_txt, 'gpt-4o-2024-05-13', client,
    num_reflections=5, num_fs_examples=1, num_reviews_ensemble=5, temperature=0.1)
print(review['Overall'], review['Decision'])
```

## Configuration

### Environment Variables
| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | GPT-4o, GPT-4.1, o1, o3-mini models |
| `ANTHROPIC_API_KEY` | Claude models (direct API) |
| `DEEPSEEK_API_KEY` | DeepSeek models |
| `GEMINI_API_KEY` | Google Gemini models |
| `S2_API_KEY` | Semantic Scholar (optional, higher throughput) |
| `OPENALEX_MAIL_ADDRESS` | OpenAlex alternative (no API key needed) |
| `OPENROUTER_API_KEY` | Llama 3.1-405b |

**AWS Bedrock**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME` + `pip install anthropic[bedrock]`

**Vertex AI**: `CLOUD_ML_REGION`, `ANTHROPIC_VERTEX_PROJECT_ID`, `VERTEXAI_LOCATION`, `VERTEXAI_PROJECT` + `pip install google-cloud-aiplatform anthropic[vertex]`

### Recommended Models
- **Claude Sonnet 3.5** - Highest success rates (~$15/paper)
- **GPT-4o** - Best for reviews (handles positivity bias)
- **DeepSeek Coder V2** - Most cost-effective

## Generated Outputs
- `templates/{experiment}/ideas.json` - Generated research ideas
- `templates/{experiment}/run_0/` - Baseline results (must create manually, machine-dependent)
- `templates/{experiment}/run_{1,2,3...}/` - AI experiment results
- `templates/{experiment}/run_i.py` - Code snapshots for each run
- `templates/{experiment}/notes.txt` - Experiment descriptions for writeup
- `templates/{experiment}/*.pdf` - Generated papers
- `templates/{experiment}/review.txt` - Review output

## Creating New Templates
1. Copy structure from `nanoGPT_lite/`
2. Implement `experiment.py --out_dir <dir>` outputting `final_info.json`
3. Implement `plot.py` reading from `run_*` directories
4. Create `prompt.json` with `system` and `task_description`
5. Create `seed_ideas.json` with 1-2 examples
6. Update `latex/template.tex` with relevant citations
7. Create baseline: `python experiment.py --out_dir run_0 && python plot.py`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing files | Complete template preparation (data prep, baseline run_0) |
| PDF not generated | Check experiment success; Claude Sonnet 3.5 has highest success rate |
| Review not generated | Use GPT-4o; other models have positivity bias |
| S2 API slow | Use `--engine openalex` or skip novelty check |
| GPU OOM | Use `nanoGPT_lite` or reduce batch size in `experiment.py` |
| Baseline missing | Run `python experiment.py --out_dir run_0` in template directory |

## Key Functions

### LLM Interface (`ai_scientist/llm.py`)
- `create_client(model)` - Factory to create appropriate API client
- `get_response_from_llm()` - Single response with retry logic
- `get_batch_responses_from_llm()` - Multiple responses for ensembling
- `extract_json_between_markers()` - Parse JSON from LLM output

### Idea Generation (`ai_scientist/generate_ideas.py`)
- `generate_ideas()` - Generate ideas with reflection (default 5 reflections)
- `check_idea_novelty()` - Verify via Semantic Scholar/OpenAlex
- `search_for_papers()` - Literature search for citations

### Review (`ai_scientist/perform_review.py`)
- `load_paper(pdf_path)` - Extract text from PDF
- `perform_review()` - Generate review with ensembling
- `perform_improvement()` - Iteratively improve paper based on reviews

## Docker Execution
```bash
docker build -t ai-scientist -f experimental/Dockerfile .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/templates:/app/AI-Scientist/templates \
  ai-scientist --model gpt-4o-2024-05-13 --experiment 2d_diffusion --num-ideas 2
```

## Important
The project is running on an Ubuntu server rather than locally, but users will synchronize all local changes to the server, as well as synchronize the results of the project running on the server to their local devices.