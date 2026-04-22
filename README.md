# AI-Monitors at SemEval-2026 Task 4
### A Hybrid Embedding and LLM Ensemble for Narrative Similarity

> **75% test accuracy · 3rd place out of 47 systems · approaching human annotator ceiling (~78%)**

---

## Overview

This repository contains the code, prompts, and results for the **AI-Monitors** system submitted to
[SemEval-2026 Task 4 (Track A): Narrative Similarity](https://narrative-similarity-task.github.io/).

The task: given an **anchor story** and two **candidate stories (A / B)**, predict which candidate is
more narratively similar to the anchor across three structured dimensions:

| Dimension | Definition |
|---|---|
| Abstract Theme | Central ideas and core motifs; excludes concrete setting |
| Course of Action | Sequence of events, conflicts, decisions, and turning points |
| Outcomes | Final resolution — who succeeds, fails, survives, or dies |

Our system progresses through four hypothesis-driven stages:

```
Embedding Baseline (H1)
      ↓
Aspect-Grounded Prompting (H2)
      ↓
K-Shot Demonstrations + Label Sensitivity (H3)
      ↓
Complementarity-Driven Ensemble (H4)
```

---

## Results

### Embedding Baselines (H1)

| Model | Dev Acc. |
|---|---|
| uhhlt/story-emb | 55.0% |
| qwen3-embedding:8b | 63.5% |
| sentence-t5-xxl | **71.0%** ← selection threshold |

### Full System (H2 → H4)

| System | Dev Acc. | Test Acc. |
|---|---|---|
| Random baseline | 50.0% | 50.0% |
| Jaccard similarity (organiser) | — | 56.3% |
| GPT-4o-mini zero-shot (organiser) | — | 67.0% |
| sentence-t5-xxl embedding (ours) | 71.0% | 63.0% |
| DeepSeek-R1:32b (ours) | 73.7% | 69.0% |
| GPT-5-mini (ours) | 77.5% | 74.0% |
| **Ensemble (ours)** | **81.0%** | **75.0%** |
| Human annotator ceiling | — | ~78.0% |

---

## Repository Structure

```
ai-monitors-semeval2026-task4/
│
├── config.yaml                    # All experiment & model configurations
├── requirements.txt               # Python dependencies
├── run_embedding.py               # Unified embedding inference for all models
├── run_llm.py                     # Unified LLM inference for all models
│
├── scripts/                       # Automation and utilities
│   └── run_experiment.py          # Automated experiment runner (recommended)
│
├── data/
│   └── Track_A/
│       ├── dev_triples.jsonl       # 200 annotated triples (text_a_is_closer bool)
│       ├── test_triples.jsonl      # 400 triples (no ground truth)
│       └── sample_triples.jsonl    # 5 example triples for quick testing
│
├── models/                        # Pre-trained models (embeddings & LLMs)
│   ├── sentence-t5-xxl/           # Sentence-T5-XXL embedding model
│   └── story-emb/                  # Story embedding model
│
├── prompts/                       # Prompt templates (H2 + H3)
│   ├── family1_no_formal_grounding.txt      # Zero-shot baseline prompt
│   ├── family2_aspect_grounded.txt # Aspect-grounded prompt with guidelines
│   └── family3_kshot.txt          # K-shot prompt with neutral label
│
├── ensemble/                      # H4 — ensemble construction
│   ├── complementarity.py         # Agreement rate analysis for ensemble
│   └── majority_vote.py           # 3-way majority vote ensemble
│
└── results/                       # Output predictions (empty initially)
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-monitors-semeval2026-task4.git
cd ai-monitors-semeval2026-task4

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config.yaml` to configure your models and API keys. The file supports:
- Azure OpenAI models (GPT-5-mini)
- Ollama models (DeepSeek-R1, LLaMA, Qwen)
- Hugging Face transformers (T5-XXL, Story-Emb)

### Running Experiments

#### Automated Experiment Runner (Recommended)

Use `scripts/run_experiment.py` to run pre-configured experiments from `config.yaml`:

```bash
# List all available experiments
python scripts/run_experiment.py --list

# Run a specific experiment
python scripts/run_experiment.py --experiment 02_t5_xxl_emb --data dev

# Run with different dataset
python scripts/run_experiment.py --experiment family3_kshot_gpt --data sample

# Preview command without executing (dry run)
python scripts/run_experiment.py --experiment family3_kshot_gpt --dry_run

# Run all experiments on sample data (quick test)
python scripts/run_experiment.py --experiment all --data sample
```

**Available experiments** (shown by `--list`):
- **[H1] Embedding Baselines**: `00_story_emb`, `01_qwen3_emb`, `02_t5_xxl_emb`
- **[H2-H3] LLM Experiments**: `family1_no_formal_grounding_gpt`, `family2_aspect_grounded_gpt`, `family3_kshot_gpt`, `family3_kshot_deepseek`, `family3_kshot_llama`, `family3_kshot_qwen`

#### Direct Command Usage (Advanced)

For manual control, run scripts directly:

**Embedding Baselines (H1)**
```bash
# Run T5-XXL embedding on dev set
python run_embedding.py \
    --model t5-xxl \
    --data data/Track_A/dev_triples.jsonl \
    --output results/dev_t5_xxl.jsonl

# Run Story-Emb on sample data
python run_embedding.py \
    --model story-emb \
    --data data/Track_A/sample_triples.jsonl \
    --output results/sample_story_emb.jsonl
```

**LLM Inference (H2 + H3)**
```bash
# Run GPT-5-mini with k-shot prompting
python run_llm.py \
    --model gpt-5-mini \
    --data data/Track_A/dev_triples.jsonl \
    --prompt prompts/family3_kshot.txt \
    --output results/dev_gpt.jsonl

# Run DeepSeek-R1 with aspect-grounded prompts
python run_llm.py \
    --model deepseek-r1:32b \
    --data data/Track_A/dev_triples.jsonl \
    --prompt prompts/family2_aspect_grounded.txt \
    --output results/dev_deepseek.jsonl
```

#### Ensemble Construction (H4)

```bash
# Analyze complementarity between models
python ensemble/complementarity.py \
    --baseline results/dev_t5_xxl.jsonl \
    --llm results/dev_gpt.jsonl results/dev_deepseek.jsonl \
    --data data/Track_A/dev_triples.jsonl

# Create majority vote ensemble
python ensemble/majority_vote.py \
    --predictions results/dev_t5_xxl.jsonl results/dev_gpt.jsonl results/dev_deepseek.jsonl \
    --output results/dev_ensemble.jsonl
```

---

## Data Format

The dataset consists of JSONL files with the following structure:

```json
{
  "anchor_text": "Story text for the anchor narrative...",
  "text_a": "First candidate story...",
  "text_b": "Second candidate story...",
  "text_a_is_closer": true  // true if A is closer to anchor, false if B
}
```

- `anchor_text`: The reference story
- `text_a` & `text_b`: Two candidate stories to compare
- `text_a_is_closer`: Ground truth label (only in dev set)

---

## Configuration Details

### Models

The `config.yaml` file defines supported models:

- **Embeddings**: sentence-t5-xxl, story-emb, qwen3-embedding
- **LLMs**: gpt-5-mini (Azure), deepseek-r1:32b (Ollama), llama3.1:8b (Ollama), qwen2.5vl:7b (Ollama)

### Prompts

Three prompt families are included:
- `family1_no_formal_grounding.txt`: Basic zero-shot prompting
- `family2_aspect_grounded.txt`: Includes annotation guidelines
- `family3_kshot.txt`: K-shot with neutral examples

### Environment Variables

Set the following for API access:
```bash
# Azure OpenAI (for GPT models)
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"

# Ollama (for local models)
# Ensure Ollama is running with required models
```

---

<!-- ## Future Work

We plan to extend this work with:
- **Contrastive Learning**: Fine-tuning embedding models with contrastive objectives for better narrative similarity
- Additional ensemble methods
- Cross-domain evaluation

---

## Citation

If you use this code, please cite our SemEval-2026 submission:

```bibtex
@inproceedings{ai-monitors-semeval2026,
  title={AI-Monitors at SemEval-2026 Task 4: A Hybrid Embedding and LLM Ensemble for Narrative Similarity},
  author={Your Name et al.},
  booktitle={Proceedings of SemEval},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
├── analysis/                      # ablation studies
│   ├── label_priming.py           # HARD EXAMPLE vs EXAMPLE → Table 4 (+5.6 pts)
│   └── generalization_gap.py      # dev→test gap analysis → Section 4.7
│
└── results/                       # all prediction outputs (.jsonl)
    ├── dev_*.jsonl
    └── test_ensemble.jsonl        ← official submission file
```

--- -->

## Setup

```bash
git clone https://github.com/your-username/ai-monitors-semeval2026-task4.git
cd ai-monitors-semeval2026-task4
pip install -r requirements.txt
```

For models run via Ollama (qwen3-embedding, DeepSeek, LLaMA, Qwen):
```bash
# Install Ollama: https://ollama.com
ollama pull qwen3-embedding:8b
ollama pull deepseek-r1:32b
ollama pull llama3.1:8b
ollama pull qwen2.5vl:7b
```

For GPT-5-mini via Azure OpenAI, set environment variables in `.env`:
```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini-2025-08-07
AZURE_OPENAI_API_VERSION=2025-04-01-preview
```

---

## Quickstart

### Automated Experiment Runner

All experiments are configured in `config.yaml` and run through a unified interface:

```bash
# List all available experiments
python scripts/run_experiment.py --list

# Run a specific experiment on dev data
python scripts/run_experiment.py --experiment 14_K-shot_gpt --data dev

# Run all experiments on sample data (quick test)
python scripts/run_experiment.py --experiment all --data sample

# Dry run to preview commands
python scripts/run_experiment.py --experiment 14_K-shot_gpt --dry_run
```

**Key Features:**
- **Single unified script** for all LLM models (GPT, DeepSeek, LLaMA, Qwen)
- **Model-agnostic**: Switch models by just changing the experiment name
- **Config-driven**: All combinations defined in `config.yaml`
- **Easily extensible**: Add new models/prompts by editing `config.yaml`

### Available Experiments

| Category | Experiment | Model | Type | Dev Acc. |
|---|---|---|---|---|
| **Embedding (H1)** | `00_story_emb` | Story-emb | embedding | 55.0% |
| | `01_qwen3_emb` | Qwen3-embedding | embedding | 63.5% |
| | `02_t5_xxl_emb` | T5-XXL | embedding | 71.0% |
| **LLM (H2+H3)** | `14_K-shot_gpt` | GPT-5-mini | LLM | 77.5% |
| | `06_K-shot_gpt` | GPT-5-mini | LLM | ~74% |
| | `family3_kshot_gpt` | GPT-5-mini | LLM | 77.5% |
| | `06_K-shot_deepseek` | DeepSeek-R1 | LLM | ~73% |
| | `family3_kshot_deepseek` | DeepSeek-R1 | LLM | 73.7% |
| | `family3_kshot_llama` | LLaMA-3.1 | LLM | 58.5% |
| | `family3_kshot_qwen` | Qwen2.5vl | LLM | 67.0% |

### Manual Inference (Advanced)

**For LLM models**, use `run_llm.py`:

```bash
# GPT-5-mini with custom prompt
python run_llm.py \
    --model gpt-5-mini \
    --data data/Track_A/dev_triples.json \
    --prompt prompts/14_K-shot_example.txt \
    --output results/my_output.jsonl

# DeepSeek-R1 with custom prompt
python run_llm.py \
    --model deepseek-r1:32b \
    --data data/Track_A/dev_triples.json \
    --prompt prompts/06_K-shot_llm_examples.txt \
    --output results/my_output.jsonl
```

**For embedding models**, use `run_embedding.py`:

```bash
# T5-XXL embedding
python run_embedding.py \
    --model t5-xxl \
    --data data/Track_A/dev_triples.json \
    --output results/my_output.jsonl

# Qwen3-embedding via Ollama
python run_embedding.py \
    --model qwen3-embedding:8b \
    --data data/Track_A/dev_triples.json \
    --output results/my_output.jsonl

# Story-emb with prefix strategy experimentation
python run_embedding.py \
    --model story-emb \
    --data data/Track_A/dev_triples.json \
    --output results/my_output.jsonl \
    --mode experiment
```

### Step-by-Step Reproducibility

#### Step 1 — Embedding baselines (H1)

```bash
# Run all embedding baselines
python scripts/run_experiment.py --experiment 00_story_emb --data dev
python scripts/run_experiment.py --experiment 01_qwen3_emb --data dev
python scripts/run_experiment.py --experiment 02_t5_xxl_emb --data dev
```

#### Step 2 — LLM inference (H2 + H3)

```bash
# Run best configurations for each model
python scripts/run_experiment.py --experiment family3_kshot_gpt --data dev
python scripts/run_experiment.py --experiment family3_kshot_deepseek --data dev

# Run all experiments (embedding + LLM)
python scripts/run_experiment.py --experiment all --data dev
```

#### Step 3 — Ensemble (H4)

```bash
# Complementarity analysis
python ensemble/complementarity.py \
    --baseline results/dev_00_t5_xxl_embedding.jsonl \
    --models results/dev_14_K-shot_gpt.jsonl results/dev_06_K-shot_deepseek.jsonl \
    --data data/Track_A/dev_triples.json

# Majority vote
python ensemble/majority_vote.py \
    --inputs results/dev_00_t5_xxl_embedding.jsonl results/dev_14_K-shot_gpt.jsonl results/dev_06_K-shot_deepseek.jsonl \
    --data data/Track_A/dev_triples.json \
    --output results/dev_ensemble.jsonl
```

### Configuration System

#### Adding a New Model

Edit `config.yaml`:

**For LLM models:**
```yaml
models:
  my-llm-model:
    type: ollama  # or azure_openai
    model: my-llm-model:7b
    script: run_llm.py
    llm_model: true
```

**For embedding models:**
```yaml
models:
  my-embedding-model:
    type: transformers  # or ollama
    model: my-org/my-embedding-model
    script: run_embedding.py
    embedding_model: true
```

#### Adding a New Prompt

```yaml
prompts:
  my_prompt:
    file: prompts/my_prompt.txt
    description: My custom prompt
```

#### Adding a New Experiment

**For LLM experiments:**
```yaml
experiments:
  my_exp:
    model: my-llm-model
    prompt: my_prompt
    description: My LLM experiment
```

**For embedding experiments (with optional strategy args):**
```yaml
experiments:
  my_emb_exp:
    model: my-embedding-model
    prompt: null
    description: My embedding experiment
    embedding_args:
      mode: run
      strategy: none
```

Then run:
```bash
python scripts/run_experiment.py --experiment my_exp --data dev
```

---

## Key Findings

**H1 — Embedding ceiling confirmed.**
sentence-t5-xxl achieves 71.0% dev accuracy — the strongest embedding baseline —
but all three embedding models plateau below what structured LLM prompting achieves,
confirming that dense embeddings lack explicit narrative reasoning mechanisms.

**H2 — Aspect-grounded prompting works.**
Adding official annotation guideline definitions to the prompt improves GPT-5-mini
from 70.5% to 77.4% — a gain of +6.9 points over unguided zero-shot.

**H3 — A single word matters.**
Labelling a demonstration `HARD EXAMPLE` instead of `EXAMPLE` drops accuracy by
5.6 points. Evaluative descriptors prime the model to treat the example as atypical,
suppressing its reasoning signal during inference.

**H4 — Complementarity beats accuracy-only selection.**
GPT-5-mini (77.5%) and DeepSeek-R1:32b (73.7%) agree with the t5-xxl baseline on
only 67.0% and 61.5% of instances — meaning they recover cases where the baseline
fails. Majority voting over these three produces the strongest overall system.

---

## Data Format

**Dev set** (`.jsonl`, one record per line):
```json
{
  "anchor_text": "...",
  "text_a": "...",
  "text_b": "...",
  "text_a_is_closer": true
}
```

**Test set** (`.jsonl`, one record per line):
```json
{
  "index": 1,
  "anchor_text": "...",
  "text_a": "...",
  "text_b": "..."
}
```

All scripts in this repo accept both formats automatically.

---

## Citation

```bibtex
@inproceedings{aimonitors-semeval2026-task4,
  title     = {AI-Monitors at SemEval-2026 Task 4: A Hybrid Embedding and LLM Ensemble for Narrative Similarity},
  booktitle = {Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year      = {2026},
  address   = {San Diego, CA, USA}
}
```

---

## License

Code released under MIT. Prompt templates and sample data released under CC BY 4.0.