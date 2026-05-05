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
├── CITATION.cff                   # Citation metadata for the project
├── LICENSE                        # MIT License
├── README.md                      # This file
├── config.yaml                    # All experiment & model configurations
├── requirements.txt               # Python dependencies
├── run_embedding.py               # Unified embedding inference for all models
├── run_llm.py                     # Unified LLM inference for all models
│
├── data/                          # Dataset directory
│   └── Track_A/
│       ├── dev_triples.jsonl      # 200 annotated triples (text_a_is_closer bool)
│       ├── test_triples.jsonl     # 400 triples (no ground truth)
│       └── sample_triples.jsonl   # 5 example triples for quick testing
│
├── models/                        # Pre-trained model checkpoints
│   ├── sentence-t5-xxl/           # Sentence-T5-XXL embedding model
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   └── ...
│   └── story-emb/                 # Story-Embedding model
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.model
│       └── ...
│
├── prompts/                       # LLM prompt templates (H2 + H3)
│   ├── family1_no_formal_grounding.txt   # Zero-shot baseline prompt
│   ├── family2_aspect_grounded.txt       # Aspect-grounded prompt with guidelines
│   └── family3_kshot.txt                 # K-shot prompt with neutral examples
│
├── ensemble/                      # Ensemble methods (H4)
│   ├── complementarity.py         # Agreement rate analysis for ensemble
│   └── majority_vote.py           # 3-way majority vote ensemble combiner
│
├── scripts/                       # Automation and utilities
│   └── run_experiment.py          # Automated experiment runner (recommended)
│
└── results/                       # Output predictions (populated after running)
    ├── dev_*.jsonl                # Development set predictions
    └── test_*.jsonl               # Test set predictions
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/aimonitors25/narrative-similarity-semeval2026.git
cd narrative-similarity-semeval2026

# Install dependencies
pip install -r requirements.txt
```

### Setup & Configuration

**For Azure OpenAI (GPT-5-mini):**
```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_DEPLOYMENT="gpt-5-mini-2025-08-07"
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"
```

**For Ollama Local Models:**

1. [Install Ollama](https://ollama.com)
2. Pull required models:
   ```bash
   ollama pull deepseek-r1:32b
   ollama pull llama3.1:8b
   ollama pull qwen2.5vl:7b
   ollama pull qwen3-embedding:8b
   ```
3. Ensure Ollama is running: `ollama serve`

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

## Core Scripts

### [run_embedding.py](run_embedding.py)
Unified embedding inference script for all embedding models.

**Usage:**
```bash
python run_embedding.py --model <model_name> --data <input_file> --output <output_file>
```

**Supported Models:** `t5-xxl`, `story-emb`, `qwen3-embedding:8b`

### [run_llm.py](run_llm.py)
Unified LLM inference script for all language models.

**Usage:**
```bash
python run_llm.py --model <model_name> --data <input_file> --prompt <prompt_file> --output <output_file>
```

**Supported Models:** `gpt-5-mini`, `deepseek-r1:32b`, `llama3.1:8b`, `qwen2.5vl:7b`

### [scripts/run_experiment.py](scripts/run_experiment.py)
Automation runner that executes pre-configured experiments from `config.yaml`.

**Key Features:**
- Lists all available experiments: `--list`
- Runs single or multiple experiments: `--experiment <name>`
- Supports different data splits: `--data dev|test|sample`
- Dry-run mode to preview commands: `--dry_run`

### [ensemble/majority_vote.py](ensemble/majority_vote.py)
Constructs 3-way majority vote ensemble from individual model predictions.

**Usage:**
```bash
python ensemble/majority_vote.py --predictions <file1> <file2> <file3> --output <output_file>
```

### [ensemble/complementarity.py](ensemble/complementarity.py)
Analyzes agreement rates between models to identify complementarity.

**Usage:**
```bash
python ensemble/complementarity.py --baseline <file1> --llm <file2> <file3> --data <data_file>
```

---

## Configuration

Edit `config.yaml` to configure your models and API keys. The file supports:
- Azure OpenAI models (GPT-5-mini)
- Ollama models (DeepSeek-R1, LLaMA, Qwen)
- Hugging Face transformers (T5-XXL, Story-Emb)

### Models

Seven models are supported across embedding and LLM categories:

**Embedding Models** (for H1 baseline):
- `sentence-t5-xxl` - Sentence Transformers T5-XXL (71.0% dev accuracy) ← Best embedding baseline
- `story-emb` - Story-specific embedding model (55.0% dev accuracy)
- `qwen3-embedding:8b` - Qwen3 embedding via Ollama (63.5% dev accuracy)

**LLM Models** (for H2-H3 prompting):
- `gpt-5-mini` - Azure OpenAI GPT-5-mini (requires API key)
- `deepseek-r1:32b` - DeepSeek R1 via Ollama (best open-source reasoning)
- `llama3.1:8b` - LLaMA 3.1 8B via Ollama
- `qwen2.5vl:7b` - Qwen 2.5 Vision-Language 7B via Ollama

### Prompts

Three prompt families are included in `prompts/`:

| Family | File | Description | Dev Accuracy |
|--------|------|-------------|--------------|
| Family 1 | `family1_no_formal_grounding.txt` | Zero-shot baseline (H2) | ~66% |
| Family 2 | `family2_aspect_grounded.txt` | Aspect-grounded with annotation guidelines (H2) | ~70% |
| Family 3 | `family3_kshot.txt` | K-shot with neutral examples + label sensitivity (H3) | ~77% |

### Experiments

Pre-configured experiments in `config.yaml` combine models with prompts:

```yaml
# Embedding Baselines (H1)
00_story_emb           # Story-emb baseline
01_qwen3_emb           # Qwen3-embedding baseline
02_t5_xxl_emb          # T5-XXL baseline ← Best embedding

# LLM Experiments (H2-H3)
family1_no_formal_grounding_gpt      # GPT-5-mini with Family 1 prompt
family2_aspect_grounded_gpt          # GPT-5-mini with Family 2 prompt
family3_kshot_gpt                    # GPT-5-mini with Family 3 prompt
family3_kshot_deepseek               # DeepSeek with Family 3 prompt
family3_kshot_llama                  # LLaMA 3.1 with Family 3 prompt
family3_kshot_qwen                   # Qwen 2.5 with Family 3 prompt

# Ensemble (H4)
ensemble_majority_vote               # 3-way majority vote
```

## Support & Questions

For questions about the code, experiments, or methodology, please open an issue on GitHub or contact the AI-Monitors team.

## Data Format

**Input Format (dev/test sets)** — JSONL files:

```json
{
  "anchor_text": "The reference story...",
  "text_a": "First candidate story...",
  "text_b": "Second candidate story...",
  "text_a_is_closer": true
}
```

**Output Format** — All predictions are returned as JSONL with added fields:

```json
{
  "anchor_text": "The reference story...",
  "text_a": "First candidate story...",
  "text_b": "Second candidate story...",
  "text_a_is_closer": true,
  "prediction": true,
  "confidence": 0.92
}
```

---

## Key Insights

**H1 — Embedding Ceiling**  
Sentence-T5-XXL achieves 71.0% dev accuracy, but all embedding models plateau below structured LLM prompting, confirming that dense embeddings lack explicit narrative reasoning.

**H2 — Aspect-Grounded Prompting**  
Adding official annotation guidelines to prompts improves GPT-5-mini from ~70% to 77.4% (+7% gain).

**H3 — Label Sensitivity**  
Changing example labels from `EXAMPLE` to `HARD EXAMPLE` decreases accuracy by 5.6 points, showing how evaluative language primes model reasoning.

**H4 — Complementarity > Accuracy**  
The 3-way ensemble outperforms individual systems by combining complementary strengths: disagreement on 30-40% of instances reveals where each model excels.



<!-- 

---
## Citation

If you use this code in your research, please cite our SemEval-2026 submission:

```bibtex
@inproceedings{ai-monitors-semeval2026,
  title={AI-Monitors at SemEval-2026 Task 4: A Hybrid Embedding and LLM Ensemble for Narrative Similarity},
  author={AI-Monitors Team},
  booktitle={Proceedings of SemEval-2026},
  year={2026}
}
``` -->


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
