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
narrative-similarity-semeval2026/
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

```bash
# Clone and install
git clone https://github.com/aimonitors25/narrative-similarity-semeval2026.git
cd narrative-similarity-semeval2026
pip install -r requirements.txt

# Setup environment (Azure OpenAI)
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_DEPLOYMENT="gpt-5-mini-2025-08-07"
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"

# For local models: install Ollama, run: ollama serve
# ollama pull deepseek-r1:32b llama3.1:8b qwen2.5vl:7b qwen3-embedding:8b
```

## Usage

### Recommended: Automated Experiments

Run pre-configured experiments from `config.yaml`:

```bash
python scripts/run_experiment.py --list                  # List all experiments
python scripts/run_experiment.py --experiment 02_t5_xxl_emb --data dev  # Run single
python scripts/run_experiment.py --experiment all --data sample          # Run all
```

### Direct Script Usage

For custom configurations, use core scripts:

**Embeddings (H1):**
```bash
python run_embedding.py --model t5-xxl --data data/Track_A/dev_triples.jsonl --output results/dev_t5_xxl.jsonl
```

**LLM Inference (H2-H3):**
```bash
python run_llm.py --model gpt-5-mini --data data/Track_A/dev_triples.jsonl --prompt prompts/family3_kshot.txt --output results/dev_gpt.jsonl
```

**Ensemble (H4):**
```bash
python ensemble/majority_vote.py --predictions results/dev_t5_xxl.jsonl results/dev_gpt.jsonl results/dev_deepseek.jsonl --output results/dev_ensemble.jsonl
```

```bash
python ensemble/complementarity.py --baseline results/dev_t5_xxl.jsonl --llm results/dev_gpt.jsonl results/dev_deepseek.jsonl --data data/Track_A/dev_triples.jsonl
```

---

## Configuration

Edit `config.yaml` to configure models, API keys, and experiments. Supported:
- **Embedding models:** `t5-xxl`, `story-emb`, `qwen3-embedding:8b`
- **LLM models:** `gpt-5-mini`, `deepseek-r1:32b`, `llama3.1:8b`, `qwen2.5vl:7b`
- **Prompts:** 3 families (zero-shot, aspect-grounded, k-shot) in `prompts/`
- **Experiments:** 13 pre-configured runs (embeddings H1, LLM H2-H3, ensemble H4)

See `config.yaml` for full details and experiment definitions.

## Data Format

**Input (JSONL):**
```json
{
    "anchor_text": "...", 
    "text_a": "...", 
    "text_b": "...", 
    "text_a_is_closer": true
}
```

**Output (JSONL):**
```json
{
    "anchor_text": "...", 
    "text_a": "...", 
    "text_b": "...", 
    "text_a_is_closer": true, 
    "prediction": true, 
    "confidence": 0.92
}
```

---

## Methodology

Our system validates four hypotheses progressively:

**H1 — Embedding Ceiling:** Sentence-T5-XXL achieves 71.0% dev accuracy, but dense embeddings plateau without explicit narrative reasoning.

**H2 — Aspect-Grounded Prompting:** Adding annotation guidelines improves GPT-5-mini from ~70% to 77.4% (+7% gain).

**H3 — Label Sensitivity:** Evaluative language in examples primes model reasoning; changing labels decreased accuracy by 5.6 points.

**H4 — Complementarity > Accuracy:** The 3-way ensemble outperforms individual systems by combining complementary error patterns; models disagree on 30-40% of instances.

---

## Citation

will be added soon...
<!-- If you use this code in your research, please cite our SemEval-2026 submission:

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

MIT License - see [LICENSE](LICENSE) for details.
