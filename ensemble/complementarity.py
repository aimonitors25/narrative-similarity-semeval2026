"""
ensemble/complementarity.py — AI-Monitors SemEval-2026 Task 4
--------------------------------------------------------------
Computes pairwise agreement rates between the t5-xxl embedding baseline
and each LLM to identify complementary error patterns for ensemble
construction (H4, Section 2.5 and Section 4.6).

Key insight (Table 5):
  A large gap between a model's standalone accuracy and its agreement
  rate with t5-xxl indicates the model is frequently correct on instances
  where the embedding baseline fails — signalling useful complementarity.

Paper results (Table 5):
  GPT-5-mini:      77.5% accuracy, 67.0% agreement → gap +10.5 pts ✓ Selected
  DeepSeek-R1:32b: 73.7% accuracy, 61.5% agreement → gap +10.0 pts ✓ Selected
  LLaMA-3.1-8b:    58.5% accuracy — eliminated before this step
  Qwen2.5vl:7b:    67.0% accuracy — eliminated before this step

Error categories per instance:
  Both Correct        → both systems agree and are right
  Only LLM Correct    → LLM recovers a case the embedding misses ← key signal
  Only Embedding Correct → embedding recovers a case the LLM misses
  Both Wrong          → neither system is correct

Usage:
  # Compare one LLM against the embedding baseline
  python ensemble/complementarity.py \
      --baseline results/dev_t5_xxl.jsonl \
      --llm results/dev_gpt.jsonl \
      --data data/Track_A/dev_triples.jsonl \
      --name GPT-5-mini

  # Compare all LLMs at once
  python ensemble/complementarity.py \
      --baseline results/dev_t5_xxl.jsonl \
      --llm results/dev_gpt.jsonl results/dev_deepseek.jsonl \
             results/dev_llama.jsonl results/dev_qwen_inf.jsonl \
      --data data/Track_A/dev_triples.json
"""

#    python ensemble/complementarity.py   --baseline results/story_emb_sample.jsonl  --llm results/qwen3_sample.jsonl results/llama_zero_shot.jsonl    --data data/Track_A/sample_triples.jsonl

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# -----------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------
def load_predictions(file_path: str) -> dict[int, dict]:
    """
    Load a prediction .jsonl file and return {index: record}.
    Handles both string predictions ("A"/"B") and nested dicts.
    """
    index_map = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                index_map[rec["index"]] = rec
            except Exception as e:
                print(f"  Warning: skipping line {line_no} in {file_path}: {e}")
    print(f"  Loaded {len(index_map)} records from {Path(file_path).name}")
    return index_map


def extract_choice(record: dict) -> str | None:
    """
    Extract 'A' or 'B' from a prediction record.
    Handles all output formats in this repo.
    """
    pred = record.get("prediction")

    if isinstance(pred, str) and pred in ("A", "B"):
        return pred

    if isinstance(pred, dict):
        choice = pred.get("choice")
        if choice in ("A", "B"):
            return choice

    if "text_a_is_closer" in record and record["text_a_is_closer"] is not None:
        return "A" if record["text_a_is_closer"] else "B"

    return None


def load_ground_truth(data_file: str) -> dict[int, str]:
    """Load ground-truth labels from dev triples file. Returns {index: 'A'/'B'}."""
    labels = {}
    with open(data_file, "r", encoding="utf-8") as f:
        if data_file.endswith(".jsonl"):
            raw = [json.loads(line) for line in f if line.strip()]
        else:
            raw = json.load(f)

    for i, item in enumerate(raw):
        if "text_a_is_closer" in item and item["text_a_is_closer"] is not None:
            gt = "A" if item["text_a_is_closer"] else "B"
        elif "ground_truth" in item and item["ground_truth"] is not None:
            gt = item["ground_truth"]
        elif "label" in item:
            gt = item["label"]
        else:
            continue
        labels[item.get("index", i + 1)] = gt

    return labels


# -----------------------------------------------------------------------
# MERGE
# -----------------------------------------------------------------------
def build_merged_df(
    baseline_map: dict,
    llm_map: dict,
    labels: dict,
) -> pd.DataFrame:
    """
    Merge baseline and LLM predictions on index.
    Returns a DataFrame with columns:
      index, emb_choice, llm_choice, ground_truth,
      emb_correct, llm_correct, models_agree, error_group
    """
    rows = []
    shared_indices = sorted(set(baseline_map) & set(llm_map))

    for idx in shared_indices:
        emb_choice = extract_choice(baseline_map[idx])
        llm_choice = extract_choice(llm_map[idx])
        gt = labels.get(idx)

        rows.append({
            "index": idx,
            "emb_choice": emb_choice,
            "llm_choice": llm_choice,
            "ground_truth": gt,
        })

    df = pd.DataFrame(rows)

    # Correctness flags — only when ground truth is available
    has_labels = df["ground_truth"].notna()
    df.loc[has_labels, "emb_correct"] = (
        df.loc[has_labels, "emb_choice"] == df.loc[has_labels, "ground_truth"]
    )
    df.loc[has_labels, "llm_correct"] = (
        df.loc[has_labels, "llm_choice"] == df.loc[has_labels, "ground_truth"]
    )

    # Agreement (does not require ground truth)
    df["models_agree"] = df["emb_choice"] == df["llm_choice"]

    # Error group
    def categorize(row):
        if pd.isna(row.get("emb_correct")):
            return "Unknown (no labels)"
        if row["emb_correct"] and row["llm_correct"]:
            return "Both Correct"
        if row["emb_correct"] and not row["llm_correct"]:
            return "Only Embedding Correct"
        if not row["emb_correct"] and row["llm_correct"]:
            return "Only LLM Correct"
        return "Both Wrong"

    df["error_group"] = df.apply(categorize, axis=1)

    return df


# -----------------------------------------------------------------------
# ANALYSIS
# -----------------------------------------------------------------------
def print_complementarity_report(df: pd.DataFrame, llm_name: str) -> dict:
    """
    Print the complementarity report for one LLM vs t5-xxl baseline.
    Returns a summary dict for comparison tables.
    """
    total = len(df)
    labeled = df["ground_truth"].notna().sum()

    # Agreement rate (all instances)
    agreement_rate = df["models_agree"].mean()

    # Accuracy (labeled instances only)
    llm_acc = emb_acc = None
    if labeled > 0:
        llm_acc = df.loc[df["ground_truth"].notna(), "llm_correct"].mean()
        emb_acc = df.loc[df["ground_truth"].notna(), "emb_correct"].mean()

    print(f"\n{'='*60}")
    print(f"COMPLEMENTARITY REPORT: {llm_name}")
    print(f"{'='*60}")
    print(f"Total instances      : {total}")
    print(f"Labeled instances    : {labeled}")
    print(f"Embedding accuracy   : {emb_acc:.2%}" if emb_acc else "Embedding accuracy   : N/A")
    print(f"LLM accuracy         : {llm_acc:.2%}" if llm_acc else "LLM accuracy         : N/A")
    print(f"Agreement rate       : {agreement_rate:.2%}")

    if llm_acc:
        gap = llm_acc - agreement_rate
        print(f"Complementarity gap  : {gap:+.2%}  (accuracy - agreement rate)")
        decision = "✓ SELECTED" if llm_acc > 0.71 and gap > 0.05 else "✗ Examine further"
        print(f"Ensemble decision    : {decision}")

    # Error group breakdown
    if labeled > 0:
        print(f"\nError group breakdown:")
        for group in ["Both Correct", "Only LLM Correct", "Only Embedding Correct", "Both Wrong"]:
            count = (df["error_group"] == group).sum()
            pct = count / labeled * 100
            print(f"  {group:<30}: {count:>4} ({pct:.1f}%)")

    return {
        "model": llm_name,
        "total": total,
        "emb_accuracy": emb_acc,
        "llm_accuracy": llm_acc,
        "agreement_rate": agreement_rate,
        "gap": (llm_acc - agreement_rate) if llm_acc else None,
    }


# -----------------------------------------------------------------------
# VISUALISATION
# -----------------------------------------------------------------------
def plot_complementarity(df: pd.DataFrame, llm_name: str, output_dir: str) -> None:
    """
    Generate a 2x2 complementarity plot:
      1. Error group bar chart
      2. Accuracy comparison
      3. Agreement pie chart
      4. Embedding vs LLM choice confusion matrix
    """
    labeled = df[df["ground_truth"].notna()].copy()
    if labeled.empty:
        print("  No labeled data — skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Complementarity Analysis: t5-xxl vs {llm_name}",
        fontsize=14, fontweight="bold"
    )

    # 1. Error group bar chart
    order = ["Both Correct", "Only Embedding Correct", "Only LLM Correct", "Both Wrong"]
    counts = labeled["error_group"].value_counts().reindex(order, fill_value=0)
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
    axes[0, 0].bar(counts.index, counts.values, color=colors)
    axes[0, 0].set_title("Error Group Distribution")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis="x", rotation=30)
    for i, (_, v) in enumerate(counts.items()):
        axes[0, 0].text(i, v + 1, f"{v/len(labeled)*100:.1f}%", ha="center", fontsize=9)

    # 2. Accuracy comparison
    emb_acc = labeled["emb_correct"].mean()
    llm_acc = labeled["llm_correct"].mean()
    bars = axes[0, 1].bar(["t5-xxl (embedding)", llm_name],
                          [emb_acc, llm_acc], color=["#3498db", "#f39c12"])
    axes[0, 1].set_title("Accuracy Comparison")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim([0, 1.1])
    for bar, acc in zip(bars, [emb_acc, llm_acc]):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.2%}", ha="center", fontweight="bold"
        )

    # 3. Agreement pie chart
    agree = df["models_agree"].sum()
    disagree = len(df) - agree
    axes[1, 0].pie(
        [agree, disagree],
        labels=["Agree", "Disagree"],
        colors=["#2ecc71", "#e74c3c"],
        autopct="%1.1f%%", startangle=90, explode=(0.05, 0)
    )
    axes[1, 0].set_title(f"Prediction Agreement\n(agreement rate: {agree/len(df):.2%})")

    # 4. Choice confusion matrix
    confusion = pd.crosstab(df["emb_choice"], df["llm_choice"])
    sns.heatmap(confusion, annot=True, fmt="d", cmap="YlOrRd",
                ax=axes[1, 1], cbar_kws={"label": "Count"})
    axes[1, 1].set_title("Prediction Matrix")
    axes[1, 1].set_xlabel(f"{llm_name} choice")
    axes[1, 1].set_ylabel("t5-xxl choice")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"complementarity_{llm_name.replace('/', '_')}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved to: {plot_path}")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complementarity analysis between t5-xxl and LLM predictions (H4)."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to t5-xxl embedding predictions. E.g. results/dev_t5_xxl.jsonl",
    )
    parser.add_argument(
        "--llm",
        nargs="+",
        required=True,
        help=(
            "Path(s) to LLM prediction files. "
            "E.g. results/dev_gpt.jsonl results/dev_deepseek.jsonl"
        ),
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dev triples file for ground-truth labels.",
    )
    parser.add_argument(
        "--plots",
        default="results/complementarity_plots",
        help="Directory to save plots. Default: results/complementarity_plots",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("COMPLEMENTARITY ANALYSIS (H4)")
    print("=" * 70)

    # Load baseline and labels
    print("\nLoading baseline predictions...")
    baseline_map = load_predictions(args.baseline)
    labels = load_ground_truth(args.data)
    print(f"Ground-truth labels loaded: {len(labels)}")

    # Run analysis for each LLM
    summaries = []
    for llm_path in args.llm:
        llm_name = Path(llm_path).stem  # e.g. dev_gpt → dev_gpt
        print(f"\nLoading LLM predictions: {llm_name}...")
        llm_map = load_predictions(llm_path)

        df = build_merged_df(baseline_map, llm_map, labels)
        summary = print_complementarity_report(df, llm_name)
        summaries.append(summary)
        plot_complementarity(df, llm_name, args.plots)

    # Summary comparison table
    if len(summaries) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY TABLE (reproduce Table 5 from paper)")
        print(f"{'='*70}")
        print(f"{'Model':<30} {'LLM Acc':>10} {'Agreement':>12} {'Gap':>8} {'Decision':>12}")
        print("-" * 70)
        for s in summaries:
            acc_str = f"{s['llm_accuracy']:.2%}" if s["llm_accuracy"] else "N/A"
            agr_str = f"{s['agreement_rate']:.2%}"
            gap_str = f"{s['gap']:+.2%}" if s["gap"] else "N/A"
            decision = (
                "✓ Selected"
                if s["llm_accuracy"] and s["llm_accuracy"] > 0.71 and s["gap"] > 0.05
                else "Eliminated"
            )
            print(f"{s['model']:<30} {acc_str:>10} {agr_str:>12} {gap_str:>8} {decision:>12}")








# Compare all four LLMs against embedding baseline at once
# python ensemble/complementarity.py --baseline results/dev_t5_xxl.jsonl --llm results/dev_gpt.jsonl results/dev_deepseek.jsonl results/dev_llama.jsonl results/dev_qwen_inf.jsonl --data data/Track_A/dev_triples.json

# Compare only the two selected models (reproduce Table 5)
# python ensemble/complementarity.py --baseline results/dev_t5_xxl.jsonl --llm results/dev_gpt.jsonl results/dev_deepseek.jsonl --data data/Track_A/dev_triples.json