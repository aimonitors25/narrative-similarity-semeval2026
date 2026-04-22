"""
ensemble/majority_vote.py — AI-Monitors SemEval-2026 Task 4
------------------------------------------------------------
Combines predictions from three systems using majority voting:
  1. sentence-t5-xxl embedding baseline (cosine similarity)
  2. GPT-5-mini with family3_kshot_neutral_label.txt
  3. DeepSeek-R1:32b with family2_kshot_llm_examples.txt

Majority voting was chosen over weighted alternatives because:
  - Only three components — ties are impossible with binary A/B choice
  - No reliable confidence scores available across heterogeneous model types
  - Simple majority is robust given three complementary error profiles

Reference:
  Lam & Suen (1997). Application of majority voting to pattern recognition.
  IEEE Transactions on Systems, Man, and Cybernetics, 27(5):553-568.

Paper results (Table 6):
  Dev accuracy:  81.0%  (+3.5 pts over strongest individual, GPT-5-mini)
  Test accuracy: 75.0%  (3rd out of 47 systems)

Tie handling:
  With three binary (A/B) classifiers, a 3-way vote always produces a majority.
  Ties (1-1-1) are impossible. Any tied records are logged and skipped.

Input format (each --inputs file, one record per line):
  Embedding baseline:  {"index": 1, "prediction": "A", "text_a_is_closer": true, ...}
  LLM inference:       {"index": 1, "prediction": "A", "text_a_is_closer": true, ...}

Output format:
  {"index": 1, "anchor_text": "...", "text_a": "...", "text_b": "...", "text_a_is_closer": true}

Usage:
  # Dev set (with accuracy report)
  python ensemble/majority_vote.py \
      --inputs results/dev_t5_xxl.jsonl results/dev_gpt.jsonl results/dev_deepseek.jsonl \
      --data data/Track_A/dev_triples.json \
      --output results/dev_ensemble.jsonl

  # Test set (official submission)
  python ensemble/majority_vote.py \
      --inputs results/test_t5_xxl.jsonl results/test_gpt.jsonl results/test_deepseek.jsonl \
      --output results/test_ensemble.jsonl
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter


# -----------------------------------------------------------------------
# PREDICTION EXTRACTION
# -----------------------------------------------------------------------
def extract_choice(record: dict) -> str | None:
    """
    Extract 'A' or 'B' from a prediction record.
    Handles three output formats produced by scripts in this repo:

      1. Embedding scripts:  {"prediction": "A", ...}
      2. LLM scripts:        {"prediction": "A", ...}  (same — normalised at inference time)
      3. Legacy format:      {"prediction": {"choice": "A"}, ...}
    """
    pred = record.get("prediction")

    # Format 1 & 2: prediction is a string
    if isinstance(pred, str) and pred in ("A", "B"):
        return pred

    # Format 3: prediction is a nested dict (legacy)
    if isinstance(pred, dict):
        choice = pred.get("choice")
        if choice in ("A", "B"):
            return choice

    # Fallback: text_a_is_closer boolean field
    if "text_a_is_closer" in record and record["text_a_is_closer"] is not None:
        return "A" if record["text_a_is_closer"] else "B"

    return None


# -----------------------------------------------------------------------
# LOADING
# -----------------------------------------------------------------------
def load_predictions(file_path: str) -> dict[int, dict]:
    """
    Load a prediction file (.jsonl) and return a dict of {index: record}.
    Skips malformed lines with a warning.
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


def load_ground_truth(data_file: str) -> dict[int, str]:
    """
    Load ground-truth labels from the dev triples file.
    Returns {index: 'A' or 'B'}.
    Only used for accuracy reporting — not required for test set.
    """
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
# MAJORITY VOTE
# -----------------------------------------------------------------------
def majority_vote(all_index_maps: list[dict]) -> list[dict]:
    """
    For each index, collect predictions from all models and take majority vote.
    Skips any index where a tie occurs (logged as warning).
    Returns list of final prediction records.
    """
    # Collect all indices across all files
    all_indices = sorted(
        set(idx for index_map in all_index_maps for idx in index_map.keys())
    )
    print(f"\nTotal unique indices across all files: {len(all_indices)}")

    final_records = []
    tie_count = 0
    missing_count = 0

    for idx in all_indices:
        anchor_text = text_a = text_b = None
        predictions = []

        for index_map in all_index_maps:
            rec = index_map.get(idx)
            if not rec:
                continue

            # Capture story texts from the first file that has them
            if anchor_text is None:
                anchor_text = rec.get("anchor_text", "")
                text_a = rec.get("text_a", "")
                text_b = rec.get("text_b", "")

            choice = extract_choice(rec)
            if choice in ("A", "B"):
                predictions.append(choice)

        # Skip records with missing story texts or no predictions
        if not anchor_text or not predictions:
            missing_count += 1
            continue

        counts = Counter(predictions)
        top = counts.most_common(2)

        # Tie: all three models disagree (e.g. A, B, A ties should not occur
        # with 3 binary classifiers but guard anyway)
        if len(top) > 1 and top[0][1] == top[1][1]:
            print(f"  Warning: tie at index {idx} — predictions: {predictions}. Skipping.")
            tie_count += 1
            continue

        final_choice = top[0][0]

        final_records.append({
            "index": idx,
            "anchor_text": anchor_text,
            "text_a": text_a,
            "text_b": text_b,
            "text_a_is_closer": final_choice == "A",   # official submission field
            "prediction": final_choice,
            "votes": dict(counts),
        })

    if tie_count:
        print(f"  Ties skipped: {tie_count}")
    if missing_count:
        print(f"  Records skipped (missing data): {missing_count}")

    return final_records


# -----------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------
def evaluate(final_records: list[dict], labels: dict) -> None:
    """Print accuracy against ground-truth labels if available."""
    if not labels:
        print("No ground-truth labels provided — skipping accuracy.")
        return

    correct = sum(
        1 for r in final_records
        if r["index"] in labels and r["prediction"] == labels[r["index"]]
    )
    total = sum(1 for r in final_records if r["index"] in labels)

    if total > 0:
        print(f"\n{'='*60}")
        print("ENSEMBLE RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy : {correct/total:.2%} ({correct}/{total})")
        print(f"Paper result: 81.0% dev / 75.0% test (3rd/47 systems)")

        # Prediction distribution
        a_count = sum(1 for r in final_records if r["prediction"] == "A")
        b_count = len(final_records) - a_count
        print(f"Predictions — A: {a_count} ({a_count/len(final_records):.1%}), "
              f"B: {b_count} ({b_count/len(final_records):.1%})")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Majority-vote ensemble for narrative similarity (H4)."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "Prediction .jsonl files to combine. "
            "Paper order: t5_xxl, gpt, deepseek. "
            "E.g. results/dev_t5_xxl.jsonl results/dev_gpt.jsonl results/dev_deepseek.jsonl"
        ),
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to dev triples file for accuracy reporting. Omit for test set.",
    )
    parser.add_argument(
        "--output",
        default="results/dev_ensemble.jsonl",
        help="Path to save ensemble predictions. Default: results/dev_ensemble.jsonl",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MAJORITY VOTE ENSEMBLE")
    print("=" * 70)
    print(f"Inputs : {args.inputs}")
    print(f"Output : {args.output}")

    # Load all prediction files
    print("\nLoading prediction files...")
    all_index_maps = [load_predictions(p) for p in args.inputs]

    # Run majority vote
    final_records = majority_vote(all_index_maps)
    print(f"Final ensemble records: {len(final_records)}")

    # Evaluate if dev labels provided
    if args.data:
        labels = load_ground_truth(args.data)
        evaluate(final_records, labels)
    else:
        print("No --data provided — skipping accuracy (test set mode).")

    # Save output — two versions:
    # 1. With index and votes (for analysis)
    # os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in final_records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
    print(f"\nSaved {len(final_records)} predictions to: {args.output}")

    # 2. Official submission format (no index, no votes, only required fields)
    submission_path = args.output.replace(".jsonl", "_submission.jsonl")
    with open(submission_path, "w", encoding="utf-8") as f:
        for rec in final_records:
            f.write(json.dumps({
                "anchor_text": rec["anchor_text"],
                "text_a": rec["text_a"],
                "text_b": rec["text_b"],
                "text_a_is_closer": rec["text_a_is_closer"],
            }, ensure_ascii=True) + "\n")
    print(f"Saved official submission to: {submission_path}")