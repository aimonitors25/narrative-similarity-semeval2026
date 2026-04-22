#!/usr/bin/env python3
"""
Unified Embedding Runner (Stable + Production + Multi-model)
"""

import os
import json
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# SAFE JSONL LOADER (FIXED FOR YOUR DATA)
# =========================================================
def load_data(path):
    data = []

    print("Loading dataset safely...")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                r = json.loads(line)
            except Exception:
                print(f"Skipping bad line {i}")
                continue

            data.append({
                "index": r.get("index", i),
                "anchor_text": r["anchor_text"],
                "text_a": r["text_a"],
                "text_b": r["text_b"],
                "label": r.get("text_a_is_closer")
            })

    return data


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_jsonl(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


# =========================================================
# 1. T5-XXL (YOUR WORKING VERSION — CLEANED)
# =========================================================
def load_t5(config):
    from transformers import T5EncoderModel, T5Tokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # MODEL_PATH = config["models"]["t5-xxl"]["path"]
    MODEL_PATH = "models/sentence-t5-xxl"  # Use HuggingFace Hub version
    print("Loading T5-XXL...")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    # model = T5EncoderModel.from_pretrained(
    #     MODEL_PATH,
    #     torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    #     device_map="auto" if DEVICE == "cuda" else None
    # ).to(DEVICE).eval()
    model = T5EncoderModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32    
    ).to(DEVICE).eval()

    def mean_pool(hidden, mask):
        mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def embed(texts):
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            out = model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)

        return emb.cpu().numpy()

    return embed


# =========================================================
# 2. QWEN (OLLAMA)
# =========================================================
def load_qwen(config):
    import requests

    host = config["env"]["ollama_host"]

    def embed(texts):
        out = []

        for t in texts:
            r = requests.post(
                f"{host}/api/embed",
                json={"model": "qwen3-embedding:8b", "input": t},
                timeout=60
            )
            out.append(np.array(r.json()["embeddings"][0], dtype=np.float32))

        return np.array(out)

    return embed


# =========================================================
# 3. STORY EMB
# =========================================================
def load_story():
    from sentence_transformers import SentenceTransformer
    import torch
    import numpy as np

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "models/story-emb"

    print("Loading STORY-EMB (SentenceTransformer mode)...")

    model = SentenceTransformer(model_path, device=DEVICE)

    def embed(texts):
        emb = model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.array(emb)

    return embed

# =========================================================
# CORE RUNNER (SAME LOGIC AS YOUR SCRIPT)
# =========================================================
def run(data, embed_fn):
    results = []
    preds = []
    labels = []

    for r in tqdm(data):

        emb = embed_fn([
            r["anchor_text"],
            r["text_a"],
            r["text_b"]
        ])

        sim_a = np.dot(emb[0], emb[1])
        sim_b = np.dot(emb[0], emb[2])

        pred = "A" if sim_a > sim_b else "B"

        if r["label"] is not None:
            gold = "A" if r["label"] else "B"
            preds.append(pred)
            labels.append(gold)

        results.append({
            "index": r["index"],
            "prediction": pred,
            "anchor_text": r.get("anchor_text"),
            "text_a": r.get("text_a"),
            "text_b": r.get("text_b"),
        
            "sim_a": float(sim_a),
            "sim_b": float(sim_b),
            "ground_truth": "A" if r["label"] else "B" if r["label"] is not None else None
        })


    # accuracy
    if labels:
        acc = accuracy_score(labels, preds)
        print("\n==============================")
        print("RESULTS")
        print("==============================")
        print(f"Accuracy: {acc:.4f}")

    return results


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args()

    print("\n============================================================")
    print("UNIFIED EMBEDDING ENGINE (STABLE)")
    print("============================================================")

    config = load_config(args.config)
    data = load_data(args.data)

    print(f"Samples: {len(data)}")

    # ---------------- MODEL SELECT ----------------
    if args.model == "t5-xxl":
        embed_fn = load_t5(config)

    elif args.model == "qwen3-embedding:8b":
        embed_fn = load_qwen(config)

    elif args.model == "story-emb":
        embed_fn = load_story()

    else:
        raise ValueError("Unknown model")

    # ---------------- RUN ----------------
    results = run(data, embed_fn)

    # ---------------- SAVE ONLY AT END ----------------
    save_jsonl(args.output, results)

    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()