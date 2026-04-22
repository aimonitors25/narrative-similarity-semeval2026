#!/usr/bin/env python3
"""
inference/run_llm.py — AI-Monitors SemEval-2026 Task 4
-------------------------------------------------------
Unified LLM inference for narrative similarity.
Supports multiple models via config.yaml: GPT-5-mini, DeepSeek-R1, LLaMA, Qwen.

Usage:
  python inference/run_llm.py \
      --model gpt-5-mini \
      --data data/Track_A/dev_triples.json \
      --prompt prompts/14_K-shot_example.txt \
      --output results/dev_gpt.jsonl

  python inference/run_llm.py \
      --model deepseek-r1:32b \
      --data data/Track_A/dev_triples.json \
      --prompt prompts/14_K-shot_example.txt \
      --output results/dev_deepseek.jsonl
"""

#!/usr/bin/env python3
"""
Unified LLM Inference (Clean + DRY + Streaming JSONL)
"""

import os
import json
import argparse
import yaml
import requests
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# CONFIG + DATA
# =========================================================

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = [json.loads(l) for l in f] if path.endswith(".jsonl") else json.load(f)

    data = []
    for i, r in enumerate(raw):
        data.append({
            "index": r.get("index", i),
            "anchor_text": r.get("anchor_text") or r.get("anchor"),
            "text_a": r.get("text_a") or r.get("story_a"),
            "text_b": r.get("text_b") or r.get("story_b"),
            "ground_truth": "A" if r.get("text_a_is_closer") else "B" if r.get("text_a_is_closer") is not None else None
        })
    return data


# =========================================================
# UTILS (NO REDUNDANCY)
# =========================================================

def build_prompt(template, record):
    return template.format(
        anchor_text=record["anchor_text"],
        text_a=record["text_a"],
        text_b=record["text_b"]
    )


def parse_choice(text):
    try:
        text = text.strip()

        if text.startswith("{"):
            return json.loads(text).get("choice", "A").upper()

        if text:
            c = text[0].upper()
            if c in ["A", "B"]:
                return c

        return "A" if "A" in text else "B"
    except:
        return "A"


def write_jsonl(f, obj):
    f.write(json.dumps(obj) + "\n")
    f.flush()


# =========================================================
# MODEL CALLERS (MINIMAL LOGIC)
# =========================================================

def ollama_client(model_name, host, timeout=1200):
    def call(prompt):
        r = requests.post(
            f"{host}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,   # IMPORTANT FIX
                "temperature": 0.0
            },
            timeout=timeout
        )
        r.raise_for_status()
        return r.json().get("response", "")
    return call


def azure_client(client, deployment):
    def call(prompt):
        r = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return r.choices[0].message.content
    return call


# =========================================================
# CORE ENGINE (SINGLE LOOP → NO DUPLICATION)
# =========================================================

def run_inference(data, prompt, model_fn, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(data):

            print(f"Processing {i+1}/{len(data)}...", end="\r")

            prompt_text = build_prompt(prompt, record)
            response = model_fn(prompt_text)
            choice = parse_choice(response)

            # result = {
            #     "index": record["index"],
            #     "prediction": choice,
            #     "ground_truth": record["ground_truth"]
            # }
            result = {
                "index": record["index"],
                "anchor_text": record.get("anchor_text"),
                "text_a": record.get("text_a"),
                "text_b": record.get("text_b"),
                "prediction": choice,
                "ground_truth": record.get("ground_truth")
            }

            write_jsonl(f, result)

    print(f"\nSaved → {output_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args()

    print("=" * 60)
    print("UNIFIED INFERENCE (REFRACTORED)")
    print("=" * 60)

    config = load_config(args.config)

    if args.model not in config["models"]:
        raise ValueError(f"Model '{args.model}' not found")

    model_cfg = config["models"][args.model]
    model_type = model_cfg["type"]

    print("Loading data...")
    data = load_data(args.data)

    print("Loading prompt...")
    prompt = load_prompt(args.prompt)

    print(f"Samples: {len(data)}")

    # -------------------------
    # MODEL SELECTION
    # -------------------------
    if model_type == "ollama":
        model_fn = ollama_client(
            model_cfg["model"],
            config.get("env", {}).get("ollama_host", "http://localhost:11434")
        )

    elif model_type == "azure_openai":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        model_fn = azure_client(client, model_cfg["deployment"])

    else:
        raise ValueError("Unknown model type")

    # -------------------------
    # RUN
    # -------------------------
    print("\nRunning inference...\n")
    run_inference(data, prompt, model_fn, args.output)


if __name__ == "__main__":
    main()