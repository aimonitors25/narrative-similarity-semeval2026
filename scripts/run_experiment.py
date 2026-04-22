#!/usr/bin/env python3
"""
scripts/run_experiment.py — AI-Monitors SemEval-2026 Task 4
-----------------------------------------------------------
Automated experiment runner. Loads experiments from config.yaml and executes them.

Usage:
  # List all available experiments
  python scripts/run_experiment.py --list

  # Run a specific experiment on dev data
  python scripts/run_experiment.py --experiment 02_t5_xxl_emb --data dev

  # Run all experiments on sample data (quick test)
  python scripts/run_experiment.py --experiment all --data sample

  # Dry run to preview commands
  python scripts/run_experiment.py --experiment family3_kshot_gpt --dry_run

  # Run on test set
  python scripts/run_experiment.py --experiment 02_t5_xxl_emb --data test
"""

import os
import sys
import argparse
import yaml
import subprocess
from pathlib import Path


def load_config(config_path="config.yaml"):
    """Load configuration from config.yaml."""
    if not os.path.exists(config_path):
        print(f"ERROR: config.yaml not found at {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_experiment_command(exp_name, exp_config, config, data_choice, dry_run=False):
    """
    Construct the command to run an experiment.
    
    Args:
        exp_name: Name of the experiment
        exp_config: Experiment configuration dict
        config: Full config dict
        data_choice: 'dev', 'test', or 'sample'
    
    Returns:
        Command string to execute
    """
    model_name = exp_config["model"]
    
    if model_name not in config["models"]:
        print(f"ERROR: Model '{model_name}' not found in config")
        return None
    
    model_config = config["models"][model_name]
    script = model_config["script"]
    
    # Determine data file
    data_file = config["data"].get(data_choice)
    if not data_file:
        print(f"ERROR: Invalid data choice '{data_choice}'. Must be: dev, test, or sample")
        return None
    
    # Prepare output path
    output_dir = config.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename based on experiment name and data choice
    output_file = f"{output_dir}/{data_choice}_{exp_name}.jsonl"
    
    # Build command
    cmd = f"python {script} --model {model_name} --data {data_file} --output {output_file}"
    
    # Add prompt if this is an LLM experiment
    if exp_config["prompt"]:
        prompt_config = config["prompts"].get(exp_config["prompt"])
        if prompt_config:
            prompt_file = prompt_config["file"]
            cmd += f" --prompt {prompt_file}"
        else:
            print(f"ERROR: Prompt '{exp_config['prompt']}' not found in config")
            return None
    
    return cmd


def list_experiments(config):
    """List all available experiments."""
    print("\n" + "="*70)
    print("AVAILABLE EXPERIMENTS")
    print("="*70)
    
    experiments = config.get("experiments", {})
    
    if not experiments:
        print("No experiments found in config.yaml")
        return
    
    # Group experiments by type
    embedding_exps = {}
    llm_exps = {}
    
    for exp_name, exp_config in experiments.items():
        model_name = exp_config["model"]
        model_config = config["models"].get(model_name, {})
        
        if model_config.get("embedding_model"):
            embedding_exps[exp_name] = exp_config
        else:
            llm_exps[exp_name] = exp_config
    
    # Print embedding experiments
    if embedding_exps:
        print("\n[H1] EMBEDDING BASELINES:")
        print("-" * 70)
        for name, exp in embedding_exps.items():
            print(f"  {name:<40} → {exp['description']}")
    
    # Print LLM experiments
    if llm_exps:
        print("\n[H2-H3] LLM EXPERIMENTS:")
        print("-" * 70)
        for name, exp in llm_exps.items():
            print(f"  {name:<40} → {exp['description']}")
    
    print("\n" + "="*70)
    print(f"Total: {len(experiments)} experiments")
    print("="*70)


def run_experiment(exp_name, config, data_choice="dev", dry_run=False):
    """
    Run a single experiment.
    
    Args:
        exp_name: Name of the experiment to run
        config: Config dict
        data_choice: 'dev', 'test', or 'sample'
        dry_run: If True, only print the command
    """
    experiments = config.get("experiments", {})
    
    if exp_name not in experiments:
        print(f"ERROR: Experiment '{exp_name}' not found")
        print("Use --list to see available experiments")
        return False
    
    exp_config = experiments[exp_name]
    cmd = get_experiment_command(exp_name, exp_config, config, data_choice, dry_run)
    
    if not cmd:
        return False
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*70}")
    print(f"Description: {exp_config['description']}")
    print(f"Data: {data_choice}")
    print(f"Command: {cmd}")
    print(f"{'='*70}\n")
    
    if dry_run:
        print("[DRY RUN] Command would be executed (use without --dry_run to run)")
        return True
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed with return code {e.returncode}")
        return False


def run_all_experiments(config, data_choice="dev", dry_run=False):
    """
    Run all experiments.
    
    Args:
        config: Config dict
        data_choice: 'dev', 'test', or 'sample'
        dry_run: If True, only print commands
    """
    experiments = config.get("experiments", {})
    
    print(f"\n{'='*70}")
    print(f"RUNNING ALL EXPERIMENTS ({len(experiments)} total)")
    print(f"{'='*70}\n")
    
    results = {}
    for exp_name in sorted(experiments.keys()):
        exp_config = experiments[exp_name]
        success = run_experiment(exp_name, config, data_choice, dry_run)
        results[exp_name] = success
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\nFailed experiments:")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")
    
    print(f"{'='*70}\n")
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Automated experiment runner for AI-Monitors SemEval-2026 Task 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments"
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name to run (or 'all' to run all experiments)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        choices=["dev", "test", "sample"],
        default="dev",
        help="Dataset to use (default: dev)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Handle --list
    if args.list:
        list_experiments(config)
        return 0
    
    # Handle --experiment
    if args.experiment:
        if args.experiment == "all":
            success = run_all_experiments(config, args.data, args.dry_run)
            return 0 if success else 1
        else:
            success = run_experiment(args.experiment, config, args.data, args.dry_run)
            return 0 if success else 1
    
    # If no action specified, print help
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
