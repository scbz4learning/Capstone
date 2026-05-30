#!/usr/bin/env python3
"""
Run SmolLM / SmolLM2 135M inference using the official model_chat.py script.

Usage:
  # Run all 4 models with a prompt:
  python scripts/run.py -p "What is AI?"

  # Run specific models:
  python scripts/run.py -p "Hello" -m smollm-npu,smollm2-hybrid

  # Run NPU-only models with timing:
  python scripts/run.py -p "Hello" -m npu -t

  # Interactive mode on NPU models:
  python scripts/run.py -m npu -i

  # Use run_model.py instead of model_chat.py:
  python scripts/run.py -p "Hello" --simple
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "venv-ryzen-ai", "LLM", "examples")

MODELS = {
    "smollm-npu": {
        "dir": "models/SmolLM-135M-Instruct_rai_1.7.1_npu_4K",
        "group": "npu",
    },
    "smollm2-npu": {
        "dir": "models/SmolLM2-135M-Instruct_rai_1.7.1_npu_4K",
        "group": "npu",
    },
    "smollm-hybrid": {
        "dir": "models/SmolLM-135M-Instruct_rai_1.7.1_hybrid",
        "group": "hybrid",
    },
    "smollm2-hybrid": {
        "dir": "models/SmolLM2-135M-Instruct_rai_1.7.1_hybrid",
        "group": "hybrid",
    },
}

DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
]

EXAMPLE_PROMPT_FILE = os.path.join(EXAMPLES_DIR, "amd_genai_prompt.txt")
EXAMPLE_PROMPT_LONG_FILE = os.path.join(EXAMPLES_DIR, "amd_genai_prompt_long.txt")


def parse_model_selection(selection):
    all_keys = list(MODELS.keys())
    if selection == "all":
        return all_keys
    if selection == "npu":
        return [k for k, v in MODELS.items() if v["group"] == "npu"]
    if selection == "hybrid":
        return [k for k, v in MODELS.items() if v["group"] == "hybrid"]
    selected = [s.strip() for s in selection.split(",")]
    valid = [s for s in selected if s in MODELS]
    invalid = [s for s in selected if s not in MODELS]
    if invalid:
        print(f"Unknown models: {invalid}. Valid: {list(MODELS.keys())}")
    return valid if valid else all_keys


def get_venv_python():
    candidates = [
        os.path.join(ROOT_DIR, "venv-ryzen-ai", "bin", "python3"),
        os.path.join(ROOT_DIR, "venv-ryzen-ai", "bin", "python"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback to system
    return sys.executable


def get_model_context_length(model_dir):
    config_path = os.path.join(model_dir, "genai_config.json")
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg["model"].get("context_length", 4096)
    except Exception:
        return 4096


def run_inference(model_key, model_info, prompt, args):
    model_dir = os.path.join(ROOT_DIR, model_info["dir"])
    if not os.path.exists(os.path.join(model_dir, "genai_config.json")):
        print(f"  [SKIP] {model_key}: model not found at {model_dir}. Run scripts/download.sh first.")
        return

    context_length = get_model_context_length(model_dir)
    venv_python = get_venv_python()

    if args.simple:
        script = os.path.join(EXAMPLES_DIR, "run_model.py")
        cmd = [venv_python, script, "-m", model_dir]
        if args.timings:
            cmd.extend(["-v", "-ds", "-p", "0.9", "-temp", "0.7"])
        tmp_prompt = None
        if prompt:
            tmp_prompt = os.path.join(ROOT_DIR, ".tmp_prompt.txt")
            with open(tmp_prompt, "w") as f:
                f.write(prompt + "\n")
            cmd.extend(["-pr", tmp_prompt])

        print(f"\n{'=' * 60}")
        print(f"[{model_key}] ({model_info['group']})")
        print(f"{'=' * 60}")
        print(f"  Prompt: {prompt[:60]}{'...' if prompt and len(prompt) > 60 else ''}")
        print(f"  Script: run_model.py")
        print()

        env = os.environ.copy()
        result = subprocess.run(cmd, env=env, cwd=ROOT_DIR)

        if tmp_prompt and os.path.exists(tmp_prompt):
            os.unlink(tmp_prompt)

        if result.returncode != 0:
            print(f"  [FAIL] {model_key}: exit code {result.returncode}")

    else:
        script = os.path.join(EXAMPLES_DIR, "model_chat.py")
        cmd = [venv_python, script, "-m", model_dir, "-mpt", str(context_length)]

        if args.timings:
            cmd.append("-tm")

        if prompt:
            tmp_prompt = os.path.join(ROOT_DIR, ".tmp_prompt.txt")
            with open(tmp_prompt, "w") as f:
                f.write(prompt)
            cmd.extend(["-pr", tmp_prompt])

        cmd.extend(["-ds", "-p", "0.9", "-k", "50", "-t", "0.7"])

        print(f"\n{'=' * 60}")
        print(f"[{model_key}] ({model_info['group']})")
        print(f"{'=' * 60}")
        print(f"  Prompt: {prompt[:60]}{'...' if prompt and len(prompt) > 60 else ''}")
        print(f"  Script: model_chat.py")
        print()

        env = os.environ.copy()
        result = subprocess.run(cmd, env=env, cwd=ROOT_DIR)

        tmp_path = os.path.join(ROOT_DIR, ".tmp_prompt.txt")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        if result.returncode != 0:
            print(f"  [FAIL] {model_key}: exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SmolLM / SmolLM2 135M on Ryzen AI (wrapper around official scripts)"
    )
    parser.add_argument("-p", "--prompt", type=str, default=None,
                        help="Input prompt")
    parser.add_argument("-f", "--prompt-file", type=str, default=None,
                        help="Path to .txt prompt file (one line per prompt)")
    parser.add_argument("-m", "--models", type=str, default="all",
                        help="Models: all, npu, hybrid, or comma-separated keys")
    parser.add_argument("-t", "--timings", action="store_true",
                        help="Print TTFT and TPS timing stats")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Interactive mode (one model at a time)")
    parser.add_argument("--simple", action="store_true",
                        help="Use run_model.py instead of model_chat.py")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for key, info in MODELS.items():
            print(f"  {key:20s} ({info['group']}) -> {info['dir']}")
        return

    models_to_run = parse_model_selection(args.models)
    if not models_to_run:
        print("No valid models selected.")
        sys.exit(1)
    print(f"Models: {models_to_run}")

    # Build prompt list
    prompts = []
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    elif args.interactive:
        prompts = None
    else:
        prompts = DEFAULT_PROMPTS

    if prompts:
        for model_key in models_to_run:
            info = MODELS[model_key]
            for prompt in prompts:
                run_inference(model_key, info, prompt, args)
    else:
        # Interactive mode
        print("\nInteractive mode. Type 'quit()' to exit.")
        while True:
            try:
                text = input("Prompt: ")
            except EOFError:
                break
            if not text or text == "quit()":
                break
            for model_key in models_to_run:
                info = MODELS[model_key]
                run_inference(model_key, info, text, args)

    print("\nDone.")


if __name__ == "__main__":
    main()