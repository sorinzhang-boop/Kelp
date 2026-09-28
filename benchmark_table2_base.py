#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce Table 2 / Table 9 first row: Qwen3-8B Base latency.

Source-aligned choices:
- Qwen/Qwen3-8B
- Transformers inference
- input target: 1000 tokens
- generation checkpoints: 1 / 512 / 1024 new tokens
- 100 measured runs
- generation settings copied from utils/demo_qwen3_with_guardrail.py:
  enable_thinking=False, top_p=1.0, top_k=0, do_sample=False,
  repetition_penalty=1.0, output_hidden_states=True,
  return_dict_in_generate=True, torch_dtype="auto", device_map="auto"
- warmup pattern follows the released utility: one "hello" warmup plus
  one benchmark-prompt warmup before measured runs.

This script does NOT modify the prompt automatically.
If the chat-templated input is not exactly 1000 tokens, --check-only reports
the mismatch and benchmark mode stops.
"""

import argparse
import inspect
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


#MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_NAME = "/data1/plugguard_repro/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_INPUT_TOKENS = 1000
TARGET_NEW_TOKENS = (1, 512, 1024)


def repo_root():
    return Path(__file__).resolve().parent


def prompt_path():
    return repo_root() / "utils" / "test_sample_1000.txt"


def build_inputs(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        max_length=4096,
        truncation=True,
    )
    return tokenizer([text], return_tensors="pt"), text


def print_environment():
    print("=== ENVIRONMENT ===")
    print("python       :", platform.python_version())
    print("torch        :", torch.__version__)
    print("transformers :", transformers.__version__)
    print("torch CUDA   :", torch.version.cuda)
    print("CUDA avail   :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU 0        :", torch.cuda.get_device_name(0))


def check_input(tokenizer):
    path = prompt_path()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    #prompt = path.read_text(encoding="utf-8")
    prompt = path.read_text(encoding="utf-8").rstrip("\n")
    model_inputs, _ = build_inputs(tokenizer, prompt)

    raw_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    templated_len = int(model_inputs.input_ids.shape[1])

    print("\n=== INPUT CHECK ===")
    print("prompt file                 :", path)
    print("raw prompt tokens           :", len(raw_ids))
    print("chat-templated input tokens :", templated_len)
    print("paper target input tokens   :", EXPECTED_INPUT_TOKENS)
    print("match                       :", templated_len == EXPECTED_INPUT_TOKENS)

    return prompt, templated_len


def generation_kwargs(model, max_new_tokens):
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        top_p=1.0,
        top_k=0,
        do_sample=False,
        repetition_penalty=1.0,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    try:
        if "with_safety_head" in inspect.signature(model.forward).parameters:
            kwargs["with_safety_head"] = False
    except (TypeError, ValueError):
        pass

    return kwargs


def run_generate(model, model_inputs, target_tokens):
    device_inputs = model_inputs.to(model.device)

    start_time = time.time()
    generated = model.generate(
        **device_inputs,
        **generation_kwargs(model, target_tokens),
    )
    spend_time = time.time() - start_time

    sequences = generated.sequences
    input_len = int(device_inputs.input_ids.shape[1])
    generated_len = int(sequences.shape[1] - input_len)

    return spend_time, generated_len


def benchmark_target(model, tokenizer, prompt, target_tokens, runs):
    model_inputs, _ = build_inputs(tokenizer, prompt)

    hello_inputs, _ = build_inputs(tokenizer, "hello")
    _ = run_generate(model, hello_inputs, target_tokens)
    warmup_time, warmup_generated = run_generate(
        model, model_inputs, target_tokens
    )

    print(f"\n=== TARGET {target_tokens} NEW TOKENS ===")
    print(
        f"benchmark-prompt warmup: {warmup_time:.6f}s, "
        f"generated={warmup_generated}"
    )

    if warmup_generated != target_tokens:
        raise RuntimeError(
            f"Warmup generated {warmup_generated} tokens, "
            f"expected exactly {target_tokens}. "
            "The model likely stopped early; do not use this timing."
        )

    times = []
    generated_lengths = []

    for _ in tqdm(range(runs), desc=f"{target_tokens} tokens"):
        spend_time, generated_len = run_generate(
            model, model_inputs, target_tokens
        )
        times.append(spend_time)
        generated_lengths.append(generated_len)

    bad = [x for x in generated_lengths if x != target_tokens]
    if bad:
        raise RuntimeError(
            f"{len(bad)}/{runs} measured runs did not generate exactly "
            f"{target_tokens} tokens. Do not report these timings."
        )

    arr = np.asarray(times, dtype=np.float64)
    result = {
        "target_tokens": target_tokens,
        "runs": runs,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

    print(
        f"mean={result['mean']:.9f}s  "
        f"std={result['std']:.9f}s  "
        f"min={result['min']:.9f}s  "
        f"max={result['max']:.9f}s"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify environment and the actual input token count.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Measured runs per checkpoint. Paper uses 100.",
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=list(TARGET_NEW_TOKENS),
        choices=list(TARGET_NEW_TOKENS),
        help="Generation checkpoints to benchmark.",
    )
    args = parser.parse_args()

    print_environment()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    prompt, input_tokens = check_input(tokenizer)

    if args.check_only:
        print("\nCHECK ONLY: model was not loaded; no latency run performed.")
        return

    if input_tokens != EXPECTED_INPUT_TOKENS:
        raise RuntimeError(
            f"Chat-templated input is {input_tokens} tokens, "
            f"not the paper target of {EXPECTED_INPUT_TOKENS}. "
            "Stop here and decide the input preparation rule before timing."
        )

    print("\n=== MODEL LOAD ===")
    print("model        :", MODEL_NAME)
    print('torch_dtype  : "auto"')
    print('device_map   : "auto"')

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    print("model dtype  :", model.dtype)
    print("model device :", model.device)

    results = []
    for target in args.targets:
        results.append(
            benchmark_target(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_tokens=target,
                runs=args.runs,
            )
        )

    paper = {
        1: 0.13281182289123536,
        512: 11.396940021514892,
        1024: 22.677662715911865,
    }

    print("\n=== TABLE 2 BASE SUMMARY ===")
    print("target\tours_mean_s\tours_std_s\tpaper_H20_s")
    for r in results:
        target = r["target_tokens"]
        print(
            f"{target}\t{r['mean']:.9f}\t"
            f"{r['std']:.9f}\t{paper[target]:.9f}"
        )

    print(
        "\nNOTE: paper uses NVIDIA H20; this server uses the current "
        "reproduction environment GPU, so absolute latency is not expected "
        "to match H20 exactly."
    )


if __name__ == "__main__":
    main()
