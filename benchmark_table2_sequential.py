#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce Table 2 / Table 9 second row: Qwen3-8B + PlugGuard (Sequential).

This is a compatibility adaptation of the repository's released latency path:
- utils/demo_qwen3_with_guardrail.py attaches StreamingSafetyHead and passes
  with_safety_head to generation.
- utils/modeling_qwen3.py inserts the safety head into Qwen3ForCausalLM.forward,
  selects hidden-state layer 20, initializes from the prefix, and executes a
  safety step inline.
- The released utils/modeling_qwen3.py uses an older StreamingSafetyHead API
  (mode/init_mode arguments and tensor return assumptions) that does not match
  the current models.py. This script keeps the same inline sequential design
  but adapts only those calls to the current StreamingSafetyHead API.

No PlugGuard checkpoint is loaded, matching the released latency demo. For
latency, parameter values do not change the executed tensor shapes/operations.
"""

import argparse
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import Qwen3ForCausalLM

from benchmark_table2_base import (
    MODEL_NAME,
    EXPECTED_INPUT_TOKENS,
    TARGET_NEW_TOKENS,
    build_inputs,
    check_input,
    print_environment,
)
from models import StreamingSafetyHead


IDX_LAYER = 20
ASSIST_LEN_FOR_DT = 2048


class Qwen3ForCausalLMSequentialPlugGuard(Qwen3ForCausalLM):
    """
    Minimal compatibility wrapper around the current Transformers Qwen3 class.

    The safety computation is executed synchronously inside forward(), so the
    next decoding step cannot begin until PlugGuard finishes: Sequential mode.
    """

    def reset_safety_runtime(self):
        self._plugguard_initialized = False
        self._plugguard_step = 0
        if hasattr(self, "safety_head"):
            self.safety_head.reset_state()

    def _run_plugguard_inline(self, hidden_states):
        if hidden_states is None:
            raise RuntimeError(
                "PlugGuard requires hidden states, but outputs.hidden_states is None."
            )

        if IDX_LAYER >= len(hidden_states):
            raise RuntimeError(
                f"idx_layer={IDX_LAYER} but only {len(hidden_states)} hidden-state "
                "entries were returned."
            )

        # Same layer choice as the released modified modeling_qwen3.py and config.py.
        layer_hidden = hidden_states[IDX_LAYER]  # (B, seq, hidden_dim)

        # Current models.py AttentionLayer returns (context_vector, weights).
        feat = self.safety_head.attention(layer_hidden)[0]  # (B, seq, proj_dim)

        if not self._plugguard_initialized:
            # First model call is the 1000-token prefill.
            # Initialize PlugGuard from the projected prefix, following the
            # released inline implementation's init_with_prefix(2048, ...).
            self.safety_head.init_with_prefix(ASSIST_LEN_FOR_DT, feat)

            # The released modified modeling_qwen3.py performs a safety step
            # unconditionally after prefix initialization. With the current
            # API, step() expects one token representation, so use the last
            # prefix representation. This preserves one safety-step cost for
            # the first generated token and yields one safety step per output
            # token across generation.
            _ = self.safety_head.step(feat[:, -1, :], self._plugguard_step)
            self._plugguard_step += 1
            self._plugguard_initialized = True
            return

        # Decode calls normally contain one token when KV cache is active.
        # Use the newest representation only.
        _ = self.safety_head.step(feat[:, -1, :], self._plugguard_step)
        self._plugguard_step += 1

    def forward(self, *args, with_safety_head=None, **kwargs):
        # Let the stock Transformers 4.56.2 Qwen3 forward do all LM work.
        outputs = super().forward(*args, **kwargs)

        if with_safety_head:
            self._run_plugguard_inline(outputs.hidden_states)

        return outputs


def generation_kwargs(max_new_tokens):
    # Kept identical to benchmark_table2_base.py / released demo, except
    # PlugGuard is enabled here.
    return dict(
        max_new_tokens=max_new_tokens,
        top_p=1.0,
        top_k=0,
        do_sample=False,
        repetition_penalty=1.0,
        output_hidden_states=True,
        return_dict_in_generate=True,
        with_safety_head=True,
    )


def run_generate(model, model_inputs, target_tokens):
    device_inputs = model_inputs.to(model.device)
    model.reset_safety_runtime()

    start_time = time.time()
    generated = model.generate(
        **device_inputs,
        **generation_kwargs(target_tokens),
    )
    spend_time = time.time() - start_time

    input_len = int(device_inputs.input_ids.shape[1])
    generated_len = int(generated.sequences.shape[1] - input_len)

    # Sanity check: the wrapper should execute one inline PlugGuard step per
    # generated token under the current greedy generation path.
    safety_steps = int(model._plugguard_step)

    return spend_time, generated_len, safety_steps


def benchmark_target(model, tokenizer, prompt, target_tokens, runs):
    model_inputs, _ = build_inputs(tokenizer, prompt)
    hello_inputs, _ = build_inputs(tokenizer, "hello")

    # Follow the same warmup pattern as the released utility and Base script.
    _ = run_generate(model, hello_inputs, target_tokens)
    warmup_time, warmup_generated, warmup_steps = run_generate(
        model, model_inputs, target_tokens
    )

    print(f"\n=== TARGET {target_tokens} NEW TOKENS / SEQUENTIAL ===")
    print(
        f"benchmark-prompt warmup: {warmup_time:.6f}s, "
        f"generated={warmup_generated}, safety_steps={warmup_steps}"
    )

    if warmup_generated != target_tokens:
        raise RuntimeError(
            f"Warmup generated {warmup_generated}, expected {target_tokens}."
        )
    if warmup_steps != target_tokens:
        raise RuntimeError(
            f"Warmup executed {warmup_steps} safety steps, "
            f"expected {target_tokens}."
        )

    times = []
    for _ in tqdm(range(runs), desc=f"seq {target_tokens} tokens"):
        spend_time, generated_len, safety_steps = run_generate(
            model, model_inputs, target_tokens
        )
        if generated_len != target_tokens:
            raise RuntimeError(
                f"Generated {generated_len}, expected {target_tokens}."
            )
        if safety_steps != target_tokens:
            raise RuntimeError(
                f"Executed {safety_steps} safety steps, expected {target_tokens}."
            )
        times.append(spend_time)

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
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=list(TARGET_NEW_TOKENS),
        choices=list(TARGET_NEW_TOKENS),
    )
    args = parser.parse_args()

    print_environment()

    # Use exactly the same local model snapshot / tokenizer protocol as the
    # already validated Base benchmark.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    prompt, input_tokens = check_input(tokenizer)
    if input_tokens != EXPECTED_INPUT_TOKENS:
        raise RuntimeError(
            f"Input is {input_tokens} tokens, expected {EXPECTED_INPUT_TOKENS}."
        )

    print("\n=== MODEL LOAD: SEQUENTIAL PLUGGUARD ===")
    print("model        :", MODEL_NAME)
    print("idx_layer    :", IDX_LAYER)
    print('torch_dtype  : "auto"')
    print('device_map   : "auto"')

    model = Qwen3ForCausalLMSequentialPlugGuard.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )

    input_dim = model.lm_head.in_features
    model.safety_head = StreamingSafetyHead(
        input_dim=input_dim,
        proj_dim=1024,
        mem_dim=1024,
        num_labels=2,
    )
    model.safety_head.to(device=model.device, dtype=model.dtype)
    model.eval()
    model.reset_safety_runtime()

    head_params = sum(p.numel() for p in model.safety_head.parameters())

    print("model dtype  :", model.dtype)
    print("model device :", model.device)
    print("PlugGuard params:", head_params, f"({head_params / 1e6:.3f}M)")

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
        1: 0.13414551019668580,
        512: 11.704826402664185,
        1024: 23.104892206192016,
    }

    print("\n=== TABLE 2 SEQUENTIAL SUMMARY ===")
    print("target\tours_mean_s\tours_std_s\tpaper_H20_s")
    for r in results:
        target = r["target_tokens"]
        print(
            f"{target}\t{r['mean']:.9f}\t"
            f"{r['std']:.9f}\t{paper[target]:.9f}"
        )


if __name__ == "__main__":
    main()
