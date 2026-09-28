#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce Table 2 / Table 9: Qwen3-8B + PlugGuard (Parallel).

Source basis:
- Paper: token-t safety check runs on a separate computational stream while
  the LM processes token t+1; only the final safety check remains exposed.
- Repository: reuse the already validated Base benchmark protocol and the
  current Sequential compatibility implementation.

Important:
The public repository does NOT provide a complete runnable parallel benchmark.
Therefore this file is a reconstruction of the paper-described deployment
using torch.cuda.Stream, while keeping all other benchmark choices aligned
with the existing Base/Sequential scripts.

Protocol:
- Qwen3-8B
- exactly 1000 input tokens
- targets: 1 / 512 / 1024 new tokens
- greedy generation settings inherited from the released demo
- PlugGuard layer 20
- current StreamingSafetyHead API from models.py
- first safety check remains synchronous
- subsequent safety checks are enqueued on a dedicated CUDA stream
- the last pending safety check is synchronized before stopping the timer
"""

import argparse
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from benchmark_table2_base import (
    MODEL_NAME,
    EXPECTED_INPUT_TOKENS,
    TARGET_NEW_TOKENS,
    build_inputs,
    check_input,
    print_environment,
)
from benchmark_table2_sequential import (
    Qwen3ForCausalLMSequentialPlugGuard,
    IDX_LAYER,
    ASSIST_LEN_FOR_DT,
    generation_kwargs,
)
from models import StreamingSafetyHead


class Qwen3ForCausalLMParallelPlugGuard(Qwen3ForCausalLMSequentialPlugGuard):
    """
    Same Qwen3 + PlugGuard path as the Sequential implementation, except:
      - first PlugGuard check executes synchronously;
      - later checks are enqueued on a dedicated CUDA stream;
      - the generation loop can continue on the default stream;
      - the caller synchronizes the safety stream only at the end.
    """

    def _ensure_parallel_stream(self):
        if getattr(self, "_plugguard_stream", None) is None:
            self._plugguard_stream = torch.cuda.Stream(device=self.device)
        if getattr(self, "_plugguard_events", None) is None:
            self._plugguard_events = []

    def synchronize_safety(self):
        stream = getattr(self, "_plugguard_stream", None)
        if stream is not None:
            stream.synchronize()
        self._plugguard_events = []

    def reset_safety_runtime(self):
        # Make sure a previous run has no outstanding safety kernels.
        stream = getattr(self, "_plugguard_stream", None)
        if stream is not None:
            stream.synchronize()

        self._plugguard_initialized = False
        self._plugguard_step = 0
        self._plugguard_async_steps = 0
        self._plugguard_events = []

        if hasattr(self, "safety_head"):
            self.safety_head.reset_state()

    def _validate_hidden_states(self, hidden_states):
        if hidden_states is None:
            raise RuntimeError(
                "PlugGuard requires hidden states, but outputs.hidden_states is None."
            )
        if IDX_LAYER >= len(hidden_states):
            raise RuntimeError(
                f"idx_layer={IDX_LAYER} but only {len(hidden_states)} "
                "hidden-state entries were returned."
            )

    def _run_plugguard_inline(self, hidden_states):
        """
        Overrides the Sequential method called by forward().

        The method name is inherited from the Sequential wrapper, but for
        decode steps after the first one the work is asynchronous.
        """
        self._validate_hidden_states(hidden_states)
        layer_hidden = hidden_states[IDX_LAYER]

        if not self._plugguard_initialized:
            # Table 2 reports the same first-token latency for Sequential and
            # Parallel. Keep the first safety path synchronous.
            feat = self.safety_head.attention(layer_hidden)[0]
            self.safety_head.init_with_prefix(ASSIST_LEN_FOR_DT, feat)
            _ = self.safety_head.step(feat[:, -1, :], self._plugguard_step)

            self._plugguard_step += 1
            self._plugguard_initialized = True
            return

        self._ensure_parallel_stream()

        # The LM hidden state was produced on the current/default stream.
        # Record readiness and make the safety stream wait only for that data.
        ready = torch.cuda.Event(blocking=False, interprocess=False)
        ready.record(torch.cuda.current_stream(device=layer_hidden.device))
        self._plugguard_events.append(ready)

        # Tell the caching allocator that this tensor is also consumed on the
        # safety stream. generate() currently retains hidden states, but this
        # makes the cross-stream lifetime explicit.
        layer_hidden.record_stream(self._plugguard_stream)

        with torch.cuda.stream(self._plugguard_stream):
            self._plugguard_stream.wait_event(ready)

            feat = self.safety_head.attention(layer_hidden)[0]
            _ = self.safety_head.step(feat[:, -1, :], self._plugguard_step)

        # Count enqueued safety checks. Completion is guaranteed by
        # synchronize_safety() before timing stops.
        self._plugguard_step += 1
        self._plugguard_async_steps += 1


def run_generate(model, model_inputs, target_tokens):
    device_inputs = model_inputs.to(model.device)
    model.reset_safety_runtime()

    start_time = time.time()

    generated = model.generate(
        **device_inputs,
        **generation_kwargs(target_tokens),
    )

    # Paper: final-token safety check is the remaining non-parallelizable
    # user-facing overhead. Wait for the dedicated safety stream here.
    model.synchronize_safety()

    spend_time = time.time() - start_time

    input_len = int(device_inputs.input_ids.shape[1])
    generated_len = int(generated.sequences.shape[1] - input_len)

    safety_steps = int(model._plugguard_step)
    async_steps = int(model._plugguard_async_steps)

    return spend_time, generated_len, safety_steps, async_steps


def benchmark_target(model, tokenizer, prompt, target_tokens, runs):
    model_inputs, _ = build_inputs(tokenizer, prompt)
    hello_inputs, _ = build_inputs(tokenizer, "hello")

    # Same warmup structure as Base and Sequential.
    _ = run_generate(model, hello_inputs, target_tokens)
    (
        warmup_time,
        warmup_generated,
        warmup_steps,
        warmup_async_steps,
    ) = run_generate(model, model_inputs, target_tokens)

    print(f"\n=== TARGET {target_tokens} NEW TOKENS / PARALLEL ===")
    print(
        f"benchmark-prompt warmup: {warmup_time:.6f}s, "
        f"generated={warmup_generated}, "
        f"safety_steps={warmup_steps}, "
        f"async_steps={warmup_async_steps}"
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

    expected_async = max(0, target_tokens - 1)
    if warmup_async_steps != expected_async:
        raise RuntimeError(
            f"Warmup enqueued {warmup_async_steps} async checks, "
            f"expected {expected_async}."
        )

    times = []

    for _ in tqdm(range(runs), desc=f"par {target_tokens} tokens"):
        (
            spend_time,
            generated_len,
            safety_steps,
            async_steps,
        ) = run_generate(model, model_inputs, target_tokens)

        if generated_len != target_tokens:
            raise RuntimeError(
                f"Generated {generated_len}, expected {target_tokens}."
            )
        if safety_steps != target_tokens:
            raise RuntimeError(
                f"Executed {safety_steps} safety steps, expected {target_tokens}."
            )
        if async_steps != expected_async:
            raise RuntimeError(
                f"Enqueued {async_steps} async checks, expected {expected_async}."
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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    prompt, input_tokens = check_input(tokenizer)

    if input_tokens != EXPECTED_INPUT_TOKENS:
        raise RuntimeError(
            f"Input is {input_tokens} tokens, expected {EXPECTED_INPUT_TOKENS}."
        )

    print("\n=== MODEL LOAD: PARALLEL PLUGGUARD ===")
    print("model        :", MODEL_NAME)
    print("idx_layer    :", IDX_LAYER)
    print('torch_dtype  : "auto"')
    print('device_map   : "auto"')
    print("parallel impl: torch.cuda.Stream")

    model = Qwen3ForCausalLMParallelPlugGuard.from_pretrained(
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

    model.safety_head.to(
        device=model.device,
        dtype=model.dtype,
    )

    model.eval()
    model._plugguard_stream = torch.cuda.Stream(device=model.device)
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
        1: 0.1341,
        512: 11.3973,
        1024: 22.6780,
    }

    print("\n=== TABLE 2 PARALLEL SUMMARY ===")
    print("target\tours_mean_s\tours_std_s\tpaper_H20_s")

    for r in results:
        target = r["target_tokens"]
        print(
            f"{target}\t{r['mean']:.9f}\t"
            f"{r['std']:.9f}\t{paper[target]:.4f}"
        )

    print(
        "\nNOTE: the paper specifies a separate computational stream but "
        "the public repository does not release the exact parallel code. "
        "This benchmark reconstructs that design with torch.cuda.Stream."
    )


if __name__ == "__main__":
    main()
