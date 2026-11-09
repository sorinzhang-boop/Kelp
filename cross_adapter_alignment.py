#!/usr/bin/env python3
"""Shared runner for four *directed* frozen-PlugGuard adapter transfers.

Derived from affine_llama_to_qwen8_bcef.py with source/target generalized.
Training: EXACT role-relative character-end content-token pairs, normalized MSE.
Evaluation: target tokenizer/assistant_start grid, source latest completed
content-token state for each target content position, prefix-state filling for
headers/special tokens. This is *offline* Response/Streaming-F1 evaluation.

B: adapter on source-generated TRAIN, source-generated TEST.
C: target LLM's actual cached states on source-generated TEST, frozen target head.
E: adapter on target-generated TRAIN, target-generated TEST.
F: SAME E adapter, source-generated TEST.

All model backbones and the target PlugGuard safety head are frozen. Only
adapter.linear is optimized. We never regenerate response text.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gc
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report

ROOT = Path.home() / "plugguard_repro"
DATA_ROOT = ROOT / "data/StreamGuardBench_v1/s_eval"
CACHE_SUFFIX = "idx20_maxlength4096"
SEED = 42

MODELS = {
    "qwen8": {"data": "qwen3_8b", "model": "Qwen3-8B", "dim": 4096,
              "head": "qwen3_8b_seval/model_epoch_0.pt"},
    "qwen14": {"data": "qwen3_14b", "model": "Qwen3-14B", "dim": 5120,
               "head": "qwen3_14b_seval/model_epoch_0.pt"},
    "llama": {"data": "llama_3_1_8b_instruct", "model": "Meta-Llama-3.1-8B-Instruct", "dim": 4096,
              "head": "llama_3_1_8b_seval/model_epoch_0.pt"},
}

@dataclass(frozen=True)
class Config:
    source: str
    target: str

    @property
    def name(self):
        return f"{self.source}_to_{self.target}"

    @property
    def source_model(self):
        return ROOT / "models" / MODELS[self.source]["model"]

    @property
    def target_model(self):
        return ROOT / "models" / MODELS[self.target]["model"]

    @property
    def source_dim(self):
        return MODELS[self.source]["dim"]

    @property
    def target_dim(self):
        return MODELS[self.target]["dim"]

    @property
    def head_ckpt(self):
        return ROOT / "checkpoints" / MODELS[self.target]["head"]

    @property
    def output(self):
        return ROOT / "checkpoints/cross_adapt_four_directions_v1" / self.name

    @property
    def maps(self):
        return ROOT / "data/cross_adapt_char_maps_v1" / self.name

    def data_split(self, split):
        k, typ = split.split("_")
        model = self.source if k == "src" else self.target
        return DATA_ROOT / MODELS[model]["data"] / typ

    def n(self, split):
        return 9000 if split.endswith("train") else 1000

    def cache(self, split, model):
        full = self.source_model if model == "src" else self.target_model
        return self.data_split(split) / "safety_cache" / str(full).replace("/", "-") / CACHE_SUFFIX

    def map_dir(self, split):
        return self.maps / split

    def adapter_dir(self, split):
        if split not in ("src_train", "tgt_train"):
            raise ValueError(split)
        return self.output / f"train_on_{split}"


def path_sample(folder, idx):
    return folder / f"sample_{idx:08d}.pt"


def count_samples(folder, n):
    existing = {p.name for p in folder.glob("sample_*.pt")}
    expected = {f"sample_{i:08d}.pt" for i in range(n)}
    return len(existing & expected), len(existing - expected)


def require_full(folder, n):
    have, extra = count_samples(folder, n)
    if have != n or extra:
        raise RuntimeError(f"Incomplete cache/map {folder}: {have}/{n}, unexpected={extra}")


def read_h(folder, idx, width):
    p = path_sample(folder, idx)
    obj = torch.load(p, map_location="cpu", weights_only=True)
    h = obj["embeddings"]
    a = int(obj["assistant_start"])
    y = torch.as_tensor(obj["labels"], dtype=torch.long)
    if h.ndim != 2 or h.shape[1] != width or not (0 < a < len(h)):
        raise RuntimeError(f"Invalid hidden state/assistant_start at {p}: {tuple(h.shape)}, {a}")
    if len(y) != len(h) - a:
        raise RuntimeError(f"Labels length mismatch at {p}: {len(y)} vs {len(h)-a}")
    if not torch.isfinite(h).all():
        raise RuntimeError(f"Nonfinite hidden states in {p}")
    return h, a, y


def read_map(cfg, split, idx):
    p = path_sample(cfg.map_dir(split), idx)
    m = torch.load(p, map_location="cpu", weights_only=True)
    if m.get("protocol") != "relative_char_end_v1" or m.get("direction") != cfg.name:
        raise RuntimeError(f"Stale or wrong-direction mapping: {p}")
    return m


def gpu0():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1 (physical GPU 1 only)")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible NVIDIA GPU (physical GPU 1)")
    torch.cuda.set_device(0)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    return torch.device("cuda:0")


def seed_all():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def check_input(cfg):
    if cfg.source == cfg.target:
        raise RuntimeError("Source and target must differ")
    for typ, full in (("SOURCE", cfg.source_model), ("TARGET", cfg.target_model)):
        if not full.is_dir():
            raise FileNotFoundError(f"{typ} backbone path missing: {full}")
    for split in ("src_train", "src_test", "tgt_train", "tgt_test"):
        path = cfg.data_split(split)
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset path missing: {path}")
        for side in ("src", "tgt"):
            have, extra = count_samples(cfg.cache(split, side), cfg.n(split))
            print(f"CACHE {cfg.name}/{split}/{side}: {have}/{cfg.n(split)}, extra={extra}", flush=True)
            if extra:
                raise RuntimeError("Unexpected cache filename in " + str(cfg.cache(split, side)))
    if not cfg.head_ckpt.is_file():
        raise FileNotFoundError(f"Frozen target head missing: {cfg.head_ckpt}; verify name before overnight run")
    checkpoint = torch.load(cfg.head_ckpt, map_location="cpu", weights_only=True)
    key = "attention.query.weight"
    if key not in checkpoint or int(checkpoint[key].shape[1]) != cfg.target_dim:
        raise RuntimeError(f"Frozen head input dimension mismatch: {cfg.head_ckpt}, "
                           f"expected={cfg.target_dim}, key={key}")
    del checkpoint
    free = shutil.disk_usage(ROOT).free / (1024 ** 3)
    print(f"Target frozen head: {cfg.head_ckpt}; free disk: {free:.1f} GiB", flush=True)
    if free < 15:
        raise RuntimeError("Less than 15 GiB free. Check disk space before caching")


def audit_data_and_templates(cfg):
    """Cheap early failure check before loading any large frozen backbone."""
    from datasets import load_from_disk
    st, tt = tokenizers(cfg)
    norm = lambda value: " ".join(str(value).split())
    prompts = {}
    for split in ("src_train", "src_test", "tgt_train", "tgt_test"):
        ds = load_from_disk(str(cfg.data_split(split)))
        if len(ds) != cfg.n(split):
            raise RuntimeError(f"Dataset size differs from expected: {cfg.data_split(split)} "
                               f"{len(ds)}/{cfg.n(split)}")
        prompts[split] = {norm(row["prompt"]) for row in ds}
        for idx in sorted({0, 1, len(ds)-1}):
            row = ds[idx]
            a = as_cached_tokens(st, row["prompt"], row["response"])
            b = as_cached_tokens(tt, row["prompt"], row["response"])
            if len(a[0]) < 3 or len(b[0]) < 3:
                raise RuntimeError(f"Suspicious short token sequence {cfg.name}/{split}/{idx}")
            src_end = {x for x, _ in content_positions(a[1], a[2], len(a[0])-1, "prompt", len(a[0]))}
            tgt_end = {x for x, _ in content_positions(b[1], b[2], len(b[0])-1, "prompt", len(b[0]))}
            if not (src_end & tgt_end):
                raise RuntimeError(f"No common prompt character ends: {cfg.name}/{split}/{idx}")
            for side, obj in (("src", a), ("tgt", b)):
                file = path_sample(cfg.cache(split, side), idx)
                if file.is_file():
                    cached = torch.load(file, map_location="cpu", weights_only=True)
                    if len(obj[0]) != len(cached["embeddings"]) + 1:
                        raise RuntimeError(f"Tokenizer/cached length mismatch at {file}")
        print(f"AUDIT DATA {cfg.name}/{split}: {len(ds)} samples; template spot check OK", flush=True)
    for tr in ("src_train", "tgt_train"):
        for te in ("src_test", "tgt_test"):
            overlap = len(prompts[tr] & prompts[te])
            print(f"PROMPT OVERLAP {cfg.name}: {tr} vs {te}: {overlap}", flush=True)
            if overlap:
                print("WARNING: train/test prompt overlap; disclose or use prompt-disjoint splits", flush=True)


def build_missing_caches(cfg):
    """Loads one frozen backbone at a time, only if any cache files are missing."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from dataset import SafetyDataset
    device = gpu0()
    for side, backbone in (("src", cfg.source_model), ("tgt", cfg.target_model)):
        missing = []
        for split in ("src_train", "src_test", "tgt_train", "tgt_test"):
            cache = cfg.cache(split, side)
            have, extra = count_samples(cache, cfg.n(split))
            if extra:
                raise RuntimeError(f"Unexpected files in {cache}")
            if have < cfg.n(split):
                missing.append((split, cache, have))
        if not missing:
            print(f"CACHE COMPLETE {cfg.name}: {side} (no backbone loading)", flush=True)
            continue
        print(f"Loading frozen {side} backbone {backbone} for {missing}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(backbone), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(backbone), local_files_only=True, torch_dtype="auto", device_map={"": 0}
        )
        model.eval().requires_grad_(False)
        try:
            for split, dest, before in missing:
                print(f"CACHE BUILD {cfg.name}/{side}/{split}: had={before}/{cfg.n(split)}", flush=True)
                ds = SafetyDataset(
                    dataset_dir=str(cfg.data_split(split)), model_name=str(backbone),
                    tokenizer=tokenizer, base_model=model, device=str(model.device),
                    idx_layer=20, max_length=4096, build_cache_if_missing=True, overwrite=False
                )
                have, _ = count_samples(dest, cfg.n(split))
                if have < cfg.n(split):
                    # SafetyDataset auto-builds only when cache is initially EMPTY.
                    ds._build_cache_per_sample(tokenizer=tokenizer, base_model=model,
                                               overwrite=False, max_build_samples=None)
                require_full(dest, cfg.n(split))
                print(f"CACHE READY {cfg.name}/{side}/{split}", flush=True)
        finally:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    del device


def tokenizers(cfg):
    from transformers import AutoTokenizer
    src = AutoTokenizer.from_pretrained(str(cfg.source_model), local_files_only=True, use_fast=True)
    tgt = AutoTokenizer.from_pretrained(str(cfg.target_model), local_files_only=True, use_fast=True)
    if not src.is_fast or not tgt.is_fast:
        raise RuntimeError("Both tokenizers must be fast (offset_mapping required)")
    return src, tgt


def as_cached_tokens(tok, prompt, response):
    """Reconstruct cached token IDs and raw-text-relative token offsets.

    Llama 3.1's chat template can trim leading/trailing whitespace from
    message content.  The hidden-state cache uses that rendered template;
    the *alignment* must nevertheless compare positions in the original
    prompt/response strings, including the whitespace removed by the template.
    This implementation only accepts verbatim content or the exact result of
    Python/Jinja-like strip(); any other template transformation raises.
    """
    if not isinstance(prompt, str) or not prompt or not isinstance(response, str) or not response:
        raise ValueError("Empty or invalid prompt/response")
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=True, max_length=4096, truncation=True
    )
    enc = tok([rendered], return_offsets_mapping=True, add_special_tokens=True)
    ids, offsets = enc["input_ids"][0], enc["offset_mapping"][0]
    if len(ids) != len(offsets):
        raise RuntimeError("Missing tokenizer offset mapping")

    def locate(raw, start, role):
        pos = rendered.find(raw, start)
        if pos >= 0:
            return pos, raw, 0
        visible = raw.strip()
        if not visible:
            raise RuntimeError(f"{role}: content disappears after template whitespace trim")
        pos = rendered.find(visible, start)
        if pos < 0:
            raise RuntimeError(
                f"{role}: template changed content beyond leading/trailing whitespace "
                f"(raw_length={len(raw)}, stripped_length={len(visible)})"
            )
        left_trim = len(raw) - len(raw.lstrip())
        return pos, visible, left_trim

    p, pv, pleft = locate(prompt, 0, "prompt")
    r, rv, rleft = locate(response, p + len(pv), "response")
    if pv == prompt and rv == response:
        return ids, offsets, (p, p + len(prompt)), (r, r + len(response))

    # Virtualize the two stripped regions, so their token offsets use the
    # SAME original-string character coordinate system as Qwen.  In particular
    # a token ending at original prompt character 20 remains at 20 even if
    # Llama's renderer removed two leading spaces.  Do not treat surrounding
    # header/end-of-turn tokens as user content.
    p_end = p + len(pv)
    r_end = r + len(rv)
    dp = len(prompt) - len(pv)
    dr = len(response) - len(rv)
    virtual_offsets = []
    for a, b in offsets:
        if b <= a:  # e.g. BOS / special-token zero offsets
            virtual_offsets.append((a, b))
        elif p <= a and b <= p_end:
            virtual_offsets.append((a + pleft, b + pleft))
        elif r <= a and b <= r_end:
            virtual_offsets.append((a + dp + rleft, b + dp + rleft))
        elif b <= p:
            virtual_offsets.append((a, b))
        elif a >= p_end and b <= r:
            virtual_offsets.append((a + dp, b + dp))
        elif a >= r_end:
            virtual_offsets.append((a + dp + dr, b + dp + dr))
        else:
            # Token crosses a content/template boundary: not a reliable
            # content-token alignment candidate; retain its index but mask
            # its offsets so content_positions() ignores it.
            virtual_offsets.append((0, 0))

    return (ids, virtual_offsets,
            (p, p + len(prompt)),
            (r + dp, r + dp + len(response)))

def content_positions(offsets, span, n, phase, assistant_start):
    start, end = span
    out = []
    for j, (a, b) in enumerate(offsets[:n]):
        if b <= a or a < start or b > end:
            continue
        if phase == "prompt" and j >= assistant_start:
            continue
        if phase == "response" and j < assistant_start:
            continue
        out.append((b - start, j))
    return out


def align_positions(src_offsets, tgt_offsets, src_p, src_r, tgt_p, tgt_r,
                    src_len, tgt_len, src_start, tgt_start):
    """Generalized EXACT training pairs + target-grid source indices.

    Same choices as affine_llama_to_qwen8_bcef.py::causal_alignment(), with
    'Llama' -> source and 'Qwen8' -> target. Note that a target's non-content
    token can use a source prefix state. No strict online causality guarantee.
    """
    sp = content_positions(src_offsets, src_p, src_len, "prompt", src_start)
    sr = content_positions(src_offsets, src_r, src_len, "response", src_start)
    if not sp or not sr:
        raise RuntimeError("Source prompt/response content tokens empty")
    sp_ends, sp_ids = zip(*sp)
    sr_ends, sr_ids = zip(*sr)
    tp0, tp1 = tgt_p
    tr0, tr1 = tgt_r
    tprompt_indices = [j for j, (a, b) in enumerate(tgt_offsets[:tgt_len])
                       if b > a and tp0 <= a and b <= tp1]
    if not tprompt_indices:
        raise RuntimeError("Target prompt content tokens empty")
    first_t_prompt = min(tprompt_indices)
    mapped, exact_t, exact_s, phases = [], [], [], []
    prev = 0
    for j in range(tgt_len):
        a, b = tgt_offsets[j]
        kind = None
        if b > a and tp0 <= a and b <= tp1 and j < tgt_start:
            end = b - tp0
            k = bisect.bisect_right(sp_ends, end) - 1
            source = sp_ids[k] if k >= 0 else max(0, sp_ids[0] - 1)
            if k >= 0 and sp_ends[k] == end:
                kind = 0
        elif b > a and tr0 <= a and b <= tr1 and j >= tgt_start:
            end = b - tr0
            k = bisect.bisect_right(sr_ends, end) - 1
            source = sr_ids[k] if k >= 0 else src_start - 1
            if k >= 0 and sr_ends[k] == end:
                kind = 1
        elif j < tgt_start:
            if j < first_t_prompt:
                source = max(0, sp_ids[0] - 1)
            else:
                source = src_start - 1
        else:
            source = prev
        if not 0 <= source < src_len:
            raise RuntimeError(f"Bad source index target={j}: {source}/{src_len}")
        mapped.append(source)
        prev = source
        if kind is not None:
            exact_t.append(j)
            exact_s.append(source)
            phases.append(kind)
    return {
        "src_for_target": torch.tensor(mapped, dtype=torch.long),
        "exact_target": torch.tensor(exact_t, dtype=torch.long),
        "exact_source": torch.tensor(exact_s, dtype=torch.long),
        "exact_phase": torch.tensor(phases, dtype=torch.long),
        "source_length": src_len, "target_length": tgt_len,
        "source_assistant_start": src_start, "target_assistant_start": tgt_start,
        "target_prompt_content_tokens": len(content_positions(tgt_offsets, tgt_p, tgt_len, "prompt", tgt_start)),
        "target_response_content_tokens": len(content_positions(tgt_offsets, tgt_r, tgt_len, "response", tgt_start)),
    }


def prepare_maps(cfg):
    from datasets import load_from_disk
    src_tok, tgt_tok = tokenizers(cfg)
    for split in ("src_train", "src_test", "tgt_train", "tgt_test"):
        n = cfg.n(split)
        s_cache, t_cache = cfg.cache(split, "src"), cfg.cache(split, "tgt")
        require_full(s_cache, n)
        require_full(t_cache, n)
        dest = cfg.map_dir(split)
        dest.mkdir(parents=True, exist_ok=True)
        ds = load_from_disk(str(cfg.data_split(split)))
        if len(ds) != n:
            raise RuntimeError(f"Data length mismatch: {split}: {len(ds)} vs {n}")
        total_exact = total_content = new_count = 0
        for idx in range(n):
            out = path_sample(dest, idx)
            if out.is_file():
                old = read_map(cfg, split, idx)
                if len(old["src_for_target"]) != old["target_length"]:
                    raise RuntimeError("Existing mapping invalid: " + str(out))
                total_exact += len(old["exact_target"])
                total_content += old["target_prompt_content_tokens"] + old["target_response_content_tokens"]
                continue
            row = ds[idx]
            s_obj = torch.load(path_sample(s_cache, idx), map_location="cpu", weights_only=True)
            t_obj = torch.load(path_sample(t_cache, idx), map_location="cpu", weights_only=True)
            s_h, t_h = s_obj["embeddings"], t_obj["embeddings"]
            sa, ta = int(s_obj["assistant_start"]), int(t_obj["assistant_start"])
            if s_h.ndim != 2 or t_h.ndim != 2 or s_h.shape[1] != cfg.source_dim or t_h.shape[1] != cfg.target_dim:
                raise RuntimeError(f"Feature shape mismatch at {cfg.name}/{split}/{idx}")
            src_ids, so, sp, sr = as_cached_tokens(src_tok, row["prompt"], row["response"])
            tgt_ids, to, tp, tr = as_cached_tokens(tgt_tok, row["prompt"], row["response"])
            # SafetyDataset returns first generation-step hidden states for all
            # input tokens except the last token; cache length = tokenized - 1.
            if len(src_ids) != len(s_h) + 1 or len(tgt_ids) != len(t_h) + 1:
                raise RuntimeError(f"Cached text differs from tokenizer at {cfg.name}/{split}/{idx}: "
                                   f"src={len(src_ids)}/{len(s_h)}, tgt={len(tgt_ids)}/{len(t_h)}")
            if not (0 < sa < len(s_h) and 0 < ta < len(t_h)):
                raise RuntimeError(f"Invalid assistant starts at {cfg.name}/{split}/{idx}")
            m = align_positions(so, to, sp, sr, tp, tr, len(s_h), len(t_h), sa, ta)
            if not len(m["exact_target"]):
                raise RuntimeError(f"Zero training matches at {cfg.name}/{split}/{idx}")
            if len(m["src_for_target"]) != len(t_h) or int(m["src_for_target"].max()) >= len(s_h):
                raise RuntimeError("Mapping out of range")
            if int(s_obj["labels"][-1]) != int(t_obj["labels"][-1]) or int(row["label"]) != int(t_obj["labels"][-1]):
                raise RuntimeError(f"Label mismatch at {cfg.name}/{split}/{idx}")
            m["label"] = int(row["label"])
            m["protocol"] = "relative_char_end_v1"
            m["direction"] = cfg.name
            total_exact += len(m["exact_target"])
            total_content += m["target_prompt_content_tokens"] + m["target_response_content_tokens"]
            new_count += 1
            temp = out.with_suffix(".pt.tmp")
            torch.save(m, temp)
            os.replace(temp, out)
            if (idx + 1) % 500 == 0:
                print(f"MAP BUILD {cfg.name}/{split}: {idx+1}/{n}", flush=True)
        require_full(dest, n)
        print(f"MAP READY {cfg.name}/{split}: new={new_count}, "
              f"exact={total_exact:,}/{total_content:,} ({total_exact/max(1,total_content):.2%})", flush=True)


def verify_one(cfg, split, idx):
    source, sa, sl = read_h(cfg.cache(split, "src"), idx, cfg.source_dim)
    target, ta, tl = read_h(cfg.cache(split, "tgt"), idx, cfg.target_dim)
    m = read_map(cfg, split, idx)
    if (len(source) != m["source_length"] or len(target) != m["target_length"]
            or sa != m["source_assistant_start"] or ta != m["target_assistant_start"]):
        raise RuntimeError(f"Map/cache mismatch {cfg.name}/{split}/{idx}")
    x = m["src_for_target"]
    if len(x) != len(target) or int(x.min()) < 0 or int(x.max()) >= len(source):
        raise RuntimeError(f"Invalid source lookup {cfg.name}/{split}/{idx}")
    if int(sl[-1]) != int(tl[-1]) or int(tl[-1]) != m["label"]:
        raise RuntimeError(f"Label mismatch {cfg.name}/{split}/{idx}")
    return source, target, ta, tl, m


def split_indices():
    perm = torch.randperm(9000, generator=torch.Generator().manual_seed(SEED)).tolist()
    return perm[:8100], perm[8100:]


def take_pairs(m, tokens, seed):
    e_src, e_tgt, phase = m["exact_source"], m["exact_target"], m["exact_phase"]
    gen = torch.Generator().manual_seed(seed)
    prompt_ids = torch.where(phase == 0)[0]
    response_ids = torch.where(phase == 1)[0]
    n_prompt = min(len(prompt_ids), max(1, tokens // 4))
    n_response = min(len(response_ids), tokens - n_prompt)
    chosen = torch.cat((
        prompt_ids[torch.randperm(len(prompt_ids), generator=gen)[:n_prompt]],
        response_ids[torch.randperm(len(response_ids), generator=gen)[:n_response]],
    ))
    if not len(chosen):
        raise RuntimeError("No matched token pairs sampled")
    return e_src.index_select(0, chosen), e_tgt.index_select(0, chosen)


