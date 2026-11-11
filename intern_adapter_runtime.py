#!/usr/bin/env python3
"""Isolated InternLM3 extension for the existing cross-adapter A--F framework.

No modifications to cross_adapter_alignment.py or cross_adapter_training.py are made.
InternLM3 uses Layer 24 and a custom, potentially slow SentencePiece tokenizer.
The fallback offset implementation verifies *exact* token IDs against the real
HuggingFace tokenizer and aborts rather than silently using a wrong alignment.
"""
from __future__ import annotations

import bisect
import gc
import os
import re
import shutil
from pathlib import Path

import torch

import cross_adapter_alignment as A
import cross_adapter_training as T

ROOT = A.ROOT
OUT_ROOT = ROOT / "checkpoints/cross_adapt_intern_six_directions_v1"
MAP_ROOT = ROOT / "data/cross_adapt_intern_char_maps_v1"

DIRECTIONS = [
    ("qwen8", "intern"),
    ("qwen14", "intern"),
    ("llama", "intern"),
    ("intern", "qwen8"),
    ("intern", "qwen14"),
    ("intern", "llama"),
]


class SlowSentencePieceOffsetTokenizer:
    """Offset-map facade for InternLM3's verified slow SentencePiece tokenizer.

    Reconstructs offsets by splitting the rendered chat message on registered
    added tokens, using the same SentencePiece model as InternLM3Tokenizer to
    encode each non-special segment. Verifies resulting IDs against the full
    tokenizer call for EACH example. Emits zero offsets for specials and
    non-character-aligned byte fragments, excluding these from EXACT pairs.

    It does NOT approximate token IDs, choose a nearest token based on decoded
    strings, or infer special-token indices without full-sequence verification.
    """

    def __init__(self, tokenizer):
        self.inner = tokenizer
        if not hasattr(tokenizer, "sp_model"):
            raise RuntimeError(
                f"Slow tokenizer {type(tokenizer).__name__} lacks sp_model. "
                "Do not build maps until an exact offset extractor is supplied."
            )
        extras = {str(tok) for tok in tokenizer.added_tokens_decoder.values() if str(tok)}
        # Slow HF tokenizers split registered AddedToken instances before _tokenize.
        if not extras:
            raise RuntimeError("InternLM3 slow tokenizer has no registered added tokens")
        self._added = extras
        self._re = re.compile("(" + "|".join(re.escape(x) for x in sorted(extras, key=len, reverse=True)) + ")")
        self.is_fast = True  # offset-map API is explicitly implemented here

    def apply_chat_template(self, *args, **kwargs):
        return self.inner.apply_chat_template(*args, **kwargs)

    @staticmethod
    def _byte_to_char_table(text):
        chars = [0]
        for c in text:
            chars.append(chars[-1] + len(c.encode("utf-8")))
        return chars

    def _offset_ids(self, rendered):
        raw_ids = []
        offsets = []
        cursor = 0
        for match in self._re.finditer(rendered):
            if match.start() > cursor:
                s_ids, s_offs = self._segment(rendered[cursor:match.start()], cursor)
                raw_ids.extend(s_ids); offsets.extend(s_offs)
            token = match.group(0)
            raw_ids.append(int(self.inner.convert_tokens_to_ids(token)))
            offsets.append((0, 0))  # never train on special or template token itself
            cursor = match.end()
        if cursor < len(rendered):
            s_ids, s_offs = self._segment(rendered[cursor:], cursor)
            raw_ids.extend(s_ids); offsets.extend(s_offs)
        ids = self.inner.build_inputs_with_special_tokens(raw_ids)
        if len(ids) < len(raw_ids):
            raise RuntimeError("Unexpected tokenizer special-token assembly")
        # InternLM3Tokenizer prepends BOS when add_bos_token=True and optional EOS.
        prefix_len = int(bool(getattr(self.inner, "add_bos_token", False)))
        suffix_len = int(bool(getattr(self.inner, "add_eos_token", False)))
        if len(ids) != prefix_len + len(raw_ids) + suffix_len:
            raise RuntimeError("Unexpected InternLM3 special token placement")
        return ids, [(0, 0)] * prefix_len + offsets + [(0, 0)] * suffix_len

    def _segment(self, segment, global_start):
        # SentencePiece proto offsets use UTF-8 BYTE positions.  Recent
        # SentencePiece versions require return_type="proto"; older versions
        # accept only out_type="immutable_proto".  Both expose .pieces with
        # id/begin/end, and the full ID sequence is checked below.
        try:
            proto = self.inner.sp_model.encode(segment, return_type="proto")
        except TypeError as exc:
            if "return_type" not in str(exc):
                raise
            proto = self.inner.sp_model.encode(segment, out_type="immutable_proto")
        byte_ends = self._byte_to_char_table(segment)
        ids, offs = [], []
        for p in proto.pieces:
            ids.append(int(p.id))
            a, b = int(p.begin), int(p.end)
            ia = bisect.bisect_left(byte_ends, a)
            ib = bisect.bisect_left(byte_ends, b)
            if (0 <= ia < len(byte_ends) and 0 <= ib < len(byte_ends)
                    and byte_ends[ia] == a and byte_ends[ib] == b and ib > ia):
                offs.append((global_start + ia, global_start + ib))
            else:
                offs.append((0, 0))  # byte-fallback fragment; never claim a char boundary
        return ids, offs

    def __call__(self, rendered_list, **kwargs):
        if not kwargs.pop("return_offsets_mapping", False):
            return self.inner(rendered_list, **kwargs)
        if not isinstance(rendered_list, list) or len(rendered_list) != 1:
            raise RuntimeError("InternLM3 offset wrapper accepts one rendered message at a time")
        if kwargs.get("add_special_tokens", True) is not True:
            raise RuntimeError("InternLM3 cache uses add_special_tokens=True")
        rendered = rendered_list[0]
        ids, offs = self._offset_ids(rendered)
        actual = self.inner(rendered_list, **kwargs)["input_ids"][0]
        if ids != actual:
            mismatch = next((i for i, (x, y) in enumerate(zip(ids, actual)) if x != y),
                            min(len(ids), len(actual)))
            raise RuntimeError(
                f"InternLM3 slow-tokenizer offsets cannot reproduce original token IDs: "
                f"first mismatch={mismatch}, reconstructed={len(ids)}, actual={len(actual)}. "
                "Stop: do not train with unverified alignment."
            )
        return {"input_ids": [ids], "offset_mapping": [offs]}


def register():
    # Preserve the old protocol for Qwen3/Llama, and extend only the new six directions.
    A.MODELS["intern"] = {
        "data": "internlm3_8_instruct",
        "model": "InternLM3-8B-Instruct",
        "dim": 4096,
        "layer": 24,
        "head": "fig3_intern_extension_v1/source_intern_target_intern/model_epoch_0.pt",
    }
    for name in ("qwen8", "qwen14", "llama"):
        A.MODELS[name]["layer"] = 20

    A.Config.output = property(lambda self: OUT_ROOT / self.name)
    A.Config.maps = property(lambda self: MAP_ROOT / self.name)

    def cache_for_side(self, split, side):
        key = self.source if side == "src" else self.target
        m = A.MODELS[key]
        full = self.source_model if side == "src" else self.target_model
        return (self.data_split(split) / "safety_cache" /
                str(full).replace("/", "-") / f"idx{m['layer']}_maxlength4096")

    A.Config.cache = cache_for_side

    def corrected_tokenizers(cfg):
        from transformers import AutoTokenizer
        ret = []
        for key, path in ((cfg.source, cfg.source_model), (cfg.target, cfg.target_model)):
            tok = AutoTokenizer.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=True, use_fast=True
            )
            if not tok.is_fast:
                if key != "intern":
                    raise RuntimeError(f"Unexpected slow tokenizer for {key}")
                tok = SlowSentencePieceOffsetTokenizer(tok)
                print("InternLM3: using ID-verified slow SentencePiece offset mapping", flush=True)
            ret.append(tok)
        return tuple(ret)

    A.tokenizers = corrected_tokenizers

    def corrected_cache_builder(cfg):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from dataset import SafetyDataset
        device = A.gpu0()
        free_gib = shutil.disk_usage(ROOT).free / (1024**3)
        if free_gib < 30:
            raise RuntimeError(f"Disk space {free_gib:.1f} GiB below 30 GiB safety floor")
        for side, backbone in (("src", cfg.source_model), ("tgt", cfg.target_model)):
            missing = []
            for split in ("src_train", "src_test", "tgt_train", "tgt_test"):
                dest = cfg.cache(split, side)
                have, extra = A.count_samples(dest, cfg.n(split))
                if extra:
                    raise RuntimeError(f"Unexpected files in cache: {dest}")
                if have < cfg.n(split):
                    missing.append((split, dest, have))
            if not missing:
                print(f"CACHE COMPLETE {cfg.name}/{side}: existing cache reused", flush=True)
                continue
            key = cfg.source if side == "src" else cfg.target
            layer = A.MODELS[key]["layer"]
            print(f"LOADING FROZEN {key}, layer={layer}; missing={missing}", flush=True)
            tok = AutoTokenizer.from_pretrained(
                str(backbone), local_files_only=True, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(backbone), local_files_only=True, trust_remote_code=True,
                torch_dtype="auto", device_map={"": 0}
            ).eval().requires_grad_(False)
            try:
                for split, dest, have in missing:
                    print(f"CACHE BUILD {cfg.name}/{side}/{split}: {have}/{cfg.n(split)}", flush=True)
                    ds = SafetyDataset(
                        dataset_dir=str(cfg.data_split(split)), model_name=str(backbone),
                        tokenizer=tok, base_model=model, device=str(model.device),
                        idx_layer=layer, max_length=4096,
                        build_cache_if_missing=True, overwrite=False,
                    )
                    if A.count_samples(dest, cfg.n(split))[0] < cfg.n(split):
                        ds._build_cache_per_sample(
                            tokenizer=tok, base_model=model, overwrite=False,
                            max_build_samples=None,
                        )
                    A.require_full(dest, cfg.n(split))
                    print(f"CACHE READY {cfg.name}/{side}/{split}", flush=True)
                    del ds
            finally:
                del model, tok
                gc.collect()
                torch.cuda.empty_cache()
        del device

    A.build_missing_caches = corrected_cache_builder
    T.build_missing_caches = corrected_cache_builder  # imported by value by trainer


register()
