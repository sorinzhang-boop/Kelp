#!/usr/bin/env python3
"""Affine training, frozen target-head evaluation, checkpointing and CLI.

Shared by four direction-specific entry points. All token alignment/cache logic
is in cross_adapter_alignment.py. This is offline stored-response evaluation.
"""
import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report

from cross_adapter_alignment import (
    ROOT, SEED, Config, require_full, read_h, read_map, gpu0, seed_all,
    check_input, audit_data_and_templates, build_missing_caches, prepare_maps,
    verify_one, split_indices, take_pairs,
)

class AffineAdapter(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        self.linear = nn.Linear(src_dim, tgt_dim, bias=True)
        self.register_buffer("x_mean", torch.zeros(src_dim))
        self.register_buffer("x_std", torch.ones(src_dim))
        self.register_buffer("y_mean", torch.zeros(tgt_dim))
        self.register_buffer("y_std", torch.ones(tgt_dim))

    def normalized(self, x):
        return self.linear((x.float() - self.x_mean) / self.x_std)

    def forward(self, x):
        return self.normalized(x) * self.y_std + self.y_mean


def normalizers(adapter, cfg, split, train_ids, samples, tokens):
    sx = torch.zeros(cfg.source_dim, dtype=torch.float64)
    sy = torch.zeros(cfg.target_dim, dtype=torch.float64)
    ssx = torch.zeros(cfg.source_dim, dtype=torch.float64)
    ssy = torch.zeros(cfg.target_dim, dtype=torch.float64)
    count = 0
    for idx in train_ids[:samples]:
        s, t, _, _, m = verify_one(cfg, split, idx)
        ids_s, ids_t = take_pairs(m, tokens, 1000 + idx)
        x = s.index_select(0, ids_s).double()
        y = t.index_select(0, ids_t).double()
        sx += x.sum(0); sy += y.sum(0)
        ssx += x.square().sum(0); ssy += y.square().sum(0)
        count += len(x)
    if count < 32:
        raise RuntimeError("Not enough training content token pairs for normalization")
    xm, ym = sx/count, sy/count
    xs = (ssx/count-xm.square()).clamp_min(1e-8).sqrt()
    ys = (ssy/count-ym.square()).clamp_min(1e-8).sqrt()
    adapter.x_mean.copy_(xm.float()); adapter.y_mean.copy_(ym.float())
    adapter.x_std.copy_(xs.float()); adapter.y_std.copy_(ys.float())
    print(f"NORMALIZERS READY {cfg.name}/{split}: {count:,} token pairs", flush=True)


def val_mse(adapter, cfg, split, ids, device, tokens):
    adapter.eval()
    total, count = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for idx in ids:
            s, t, _, _, m = verify_one(cfg, split, idx)
            ids_s, ids_t = take_pairs(m, tokens, 1000 + idx)
            x = s.index_select(0, ids_s).float().to(device)
            y = t.index_select(0, ids_t).float().to(device)
            prediction = adapter.normalized(x).float()
            target = (y - adapter.y_mean)/adapter.y_std
            total += F.mse_loss(prediction, target, reduction="sum").item()
            count += target.numel()
    return total / count


def atomic_torch(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def train(cfg, split, device, args):
    out = cfg.adapter_dir(split)
    out.mkdir(parents=True, exist_ok=True)
    best_file, last_file, done_file = out / "affine_best.pt", out / "last_epoch.pt", out / "training_complete.json"
    if done_file.is_file():
        if not best_file.is_file():
            raise RuntimeError(f"Training marker exists but no best adapter: {out}")
        record = json.loads(done_file.read_text())
        if record.get("epochs") != args.epochs or record.get("tokens") != args.tokens or record.get("lr") != args.lr:
            raise RuntimeError(f"Completed run differs from requested hyperparameters: {done_file}")
        print(f"TRAIN SKIP already completed {cfg.name}/{split}", flush=True)
        return
    require_full(cfg.map_dir(split), 9000)
    seed_all()
    tr_ids, va_ids = split_indices()
    adapter = AffineAdapter(cfg.source_dim, cfg.target_dim)
    normalizers(adapter, cfg, split, tr_ids, args.norm_samples, args.tokens)
    adapter.to(device)
    opt = torch.optim.AdamW(adapter.linear.parameters(), lr=args.lr, weight_decay=0.0)
    best, epoch0 = float("inf"), 0
    if last_file.is_file():
        last = torch.load(last_file, map_location="cpu", weights_only=True)
        if last.get("config") != {"epochs": args.epochs, "tokens": args.tokens, "lr": args.lr,
                                   "accum": args.accum, "norm_samples": args.norm_samples}:
            raise RuntimeError(f"Training config changed since interruption: {last_file}")
        adapter.load_state_dict(last["adapter"], strict=True)
        opt.load_state_dict(last["optimizer"])
        # optimizer tensors saved on CPU are relocated to CUDA by torch optimizer state load
        epoch0 = int(last["completed_epochs"])
        best = float(last["best_val_mse"])
        print(f"TRAIN RESUME {cfg.name}/{split}: completed epochs={epoch0}; best={best:.6f}", flush=True)
    print(f"TRAIN START {cfg.name}/{split} {cfg.source_dim}->{cfg.target_dim}: "
          f"epochs={args.epochs}, starting epoch={epoch0+1}, 8100/900, tokens={args.tokens}", flush=True)
    for ep in range(epoch0, args.epochs):
        adapter.train()
        order = tr_ids.copy()
        random.Random(SEED + ep).shuffle(order)
        opt.zero_grad(set_to_none=True)
        loss_total, pair_total = 0.0, 0
        for step, idx in enumerate(order):
            s, t, _, _, m = verify_one(cfg, split, idx)
            ids_s, ids_t = take_pairs(m, args.tokens, 1000 + idx + ep * 100000)
            x = s.index_select(0, ids_s).float().to(device)
            y = t.index_select(0, ids_t).float().to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = adapter.normalized(x).float()
                tgt = (y - adapter.y_mean)/adapter.y_std
                loss = F.mse_loss(pred, tgt)
            (loss/args.accum).backward()
            loss_total += float(loss.detach())
            pair_total += len(ids_s)
            if (step+1) % args.accum == 0 or step+1 == len(order):
                torch.nn.utils.clip_grad_norm_(adapter.linear.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            if (step+1) % 500 == 0:
                print(f"TRAIN {cfg.name}/{split} ep={ep+1}/{args.epochs} "
                      f"{step+1}/8100 MSE={loss_total/(step+1):.6f}", flush=True)
        score = val_mse(adapter, cfg, split, va_ids, device, args.tokens)
        if not math.isfinite(score):
            raise RuntimeError(f"Invalid validation MSE: {score}")
        print(f"EPOCH END {cfg.name}/{split}: epoch={ep+1}/{args.epochs}, "
              f"trainMSE={loss_total/len(order):.6f}, valMSE={score:.7f}, "
              f"selected_pairs={pair_total:,}", flush=True)
        if score < best:
            best = score
            atomic_torch({
                "adapter": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                "epoch": ep+1, "val_normalized_mse": score,
                "config": {"source": cfg.source, "target": cfg.target, "train_split": split,
                           "architecture": f"Linear({cfg.source_dim},{cfg.target_dim},bias=True)",
                           "token_alignment": "EXACT same role-relative character end; content only",
                           "eval_grid": "target tokenizer grid and target assistant_start",
                           "seed": SEED, "lr": args.lr, "tokens": args.tokens, "epochs": args.epochs,
                           "accum": args.accum, "norm_samples": args.norm_samples},
            }, best_file)
            print("BEST CHECKPOINT", best_file, flush=True)
        atomic_torch({
            "adapter": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
            "optimizer": opt.state_dict(), "completed_epochs": ep+1, "best_val_mse": best,
            "config": {"epochs": args.epochs, "tokens": args.tokens, "lr": args.lr,
                       "accum": args.accum, "norm_samples": args.norm_samples},
        }, last_file)
    if not best_file.exists():
        raise RuntimeError("No best adapter checkpoint after training")
    atomic_json({"epochs": args.epochs, "tokens": args.tokens, "lr": args.lr,
                 "seed": SEED, "train_split": split, "direction": cfg.name}, done_file)
    print(f"TRAIN COMPLETE {cfg.name}/{split}; best normalized MSE={best:.6f}", flush=True)


def load_adapter(cfg, split, device):
    folder = cfg.adapter_dir(split)
    if not (folder / "training_complete.json").is_file():
        raise RuntimeError(f"Adapter training incomplete: {folder}")
    file = folder / "affine_best.pt"
    p = torch.load(file, map_location="cpu", weights_only=True)
    if p["config"]["source"] != cfg.source or p["config"]["target"] != cfg.target or p["config"]["train_split"] != split:
        raise RuntimeError(f"Wrong adapter direction/split: {file}")
    adapter = AffineAdapter(cfg.source_dim, cfg.target_dim).to(device)
    adapter.load_state_dict(p["adapter"], strict=True)
    adapter.eval().requires_grad_(False)
    return adapter, file, p


def load_guard(cfg, device):
    from models import StreamingSafetyHead
    h = StreamingSafetyHead(input_dim=cfg.target_dim, proj_dim=1024, mem_dim=1024,
                            num_labels=2, use_dt=True)
    state = torch.load(cfg.head_ckpt, map_location="cpu", weights_only=True)
    h.load_state_dict(state, strict=True)
    h.to(device, dtype=torch.bfloat16).eval().requires_grad_(False)
    return h


def smoke(cfg, device):
    for split in ("src_train", "src_test", "tgt_train", "tgt_test"):
        n = cfg.n(split)
        require_full(cfg.map_dir(split), n)
        for idx in (0, 1, n-1):
            source, target, start, labels, m = verify_one(cfg, split, idx)
            print(f"SMOKE DATA {cfg.name}/{split}/{idx}: src={tuple(source.shape)} "
                  f"tgt={tuple(target.shape)} exact={len(m['exact_target'])}", flush=True)
    h = load_guard(cfg, device)
    a = AffineAdapter(cfg.source_dim, cfg.target_dim).to(device).eval()
    source, target, start, labels, m = verify_one(cfg, "src_test", 0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        aligned = source.index_select(0, m["src_for_target"]).float().to(device)
        mapped = a(aligned).unsqueeze(0).to(torch.bfloat16)
        logits = h(mapped, start)
    if tuple(logits.shape) != (1, len(labels), 2):
        raise RuntimeError(f"Guard shape mismatch {cfg.name}: {tuple(logits.shape)}")
    print(f"SMOKE PASSED {cfg.name} (random adapter logits ignored)", flush=True)


def evaluate(cfg, case, device):
    specs = {
        "b": ("src_test", "src_train"),
        "c": ("src_test", None),
        "e": ("tgt_test", "tgt_train"),
        "f": ("src_test", "tgt_train"),
    }
    split, train_split = specs[case]
    outjson = cfg.output / f"metrics_{case}.json"
    outcsv = cfg.output / f"predictions_{case}.csv"
    if outjson.exists() and outcsv.exists():
        m = json.loads(outjson.read_text())
        if m.get("direction") != cfg.name or m.get("experiment") != case.upper():
            raise RuntimeError(f"Existing results belong to wrong run: {outjson}")
        print(f"EVAL SKIP {cfg.name}/{case}: previously completed: {m['streaming_f1_label1']:.4f}", flush=True)
        return
    n = cfg.n(split)
    require_full(cfg.cache(split, "tgt"), n)
    if train_split:
        require_full(cfg.cache(split, "src"), n)
        require_full(cfg.map_dir(split), n)
        adapter, ckpt_file, ckpt = load_adapter(cfg, train_split, device)
    else:
        adapter, ckpt_file, ckpt = None, None, None
    guard = load_guard(cfg, device)
    truth, response_pred, stream_pred = [], [], []
    print(f"EVAL {cfg.name} case={case.upper()}, text={split}, "
          f"frozen_target_head={cfg.head_ckpt}, adapter_train={train_split}", flush=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for idx in range(n):
            tgt, start, labs = read_h(cfg.cache(split, "tgt"), idx, cfg.target_dim)
            if adapter is None:
                feat = tgt.unsqueeze(0).to(device, dtype=torch.bfloat16)
            else:
                src, _, src_labels = read_h(cfg.cache(split, "src"), idx, cfg.source_dim)
                m = read_map(cfg, split, idx)
                ix = m["src_for_target"]
                if len(ix) != len(tgt) or int(ix.max()) >= len(src):
                    raise RuntimeError(f"Bad eval mapping {cfg.name}/{split}/{idx}")
                if int(src_labels[-1]) != int(labs[-1]):
                    raise RuntimeError(f"Label mismatch {cfg.name}/{split}/{idx}")
                aligned = src.index_select(0, ix).float().to(device)
                feat = adapter(aligned).unsqueeze(0).to(torch.bfloat16)
            logits = guard(feat, start)
            preds = logits.argmax(-1).view(-1).cpu().tolist()
            if len(preds) != len(labs) or len(preds) < 2:
                raise RuntimeError(f"Response length mismatch: {cfg.name}/{split}/{idx}")
            truth.append(int(labs[-1]))
            response_pred.append(int(preds[-2]))
            stream_pred.append(int(max(preds)))
            if (idx+1) % 100 == 0:
                print(f"EVAL {cfg.name}/{case}: {idx+1}/{n}", flush=True)
    print("RESPONSE LEVEL\n" + classification_report(truth, response_pred, labels=[0,1], digits=4, zero_division=0), flush=True)
    print("STREAMING LEVEL\n" + classification_report(truth, stream_pred, labels=[0,1], digits=4, zero_division=0), flush=True)
    stats = {
        "direction": cfg.name, "experiment": case.upper(), "test_split": split,
        "adapter_train_split": train_split, "target_head_checkpoint": str(cfg.head_ckpt),
        "adapter_checkpoint": str(ckpt_file) if ckpt_file else None,
        "adapter_best_epoch": ckpt["epoch"] if ckpt else None,
        "adapter_val_normalized_mse": ckpt["val_normalized_mse"] if ckpt else None,
        "eval_grid": cfg.target, "protocol": "relative_char_end_v1",
        "n_test": n,
        "response_f1_label1": float(f1_score(truth, response_pred, pos_label=1, zero_division=0)),
        "streaming_f1_label1": float(f1_score(truth, stream_pred, pos_label=1, zero_division=0)),
        "response_accuracy": float(accuracy_score(truth, response_pred)),
        "streaming_accuracy": float(accuracy_score(truth, stream_pred)),
    }
    atomic_json(stats, outjson)
    outcsv.parent.mkdir(parents=True, exist_ok=True)
    temp = outcsv.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["index", "label", "response_prediction", "streaming_prediction"])
        wr.writerows((i, truth[i], response_pred[i], stream_pred[i]) for i in range(n))
    os.replace(temp, outcsv)
    print("RESULT", json.dumps(stats, ensure_ascii=False), flush=True)
    print(f"EXPERIMENT {cfg.name}/{case.upper()} COMPLETE", flush=True)


def report(cfg):
    print(f"=== {cfg.name}: B/C/E/F ===")
    print(f"{'CASE':<7} {'Response F1':>14} {'Streaming F1':>14} {'Response Acc':>14} {'Stream Acc':>13}")
    for case in "bcef":
        file = cfg.output / f"metrics_{case}.json"
        if not file.is_file():
            print(case.upper(), "INCOMPLETE")
            continue
        m = json.loads(file.read_text())
        print(f"{case.upper():<7} {m['response_f1_label1']:>14.4f} {m['streaming_f1_label1']:>14.4f} "
              f"{m['response_accuracy']:>14.4f} {m['streaming_accuracy']:>13.4f}")


def main_for(source, target):
    cfg = Config(source, target)
    parser = argparse.ArgumentParser(description=f"Frozen PlugGuard cross-model transfer: {cfg.name}")
    parser.add_argument("--mode", required=True,
                        choices=["audit", "prepare-caches", "prepare-maps", "smoke",
                                 "train-b", "eval-b", "eval-c", "train-legacy", "eval-e", "eval-f", "report"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--norm-samples", type=int, default=320)
    args = parser.parse_args()
    if args.epochs < 1 or args.tokens < 8 or args.accum < 1 or not (1 <= args.norm_samples <= 8100):
        parser.error("Invalid training parameters")
    print(f"=== {cfg.name} mode={args.mode} ===", flush=True)
    if args.mode == "audit":
        check_input(cfg)
        audit_data_and_templates(cfg)
    elif args.mode == "prepare-caches":
        build_missing_caches(cfg)
    elif args.mode == "prepare-maps":
        prepare_maps(cfg)
    elif args.mode == "report":
        report(cfg)
    else:
        device = gpu0()
        if args.mode == "smoke":
            smoke(cfg, device)
        elif args.mode == "train-b":
            train(cfg, "src_train", device, args)
        elif args.mode == "train-legacy":
            train(cfg, "tgt_train", device, args)
        elif args.mode.startswith("eval-"):
            evaluate(cfg, args.mode[-1], device)

if __name__ == "__main__":
    raise SystemExit("Use one of the four exp5/exp6/exp7/exp8 entry point files")
