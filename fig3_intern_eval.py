#!/usr/bin/env python3
"""Re-evaluate a completed checkpoint and write atomic JSON metrics."""
import argparse
import json
import os
from pathlib import Path
from collections import Counter
from sklearn.metrics import f1_score
from eval import evaluate_safety_head

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--test_dir', required=True)
p.add_argument('--model', required=True)
p.add_argument('--layer', required=True, type=int)
p.add_argument('--output', required=True)
p.add_argument('--source', required=True)
p.add_argument('--target', required=True)
a = p.parse_args()

preds, y = evaluate_safety_head(
    ckpt_path=a.checkpoint, test_dataset_dir=a.test_dir,
    model_name=a.model, idx_layer=a.layer, max_length=4096,
    batch_size=1, num_workers=2, bf16=True)
assert len(y) == 1000 and all(len(x) >= 2 for x in preds), 'Incomplete evaluation data'
response = [x[-2] for x in preds]
streaming = [max(x) for x in preds]
metrics = dict(
    source=a.source, target=a.target, samples=len(y),
    harmful_count=sum(int(v) for v in y),
    response_harmful_f1=f1_score(y, response, pos_label=1, zero_division=0),
    streaming_harmful_f1=f1_score(y, streaming, pos_label=1, zero_division=0),
    label_counts=dict(Counter(str(v) for v in y)),
    checkpoint=a.checkpoint, test_dataset=a.test_dir, layer=a.layer)
out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix('.json.tmp')
tmp.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
os.replace(tmp, out)
print(json.dumps(metrics, indent=2), flush=True)
