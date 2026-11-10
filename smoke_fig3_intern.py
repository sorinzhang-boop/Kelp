#!/usr/bin/env python3
"""Test real InternLM3 data through original SafetyDataset -> GPU -> SafetyHead.
Does not write to production train/test caches or checkpoints.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from pathlib import Path
import tempfile
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_CONFIGS
from dataset import SafetyDataset, find_sequence
from models import StreamingSafetyHead

ROOT = Path(__file__).resolve().parent.parent
MODEL = MODEL_CONFIGS['internlm3_8b']['model_name']
DATA = ROOT / 'data' / 'StreamGuardBench_v1' / 's_eval' / 'internlm3_8_instruct'
assert MODEL == str(ROOT / 'models' / 'InternLM3-8B-Instruct'), MODEL
assert torch.cuda.is_available(), 'No CUDA GPU available'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
train = load_from_disk(str(DATA / 'train'))
test = load_from_disk(str(DATA / 'test'))
assert len(train) == 9000 and sum(map(int, train['label'])) == 2611
assert len(test) == 1000 and sum(map(int, test['label'])) == 385

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
marker_ids = tok.encode(MODEL_CONFIGS['internlm3_8b']['assistant_tokens'], add_special_tokens=False)
assert len(marker_ids) > 0
# Nontrivial samples from each class, both train and test.
for ds in (train, test):
    ids = [next(i for i in range(len(ds)) if int(ds[i]['label']) == v) for v in (0, 1)]
    for idx in ids:
        row = ds[idx]
        formatted = tok.apply_chat_template(
            [{'role': 'user', 'content': row['prompt']},
             {'role': 'assistant', 'content': row['response']}],
            tokenize=False, add_generation_prompt=True)
        tokens = tok([formatted], return_tensors='pt').input_ids[0].tolist()
        start = find_sequence(tokens, marker_ids) + len(marker_ids)
        assert 0 < start < len(tokens) - 1, ('bad assistant_start', idx, start, len(tokens))
        assert find_sequence(tokens, marker_ids) >= 0, 'assistant marker missing'
        print('Tokenizer OK: label', row['label'], 'tokens', len(tokens), 'assistant_start', start, flush=True)

print('Loading InternLM3 on GPU...', flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
    local_files_only=True).eval()
with tempfile.TemporaryDirectory(prefix='intern_smoke_', dir=str(ROOT)) as tmp:
    # Build a tiny real HF dataset, so actual Kelp SafetyDataset logic is tested.
    real_indices = [next(i for i in range(len(train)) if int(train[i]['label']) == v) for v in (0, 1)]
    mini = Path(tmp) / 'mini_train'
    train.select(real_indices).save_to_disk(str(mini))
    ds = SafetyDataset(
        dataset_dir=str(mini), model_name=MODEL, tokenizer=tok,
        base_model=model, device=torch.device('cuda:0'), idx_layer=24,
        max_length=4096, build_cache_if_missing=True, overwrite=False)
    assert len(ds) == 2, f'Expected 2 smoke samples, got {len(ds)}'
    sample = ds[0]
    features = sample['embeddings'].unsqueeze(0).to('cuda', torch.bfloat16)
    astart = int(sample['assistant_start'])
    labels = sample['labels'].to('cuda').unsqueeze(0)
    assert features.shape[-1] == 4096
    assert features.shape[1] - astart == labels.shape[1]
    assert labels[0, -1].item() == train[real_indices[0]]['label']
    print('Real cache OK:', tuple(features.shape), 'start', astart, 'labels', tuple(labels.shape), flush=True)
    head = StreamingSafetyHead(input_dim=4096, proj_dim=1024, mem_dim=1024, num_labels=2, use_dt=True).to('cuda', torch.bfloat16)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = head(features, astart)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), ignore_index=-100)
    assert logits.shape[1] == labels.shape[1] and torch.isfinite(loss)
    loss.backward()
    print('Head forward/backward OK: loss', float(loss.item()), flush=True)
print('PASS: InternLM3 real-data feature extraction and PlugGuard head smoke test', flush=True)
