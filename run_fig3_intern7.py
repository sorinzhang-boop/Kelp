#!/usr/bin/env python3
"""Seven additional Source -> Target PlugGuard Figure 3 cells; sequential GPU0.

Usage: python run_fig3_intern7.py --check  (CPU preflight)
       python -u run_fig3_intern7.py  (in tmux, auto-resumable)
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from datasets import load_from_disk
from config import MODEL_CONFIGS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'data' / 'StreamGuardBench_v1' / 's_eval'
OUT = ROOT / 'checkpoints' / 'fig3_intern_extension_v1'
LOG = ROOT / 'logs' / 'fig3_intern_extension_v1'
SPEC = {
    'qwen8': ('qwen3_8b', 'qwen3_8b', 20),
    'qwen14': ('qwen3_14b', 'qwen3_14b', 20),
    'llama': ('llama_3_1_8b_instruct', 'llama_3_1_8b', 20),
    'intern': ('internlm3_8_instruct', 'internlm3_8b', 24),
}
# Source selects training PAIRS. Target selects the HIDDEN-STATE extractor and test pairs.
TASKS = [
    ('intern', 'qwen8'),
    ('intern', 'qwen14'),
    ('intern', 'llama'),
    ('intern', 'intern'),
    ('qwen8', 'intern'),
    ('qwen14', 'intern'),
    ('llama', 'intern'),
]
assert len(TASKS) == 7

parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true', help='CPU-only preflight; no training')
args = parser.parse_args()

for env_name, value in {
    'CUDA_VISIBLE_DEVICES': '0',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'TOKENIZERS_PARALLELISM': 'false',
    'PYTHONUNBUFFERED': '1',
}.items():
    os.environ[env_name] = value

def check_disk():
    free = os.statvfs(str(ROOT))
    gb = free.f_bavail * free.f_frsize / (1024**3)
    print(f'FREE DISK: {gb:.1f} GiB', flush=True)
    assert gb >= 80, 'Less than 80 GiB disk free; stop before partial writes'

def run_logged(cmd, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print('COMMAND:', ' '.join(map(str,cmd)), flush=True)
    with logfile.open('a', encoding='utf-8') as f:
        f.write('\n' + '=' * 72 + '\n')
        f.write('START ' + datetime.now().isoformat() + ' ' + ' '.join(map(str,cmd)) + '\n')
        f.flush()
        p = subprocess.Popen(cmd, cwd=HERE, env=os.environ.copy(),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
        rc = p.wait()
        f.write(f'END {datetime.now().isoformat()} RC={rc}\n')
    if rc != 0:
        raise RuntimeError(f'Command failed ({rc}): {cmd} | log: {logfile}')

def cache_path(dataset_dir, model, layer):
    return dataset_dir / 'safety_cache' / model.replace('/', '-') / f'idx{layer}_maxlength4096'

def cache_check(dataset_dir, model, layer, expected):
    cache = cache_path(dataset_dir, model, layer)
    files = sorted(cache.glob('sample_*.pt')) if cache.is_dir() else []
    count = len(files)
    print(f'CACHE: {cache} = {count}/{expected}', flush=True)
    if count == expected:
        # Ensure no missing indices that a simple count check would hide.
        assert all((cache / f'sample_{i:08d}.pt').is_file() for i in range(expected)), 'noncontiguous cache'
    return count

print('=== FIGURE 3 INTERNLM3 7-CELL EXTENSION ===', flush=True)
print('Root:', ROOT, 'GPU: visible CUDA device 0', flush=True)
assert HERE == ROOT / 'Kelp', f'Put runner under Kelp: {HERE}'
assert (HERE / 'train.py').is_file()
assert 'internlm3_8b' in MODEL_CONFIGS, 'Run prepare_fig3_intern.py first'
assert Path(MODEL_CONFIGS['internlm3_8b']['model_name']).is_dir()
for name, (data_name, config_key, layer) in SPEC.items():
    model_dir = Path(MODEL_CONFIGS[config_key]['model_name'])
    assert model_dir.is_dir(), f'Missing target model: {model_dir}'
    assert MODEL_CONFIGS[config_key]['idx_layer'] == layer
    train = load_from_disk(str(DATA / data_name / 'train'))
    test = load_from_disk(str(DATA / data_name / 'test'))
    assert len(train) == 9000 and len(test) == 1000, (name, len(train), len(test))
    assert {'prompt', 'response', 'label'}.issubset(train.column_names)
    if name == 'intern':
        assert sum(map(int, train['label'])) == 2611
        assert sum(map(int, test['label'])) == 385
    print('DATA OK:', name, 'train=9000 test=1000', flush=True)
check_disk()

if args.check:
    for src, tgt in TASKS:
        source_dir = DATA / SPEC[src][0] / 'train'
        target_dir = DATA / SPEC[tgt][0] / 'test'
        model = MODEL_CONFIGS[SPEC[tgt][1]]['model_name']
        layer = SPEC[tgt][2]
        print(f'[{src}->{tgt}]')
        cache_check(source_dir, model, layer, 9000)
        cache_check(target_dir, model, layer, 1000)
    print('PASS: 7 experiments ready for smoke/full run', flush=True)
    sys.exit(0)

OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
for n, (src, tgt) in enumerate(TASKS, start=1):
    print('\n'+'#'*80, flush=True)
    print(f'[{n}/7] SOURCE={src} TARGET={tgt}', flush=True)
    check_disk()
    source_dir = DATA / SPEC[src][0] / 'train'
    target_dir = DATA / SPEC[tgt][0] / 'test'
    model = MODEL_CONFIGS[SPEC[tgt][1]]['model_name']
    layer = SPEC[tgt][2]
    job = OUT / f'source_{src}_target_{tgt}'
    ckpt = job / 'model_epoch_0.pt'
    report = job / 'metrics.json'
    if report.exists() and ckpt.exists():
        print('SKIP COMPLETED:', report, flush=True)
        continue
    job.mkdir(parents=True, exist_ok=True)
    joblog = LOG / f'source_{src}_target_{tgt}.log'
    if not ckpt.exists():
        cmd = [sys.executable, '-u', str(HERE / 'train.py'),
               '--train_dataset_dir', str(source_dir),
               '--test_dataset_dir', str(target_dir),
               '--model_name', model,
               '--idx_layer', str(layer),
               '--save_dir', str(job),
               '--lr', '5e-5', '--weight_decay', '0',
               '--batch_size', '1', '--gradient_acc_steps', '32',
               '--max_length', '4096', '--num_train_epochs', '1']
        run_logged(cmd, joblog)
    else:
        print('Found trained checkpoint, will evaluate only:', ckpt, flush=True)
    assert ckpt.is_file(), f'Missing trained checkpoint: {ckpt}'
    cache_check(source_dir, model, layer, 9000)
    cache_check(target_dir, model, layer, 1000)
    cmd = [sys.executable, '-u', str(HERE / 'fig3_intern_eval.py'),
           '--checkpoint', str(ckpt), '--test_dir', str(target_dir),
           '--model', model, '--layer', str(layer),
           '--source', src, '--target', tgt, '--output', str(report)]
    run_logged(cmd, joblog)
    assert report.is_file(), 'Metric report was not written'
    print(f'[{n}/7] COMPLETE {src}->{tgt}: {report}', flush=True)
print('=== ALL SEVEN FIGURE 3 EXPERIMENTS COMPLETED ===', flush=True)
