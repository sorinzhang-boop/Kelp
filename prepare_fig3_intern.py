#!/usr/bin/env python3
"""Apply minimal, backed-up InternLM3 compatibility changes to Kelp.

Designed for the publicly mirrored Kelp files; refuses unexpected source layouts.
"""
from pathlib import Path
from datetime import datetime
import ast
import shutil

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
MODEL = WORKSPACE / 'models' / 'InternLM3-8B-Instruct'
MODEL_STR = str(MODEL)
assert MODEL.is_dir() and (MODEL / 'config.json').is_file(), f'Missing model: {MODEL}'
backup_dir = WORKSPACE / 'fig3_intern_source_backups'
backup_dir.mkdir(exist_ok=True)


def replace_once(data, before, after, desc):
    if after in data:
        print('  already patched:', desc)
        return data
    if data.count(before) != 1:
        raise RuntimeError(f'Cannot safely patch {desc}: expected exactly one match, found {data.count(before)}')
    return data.replace(before, after, 1)


def apply(path, modify):
    source = path.read_text(encoding='utf-8')
    result = modify(source)
    ast.parse(result, filename=str(path))
    if result != source:
        bak = backup_dir / (path.name + '.before_intern7.py')
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(result, encoding='utf-8')
        print('PATCHED', path.name, '(backup:', bak, ')')
    else:
        print('UNCHANGED', path.name)


def patch_config(t):
    if '"internlm3_8b":' in t:
        assert MODEL_STR in t, 'Existing internlm3_8b points to another path!'
        return t
    anchor = '\n}\n\n\nTRAIN_CONFIG'
    assert t.count(anchor) == 1, 'Unexpected config.py structure'
    addition = ('\n    "internlm3_8b": {\n'
                f'        "model_name": "{MODEL_STR}",\n'
                '        "idx_layer": 24,\n'
                '        "assistant_tokens": "<|im_start|>assistant\\n",\n'
                '    },')
    return t.replace(anchor, addition + anchor, 1)


def patch_dataset(t):
    old = 'if self.model_name == MODEL_CONFIGS["llama_3_1_8b"]["model_name"]:'
    new = ('if self.model_name in (MODEL_CONFIGS["llama_3_1_8b"]["model_name"], '
           'MODEL_CONFIGS["internlm3_8b"]["model_name"]):')
    old_short = ('if self.model_name == MODEL_CONFIGS["llama_3_1_8b"]["model_name"] '
                 'and seq_len < self.num_supervised_token:')
    new_short = ('if self.model_name in (MODEL_CONFIGS["llama_3_1_8b"]["model_name"], '
                 'MODEL_CONFIGS["internlm3_8b"]["model_name"]) '
                 'and seq_len < self.num_supervised_token:')
    t = replace_once(t, old, new, 'assistant marker')
    t = replace_once(t, old_short, new_short, 'short response supervision')
    print('  fixed InternLM3 assistant marker handling and short responses')

    # The original only checks whether ANY sample cache exists. An interrupted
    # 9000-sample extraction otherwise silently trains on a partial dataset.
    a = 'need_build = (len(glob.glob(os.path.join(self.cache_dir, "sample_*.pt"))) == 0)'
    b = ('expected_files = len(load_from_disk(self.dataset_dir))\n'
         '        need_build = (len(glob.glob(os.path.join(self.cache_dir, "sample_*.pt"))) != expected_files)')
    t = replace_once(t, a, b, 'resume partial hidden-state caches')
    return t


def patch_train(t):
    t = replace_once(t,
        '        device_map="auto"\n    )\n    base_model.eval()',
        '        device_map="auto",\n        ' + 'trust_remote_code=True,\n        local_files_only=True,\n    )\n    base_model.eval()',
        'train model loader')
    t = replace_once(t,
        'AutoConfig.from_pretrained(args.model_name).hidden_size',
        'AutoConfig.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=True).hidden_size',
        'train model config')
    return t


def patch_eval(t):
    t = replace_once(t,
        '            device_map=device\n        )\n        base_model.to(',
        '            device_map=device,\n            trust_remote_code=True,\n            local_files_only=True,\n        )\n        base_model.to(',
        'eval model loader')
    t = replace_once(t,
        'AutoTokenizer.from_pretrained(model_name)\n',
        'AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)\n',
        'eval tokenizer loader')
    return t


if __name__ == '__main__':
    assert (HERE / 'train.py').is_file(), f'Put this script in Kelp root: {HERE}'
    apply(HERE / 'config.py', patch_config)
    apply(HERE / 'dataset.py', patch_dataset)
    apply(HERE / 'train.py', patch_train)
    apply(HERE / 'eval.py', patch_eval)
    print('PASS: Kelp changes prepared; existing checkpoints were not touched')
