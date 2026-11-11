#!/usr/bin/env python3
"""Run one of the six InternLM3 directed cross-model transfer A--F groups.

Examples:
  python intern_adapter_entry.py --source qwen8 --target intern --mode audit
  python intern_adapter_entry.py --source intern --target llama --mode report
"""
import argparse
import sys

from intern_adapter_runtime import DIRECTIONS  # patches the shared runtime first
from cross_adapter_training import main_for

p = argparse.ArgumentParser(add_help=False)
p.add_argument("--source", required=True)
p.add_argument("--target", required=True)
a, remaining = p.parse_known_args()
if (a.source, a.target) not in DIRECTIONS:
    p.error(f"Unsupported direction {a.source}->{a.target}; allowed: {DIRECTIONS}")
sys.argv = [sys.argv[0]] + remaining
main_for(a.source, a.target)
