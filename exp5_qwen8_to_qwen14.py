#!/usr/bin/env python3
"""Experiment 5: Qwen3-8B -> Qwen3-14B; B/C/E/F. Shared Llama-style tokenizer alignment."""
from cross_adapter_training import main_for

if __name__ == "__main__":
    main_for("qwen8", "qwen14")
