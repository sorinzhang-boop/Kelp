#!/usr/bin/env python3
"""Experiment 8: Qwen3-14B -> Llama-3.1-8B; B/C/E/F. Shared Llama-style tokenizer alignment."""
from cross_adapter_training import main_for

if __name__ == "__main__":
    main_for("qwen14", "llama")
