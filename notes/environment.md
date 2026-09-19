# PlugGuard Reproduction Environment

## Base Repository

- Repository: Alibaba-AAIG/Kelp
- Base commit: `de7abe9bb1af1186e12b3fe5bf7fab65c2fdc550`

## Server

- OS environment: Linux / Kubernetes Pod
- GPU: 8 × NVIDIA RTX 6000D
- GPU memory: ~85,651 MiB per GPU
- NVIDIA Driver: 595.91.07
- NVIDIA-SMI reported CUDA compatibility: 13.2

## Python Environment

- Python: 3.12.3
- Virtual environment: `/data1/plugguard_repro/.venv`

## Core Dependencies

- PyTorch: 2.7.1+cu128
- PyTorch CUDA runtime: 12.8
- NumPy: 2.2.6
- Transformers: 4.56.2
- datasets: 5.0.1
- scikit-learn: 1.9.1
- accelerate: 1.15.0

## GPU Verification

PyTorch CUDA available: `True`

Detected GPU:

`NVIDIA RTX 6000D`

## Notes

The official Kelp repository does not provide a locked Python environment
(`requirements.txt`, `environment.yml`, or `pyproject.toml`).

This reproduction environment was therefore reconstructed based on:
- the released Kelp source code,
- Qwen3 support requirements,
- compatibility with the available RTX 6000D GPU.