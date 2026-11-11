#!/usr/bin/env bash
# InternLM3 directed transfers: preflight, cache, maps, smoke, B/C/E/F.
# Strictly serial, physical GPU 1 only, resumable training and evaluations.
set -Eeuo pipefail
cd "$HOME/plugguard_repro/Kelp"
source "$HOME/plugguard_repro/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/plugguard_repro/logs"
LOG="$HOME/plugguard_repro/logs/intern_adapter_six_gpu1.log"
LOCK="$HOME/plugguard_repro/logs/intern_adapter_six_gpu1.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "Another Intern adapter controller is active" >&2; exit 1; }
exec > >(tee -a "$LOG") 2>&1

DIRS=("qwen8 intern" "qwen14 intern" "llama intern" "intern qwen8" "intern qwen14" "intern llama")
MODE=${1:-full}
if [[ "$MODE" != full && "$MODE" != check ]]; then
  echo "Usage: bash run_intern_adapter_six_gpu1.sh [check|full]" >&2; exit 2
fi

echo "START $(date -Is), mode=$MODE, physical GPU 1"
python -m py_compile intern_adapter_entry.py intern_adapter_runtime.py intern_adapter_report.py

# Read-only: verify model dirs, datasets, target heads and token reconstruction.
for pair in "${DIRS[@]}"; do
  read -r src tgt <<< "$pair"
  echo "AUDIT: $src -> $tgt"
  python -u intern_adapter_entry.py --source "$src" --target "$tgt" --mode audit
done
if [[ "$MODE" == check ]]; then
  echo "PASS ALL SIX AUDITS (no GPU jobs launched)"
  exit 0
fi

for pair in "${DIRS[@]}"; do
  read -r src tgt <<< "$pair"
  echo "=== BEGIN $src -> $tgt $(date -Is) ==="
  for mode in prepare-caches prepare-maps smoke eval-c train-b eval-b train-legacy eval-e eval-f report; do
    echo "[$src -> $tgt] mode=$mode $(date -Is)"
    python -u intern_adapter_entry.py --source "$src" --target "$tgt" --mode "$mode"
  done
  python -u intern_adapter_report.py --mode partial
  echo "=== COMPLETE $src -> $tgt $(date -Is) ==="
done
python -u intern_adapter_report.py --mode complete
echo "ALL SIX INTERN TRANSFER DIRECTIONS COMPLETE $(date -Is)"
