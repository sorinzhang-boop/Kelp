#!/usr/bin/env bash
# Strictly sequential experiments, all GPU work on physical GPU 0.
set -Eeuo pipefail
cd "$HOME/plugguard_repro/Kelp"
source "$HOME/plugguard_repro/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/plugguard_repro/logs"
LOCK="$HOME/plugguard_repro/logs/cross_adapter_four_directions.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "Another four-direction controller appears to be running ($LOCK)." >&2
    exit 1
fi
LOG="$HOME/plugguard_repro/logs/exp5_to_exp8_master.log"
exec > >(tee -a "$LOG") 2>&1

DIRS=(
  "exp5_qwen8_to_qwen14.py"
  "exp6_llama_to_qwen14.py"
  "exp7_qwen8_to_llama.py"
  "exp8_qwen14_to_llama.py"
)

fail() { echo "FATAL: $*" >&2; exit 1; }
for f in cross_adapter_alignment.py cross_adapter_training.py "${DIRS[@]}"; do
    [[ -f "$f" ]] || fail "Required script missing: $PWD/$f"
done

echo "==== START FOUR-DIRECTION SERIAL RUN $(date -Is) ===="
echo "Host: $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Python: $(command -v python)"
python -m py_compile cross_adapter_alignment.py cross_adapter_training.py "${DIRS[@]}"

# Check all inputs BEFORE spending hours on the first direction.
for py in "${DIRS[@]}"; do
    echo "PREFLIGHT $py"
    python -u "$py" --mode audit
done

run_one() {
    local py="$1"
    local group="${py%.py}"
    local group_log="$HOME/plugguard_repro/logs/${group}.log"
    echo "===== GROUP START: $group $(date -Is) ====="
    local mode
    for mode in prepare-caches prepare-maps smoke eval-c train-b eval-b train-legacy eval-e eval-f report; do
        echo "===== $group | $mode | $(date -Is) ====="
        # All Python modes are separate processes; a failed command stops entire run.
        python -u "$py" --mode "$mode" 2>&1 | tee -a "$group_log"
    done
    echo "===== GROUP COMPLETE: $group $(date -Is) ====="
}
for py in "${DIRS[@]}"; do
    run_one "$py"
done

python - <<'PY'
import csv, json
from pathlib import Path
root = Path.home()/"plugguard_repro/checkpoints/cross_adapt_four_directions_v1"
names = ["qwen8_to_qwen14", "llama_to_qwen14", "qwen8_to_llama", "qwen14_to_llama"]
rows = []
for name in names:
    for c in "bcef":
        file = root/name/f"metrics_{c}.json"
        if not file.is_file():
            raise RuntimeError(f"Missing finished metric: {file}")
        d = json.loads(file.read_text())
        rows.append([name, c.upper(), d["response_f1_label1"], d["streaming_f1_label1"],
                     d["response_accuracy"], d["streaming_accuracy"]])
out = root/"summary_16_results.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["direction", "case", "response_f1", "streaming_f1", "response_accuracy", "streaming_accuracy"])
    w.writerows(rows)
print("ALL 16 METRICS READY:", out)
for row in rows:
    print(f"{row[0]:24s} {row[1]} Response F1={row[2]:.4f} Streaming F1={row[3]:.4f}")
PY

echo "=== ALL EXPERIMENTS 5/6/7/8 COMPLETE $(date -Is) ==="
