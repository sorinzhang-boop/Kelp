#!/usr/bin/env python3
"""Merge saved B/C/E/F and native A/D metrics; CPU only, no fabricated scores."""
import argparse
import csv
import json
from pathlib import Path
from intern_adapter_runtime import DIRECTIONS, OUT_ROOT

# Native S-Eval baselines recorded in notes/L20/reproduction.md and previous runs.
NATIVE = {
    "qwen8": (0.9245, 0.9228),
    "qwen14": (0.8955, 0.8971),
    "llama": (0.9095, 0.9191),
    "intern": (0.8832, 0.8799),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["partial", "complete"], default="partial")
    args = p.parse_args()
    rows = []
    for source, target in DIRECTIONS:
        direction = f"{source}_to_{target}"
        for case in "ABCDEF":
            if case in "AD":
                model = source if case == "A" else target
                response, streaming = NATIVE[model]
                status = "native_previous_run"
            else:
                file = OUT_ROOT / direction / f"metrics_{case.lower()}.json"
                if not file.is_file():
                    if args.mode == "complete":
                        raise RuntimeError(f"Metric missing: {file}")
                    response = streaming = ""
                    status = "pending"
                else:
                    m = json.loads(file.read_text())
                    if m.get("direction") != direction or m.get("experiment") != case:
                        raise RuntimeError(f"Metric direction/case mismatch: {file}")
                    response = m["response_f1_label1"]
                    streaming = m["streaming_f1_label1"]
                    status = "measured"
            rows.append([direction, source, target, case, response, streaming, status])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUT_ROOT / "summary_36_results.csv"
    tmp = output.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["direction", "source", "target", "case", "response_f1", "streaming_f1", "status"])
        w.writerows(rows)
    tmp.replace(output)
    print(f"SAVED {sum(r[-1] != 'pending' for r in rows)}/36 metrics: {output}")
    for d in DIRECTIONS:
        direc = f"{d[0]}_to_{d[1]}"
        entries = [r for r in rows if r[0] == direc]
        summary = ' '.join(f"{r[3]}={r[5]:.4f}" if isinstance(r[5],float) else f"{r[3]}=pending" for r in entries)
        print(direc, summary)
    if args.mode == "complete":
        assert all(r[-1] != "pending" for r in rows)
        print("ALL SIX DIRECTIONS A-F READY")

if __name__ == '__main__':
    main()
