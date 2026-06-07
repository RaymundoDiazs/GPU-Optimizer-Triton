from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_generated_kernels import (
    DEFAULT_FIXED_BLOCK_SIZE,
    DEFAULT_NUM_ELEMENTS,
    DEFAULT_REPEATS,
    DEFAULT_WARMUP,
    evaluate_file,
)


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _summarize(path: Path, policy: str) -> dict[str, Any]:
    rows = _jsonl_load(path)
    n = len(rows)
    compiled = sum(row.get("triton_compiles") is True for row in rows)
    correct = sum(row.get("triton_numerically_correct") is True for row in rows)
    speedups = [
        float(row["triton_speedup"])
        for row in rows
        if row.get("triton_numerically_correct") is True
        and row.get("triton_speedup") is not None
    ]
    return {
        "launch_policy": policy,
        "n": n,
        "compiled": compiled,
        "compile_success_rate": compiled / n if n else 0.0,
        "correct": correct,
        "execution_accuracy": correct / n if n else 0.0,
        "speedup_mean": mean(speedups) if speedups else "",
        "speedups_recorded": len(speedups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed vs shape-aware launch policies on the same generated kernels."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/launch_policy_comparison"))
    parser.add_argument("--num-elements", type=int, default=DEFAULT_NUM_ELEMENTS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--fixed-block-size", type=int, default=DEFAULT_FIXED_BLOCK_SIZE)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for policy in ("fixed", "shape-aware"):
        output = args.output_dir / f"{policy}_results.jsonl"
        metadata = args.output_dir / f"{policy}_metadata.json"
        diagnostics = args.output_dir / f"{policy}_diagnostics.jsonl"
        evaluate_file(
            args.input,
            output,
            metadata,
            diagnostics,
            num_elements=args.num_elements,
            warmup=args.warmup,
            repeats=args.repeats,
            launch_policy=policy,
            fixed_block_size=args.fixed_block_size,
        )
        summaries.append(_summarize(output, policy))

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote comparison outputs to {args.output_dir}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
