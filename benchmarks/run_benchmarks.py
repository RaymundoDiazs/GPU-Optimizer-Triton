import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import run_pipeline


DEFAULT_PROBLEMS = [
    {"id": "vector_add", "code": "C = A + B"},
    {"id": "reduction_sum", "code": "output = torch.sum(x)"},
    {"id": "matmul", "code": "C = A @ B"},
]
FIELDNAMES = [
    "timestamp",
    "id",
    "code_snippet",
    "problem_type",
    "kernel_name",
    "generation_time",
    "notes",
]


def load_problems(problems_path: Path | None = None) -> list[dict[str, str]]:
    """Load benchmark problems from JSON, falling back to built-in examples."""
    candidate = problems_path or REPO_ROOT / "data" / "sample_problems.json"
    if not candidate.exists():
        return DEFAULT_PROBLEMS

    with candidate.open("r", encoding="utf-8") as file:
        problems = json.load(file)

    normalized = []
    for index, problem in enumerate(problems, start=1):
        normalized.append(
            {
                "id": str(problem.get("id") or f"problem_{index}"),
                "code": str(problem["code"]),
            }
        )
    return normalized


def _extract_problem_type(pipeline_output: str) -> str:
    marker = "Problem type: "
    for line in pipeline_output.splitlines():
        if line.startswith(marker):
            return line.removeprefix(marker)
    return "unknown"


def _infer_kernel_name(pipeline_output: str) -> str:
    if "def elementwise_kernel" in pipeline_output:
        return "elementwise_kernel"
    if "def reduction_kernel" in pipeline_output:
        return "reduction_kernel"
    if "def matmul_kernel" in pipeline_output:
        return "matmul_kernel"
    if "def generic_kernel" in pipeline_output:
        return "generic_kernel"
    return "unknown"


def run_benchmarks(
    runs: int = 1,
    problems_path: Path | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Measure kernel generation time and write reproducible benchmark rows."""
    if runs < 1:
        raise ValueError("runs must be >= 1")

    problems = load_problems(problems_path)
    destination = output_path or Path(__file__).resolve().parent / "results.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for problem in problems:
        for run_index in range(1, runs + 1):
            start = time.perf_counter()
            pipeline_output = run_pipeline(problem["code"])
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "id": problem["id"],
                    "code_snippet": problem["code"],
                    "problem_type": _extract_problem_type(pipeline_output),
                    "kernel_name": _infer_kernel_name(pipeline_output),
                    "generation_time": f"{elapsed:.9f}",
                    "notes": f"run={run_index}",
                }
            )

    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} benchmark rows to {destination}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible generation benchmarks.")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeats per problem.")
    parser.add_argument(
        "--problems",
        type=Path,
        default=None,
        help="Optional JSON file with benchmark problems.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    args = parser.parse_args()
    run_benchmarks(runs=args.runs, problems_path=args.problems, output_path=args.output)


if __name__ == "__main__":
    main()
