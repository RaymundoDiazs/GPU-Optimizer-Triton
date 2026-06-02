import argparse
import ast
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "model_eval.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "artifacts"

RESULT_FIELDS = [
    "timestamp",
    "model_id",
    "model_display_name",
    "tier",
    "provider",
    "mode",
    "constrained_decoding_backend",
    "task_id",
    "sample_index",
    "latency_seconds",
    "syntax_valid",
    "kernel_shape_valid",
    "correctness_proxy",
    "notes",
]


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the model evaluation configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def extract_code(text: str) -> str:
    """Extract Python code from a model response."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def evaluate_generated_code(code: str, task: dict[str, Any]) -> dict[str, bool | str]:
    """Score generated code with lightweight checks that run without GPU access."""
    normalized = re.sub(r"\s+", " ", code.lower())
    try:
        ast.parse(code)
        syntax_valid = True
    except SyntaxError as exc:
        syntax_valid = False
        return {
            "syntax_valid": False,
            "kernel_shape_valid": False,
            "correctness_proxy": False,
            "notes": f"syntax_error={exc.msg}",
        }

    kernel_shape_valid = all(term.lower() in normalized for term in ["@triton.jit", "tl.load", "tl.store"])
    expected_terms = [term.lower() for term in task.get("expected_terms", [])]
    correctness_proxy = all(term in normalized for term in expected_terms)
    notes = "heuristic_checks_only"
    return {
        "syntax_valid": syntax_valid,
        "kernel_shape_valid": kernel_shape_valid,
        "correctness_proxy": correctness_proxy,
        "notes": notes,
    }


def load_manual_outputs(manual_path: Path) -> list[dict[str, Any]]:
    """Load manually collected model outputs from JSONL."""
    rows = []
    with manual_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_model_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    samples_per_model: int | None = None,
    manual_outputs: Path | None = None,
) -> list[dict[str, Any]]:
    """Process PyTorch→Triton translation outputs and write evaluation artifacts."""
    if not manual_outputs:
        print(
            "ERROR: --manual-outputs is required. "
            "This script only evaluates real translation outputs collected via "
            "collect_real_outputs.py. It does not generate mock outputs.",
            file=sys.stderr,
        )
        sys.exit(1)

    load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_path = output_dir / "generated_kernels.jsonl"
    results_path = output_dir / "model_eval_results.csv"
    summary_path = output_dir / "model_eval_summary.md"

    rows: list[dict[str, Any]] = []
    generated_records: list[dict[str, Any]] = []

    candidates = load_manual_outputs(manual_outputs)

    for candidate in candidates:
        model = candidate["model"]
        task = candidate["task"]
        code = extract_code(candidate["output"])
        checks = evaluate_generated_code(code, task)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": model["id"],
            "model_display_name": model.get("display_name", model["id"]),
            "tier": model.get("tier", "unknown"),
            "provider": model.get("provider", "unknown"),
            "mode": candidate["mode"],
            "constrained_decoding_backend": candidate.get("constrained_decoding_backend", ""),
            "task_id": task["id"],
            "sample_index": candidate["sample_index"],
            "latency_seconds": f"{float(candidate.get('latency_seconds', 0.0)):.6f}",
            "syntax_valid": checks["syntax_valid"],
            "kernel_shape_valid": checks["kernel_shape_valid"],
            "correctness_proxy": checks["correctness_proxy"],
            "notes": checks["notes"],
        }
        rows.append(row)
        generated_records.append(
            {
                **row,
                "prompt": candidate.get("prompt", ""),
                "output": candidate["output"],
                "extracted_code": code,
            }
        )

    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with generated_path.open("w", encoding="utf-8") as file:
        for record in generated_records:
            file.write(json.dumps(record) + "\n")

    summary_path.write_text(_build_summary(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} model evaluation rows to {results_path}")
    print(f"Wrote generated outputs to {generated_path}")
    print(f"Wrote summary to {summary_path}")
    return rows


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(str(row[key]) == "True" or row[key] is True for row in rows) / len(rows)


def _build_summary(rows: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model_id"], row["mode"]), []).append(row)

    lines = [
        "# PyTorch to Triton Translation — Model Evaluation Summary",
        "",
        "| Model | Mode | Samples | Syntax valid | Kernel shape valid | Correctness proxy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for (model_id, mode), group_rows in sorted(groups.items()):
        lines.append(
            f"| {model_id} | {mode} | {len(group_rows)} | "
            f"{_rate(group_rows, 'syntax_valid'):.0%} | "
            f"{_rate(group_rows, 'kernel_shape_valid'):.0%} | "
            f"{_rate(group_rows, 'correctness_proxy'):.0%} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- These are lightweight heuristic checks (syntax, kernel shape, expected terms).",
            "- GPU compilation and numerical correctness are validated downstream by TritonBench.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PyTorch→Triton translation outputs from LLMs (Qwen local, GPT-4o, Claude Haiku)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--manual-outputs",
        type=Path,
        default=None,
        help="JSONL of real translation outputs collected via collect_real_outputs.py (required).",
    )
    args = parser.parse_args()
    run_model_evaluation(
        config_path=args.config,
        output_dir=args.output_dir,
        samples_per_model=args.samples,
        manual_outputs=args.manual_outputs,
    )


if __name__ == "__main__":
    main()
