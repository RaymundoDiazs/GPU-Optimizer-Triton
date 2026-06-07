import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parsing.tritonbench_grammar_rules import (
    validate_tritonbench_candidate,
    validate_tritonbench_family_candidate,
)


DEFAULT_INPUT = ROOT / "evaluation" / "artifacts" / "generated_kernels.jsonl"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "tritonbench_grammar_results.csv"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _task_id(record: dict) -> str:
    if record.get("task_id"):
        return str(record["task_id"])
    task = record.get("task", {})
    task_id = task.get("id", "")
    if task_id:
        return task_id
    source_index = task.get("source_index")
    if source_index:
        return f"tritonbench_t_{int(source_index):03d}"
    return ""


def run_grammar_benchmark(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> list[dict]:
    """Validate generated TritonBench-T candidates against grammar contracts."""
    records = _read_jsonl(input_path)
    if not records:
        raise ValueError(f"Input JSONL contains no records: {input_path}")
    results = []

    for record in records:
        task_id = _task_id(record)
        if not task_id.startswith("tritonbench_t_"):
            continue

        model = record.get("model", {})
        output = record.get("extracted_code", record.get("output", ""))
        individual = validate_tritonbench_candidate(task_id, output)
        family = validate_tritonbench_family_candidate(task_id, output)
        individual_valid = individual.valid and not any(
            "TODO/pass" in warning for warning in individual.warnings
        )
        family_valid = family.valid and not any(
            "TODO/pass" in warning for warning in family.warnings
        )
        results.append(
            {
                "model_id": record.get("model_id", model.get("id", "")),
                "provider": record.get("provider", model.get("provider", "")),
                "task_id": task_id,
                "mode": record.get("mode", ""),
                "sample_index": record.get("sample_index", ""),
                "individual_valid": individual_valid,
                "family_valid": family_valid,
                "validations_match": individual_valid == family_valid,
                "individual_passed_rule_count": len(individual.passed_rules),
                "family_passed_rule_count": len(family.passed_rules),
                "individual_error_count": len(individual.errors),
                "family_error_count": len(family.errors),
                "warning_count": len(individual.warnings + family.warnings),
                "individual_errors": " | ".join(individual.errors),
                "family_errors": " | ".join(family.errors),
                "warnings": " | ".join(individual.warnings + family.warnings),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_id",
        "provider",
        "task_id",
        "mode",
        "sample_index",
        "individual_valid",
        "family_valid",
        "validations_match",
        "individual_passed_rule_count",
        "family_passed_rule_count",
        "individual_error_count",
        "family_error_count",
        "warning_count",
        "individual_errors",
        "family_errors",
        "warnings",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark generated TritonBench-T outputs against the local grammar contracts."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSONL from collect_real_outputs.py")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    args = parser.parse_args()

    try:
        rows = run_grammar_benchmark(Path(args.input), Path(args.output))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    total = len(rows)
    individual_passed = sum(1 for row in rows if row["individual_valid"])
    family_passed = sum(1 for row in rows if row["family_valid"])
    matched = sum(1 for row in rows if row["validations_match"])
    individual_rate = (individual_passed / total * 100) if total else 0.0
    family_rate = (family_passed / total * 100) if total else 0.0
    match_rate = (matched / total * 100) if total else 0.0
    print(f"Grammar benchmark rows: {total}")
    print(f"Individual pass rate: {individual_passed}/{total} ({individual_rate:.1f}%)")
    print(f"Family pass rate: {family_passed}/{total} ({family_rate:.1f}%)")
    print(f"Individual/family agreement: {matched}/{total} ({match_rate:.1f}%)")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
