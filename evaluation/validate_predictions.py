"""
Paso 3 — Validador de archivos de predicciones.

Uso:
    python evaluation/validate_predictions.py --file evaluation/predictions_qwen.jsonl
    python evaluation/validate_predictions.py --file evaluation/predictions_gpt4o.jsonl
    python evaluation/validate_predictions.py --file evaluation/predictions_claude.jsonl

Verifica:
  - Exactamente 166 líneas válidas
  - Cada línea tiene los campos "instruction" y "predict"
  - "instruction" coincide con el de simp_alpac_v1.json (por posición)
  - "predict" no contiene fences ```
  - "predict" no está vacío
"""

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.code_safety import validate_generated_code_safety
from evaluation.model_evaluation import extract_code
from evaluation.prediction_schema import load_prediction_jsonl
from parsing.tritonbench_grammar_rules import validate_tritonbench_candidate


EXPECTED  = 166


def validate(file_path: Path) -> bool:
    if not file_path.exists():
        print(f"ERROR: archivo no encontrado: {file_path}")
        return False

    try:
        records = load_prediction_jsonl(file_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FALLO — {file_path}")
        print(f"  No se pudo cargar: {type(exc).__name__}: {exc}")
        return False

    issues: list[str] = []
    if len(records) != EXPECTED:
        issues.append(f"  Longitud: {len(records)} entradas (esperaba {EXPECTED})")

    for lineno, record in enumerate(records, start=1):
        task_id = record["task"]["id"]
        expected_task_id = f"tritonbench_t_{lineno:03d}"
        if task_id != expected_task_id:
            issues.append(
                f"  línea {lineno}: task_id={task_id!r}, esperaba {expected_task_id!r}"
            )

        code = extract_code(record["output"])
        if not code:
            issues.append(f"  línea {lineno}: output vacío")
            continue
        if "```" in code:
            issues.append(f"  línea {lineno}: output contiene fences")

        try:
            ast.parse(code)
        except SyntaxError as exc:
            issues.append(f"  línea {lineno}: sintaxis inválida: {exc.msg}")
            continue

        safety = validate_generated_code_safety(code)
        for error in safety.errors:
            issues.append(f"  línea {lineno}: {error}")

        contract = validate_tritonbench_candidate(task_id, code)
        for error in contract.errors:
            issues.append(f"  línea {lineno}: {error}")
        for warning in contract.warnings:
            if "TODO/pass" in warning:
                issues.append(f"  línea {lineno}: {warning}")

    if issues:
        print(f"FALLO — {file_path}")
        print(f"  {len(issues)} problema(s) encontrado(s):")
        for issue in issues:
            print(issue)
        return False

    print(f"OK — {file_path}")
    print(f"  {len(records)} entradas con sintaxis, seguridad y contrato válidos")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validar archivo de predicciones para TritonBench4Modal")
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Ruta al archivo .jsonl a validar",
    )
    args = parser.parse_args()

    ok = validate(args.file)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
