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
import json
import sys
from pathlib import Path

ROOT      = Path(__file__).parent.parent
SIMP_PATH = ROOT / "extras/TritonBench-main/data/TritonBench_T_simp_alpac_v1.json"
EXPECTED  = 166


def validate(file_path: Path) -> bool:
    if not file_path.exists():
        print(f"ERROR: archivo no encontrado: {file_path}")
        return False

    # Cargar dataset de referencia
    with open(SIMP_PATH, encoding="utf-8") as f:
        simp_data = json.load(f)

    lines = []
    parse_errors = []

    with open(file_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
                lines.append((lineno, entry))
            except json.JSONDecodeError as e:
                parse_errors.append(f"  línea {lineno}: JSON inválido — {e}")

    issues = []
    issues.extend(parse_errors)

    if len(lines) != EXPECTED:
        issues.append(f"  Longitud: {len(lines)} entradas (esperaba {EXPECTED})")

    for idx, (lineno, entry) in enumerate(lines):
        # Verificar campos requeridos
        if "instruction" not in entry:
            issues.append(f"  línea {lineno}: falta campo 'instruction'")
            continue
        if "predict" not in entry:
            issues.append(f"  línea {lineno}: falta campo 'predict'")
            continue

        instr   = entry["instruction"]
        predict = entry["predict"]

        # Verificar que instruction coincide con la referencia por posición
        if idx < len(simp_data):
            expected_instr = simp_data[idx]["instruction"]
            if instr != expected_instr:
                issues.append(
                    f"  línea {lineno} [idx={idx}]: instruction no coincide con referencia\n"
                    f"    got[:60]:      {instr[:60]!r}\n"
                    f"    expected[:60]: {expected_instr[:60]!r}"
                )

        # Verificar que predict no está vacío
        if not predict or not predict.strip():
            issues.append(f"  línea {lineno}: 'predict' está vacío")

        # Verificar que predict no contiene fences
        if "```" in predict:
            issues.append(f"  línea {lineno}: 'predict' contiene fences ```")

    if issues:
        print(f"FALLO — {file_path}")
        print(f"  {len(issues)} problema(s) encontrado(s):")
        for issue in issues:
            print(issue)
        return False

    print(f"OK — {file_path}")
    print(f"  {len(lines)} entradas válidas, sin fences, instructions alineadas")
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
