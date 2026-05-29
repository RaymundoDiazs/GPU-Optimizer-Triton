"""
scripts/run_modal.py
---------------------
Corre modal_app.py y guarda el summary JSON en results/ automáticamente.

Uso:
    python scripts/run_modal.py --provider openai --model gpt-4o
    python scripts/run_modal.py --provider anthropic --model claude-haiku-4-5-20251001
    python scripts/run_modal.py --evaluate-only --predictions evaluation/predictions_qwen.jsonl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[1]
MODAL_DIR   = REPO_ROOT / "extras" / "TritonBench4Modal-main"
RESULTS_DIR = REPO_ROOT / "results"


def extract_json(output: str) -> dict:
    """Extrae el último bloque JSON válido del output de modal run."""
    lines = output.strip().splitlines()
    json_lines = []
    in_json = False
    for line in lines:
        if line.strip() == "{":
            in_json = True
            json_lines = []
        if in_json:
            json_lines.append(line)
        if in_json and line.strip() == "}":
            try:
                return json.loads("\n".join(json_lines))
            except json.JSONDecodeError:
                json_lines = []
                in_json = False
    raise ValueError("No se encontró JSON válido en el output de Modal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "anthropic"])
    parser.add_argument("--model")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--predictions")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.evaluate_only:
        assert args.predictions, "--predictions requerido con --evaluate-only"
        predictions_path = REPO_ROOT / args.predictions
        cmd = [
            "modal", "run", "modal_app.py::evaluate_only",
            "--predictions", str(predictions_path),
        ]
        out_file = RESULTS_DIR / "tritonbench_qwen_baseline.json"
    else:
        assert args.provider and args.model, "--provider y --model requeridos"
        cmd = [
            "modal", "run", "modal_app.py::main",
            "--provider", args.provider,
            "--model", args.model,
        ]
        name = "gpt4o" if "gpt" in args.model else "claude"
        out_file = RESULTS_DIR / f"tritonbench_{name}_baseline.json"

    print(f"Corriendo: {' '.join(cmd)}")
    print(f"Guardando resultado en: {out_file}\n")

    result = subprocess.run(
        cmd, cwd=str(MODAL_DIR),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )

    print(result.stdout)

    try:
        summary = extract_json(result.stdout)
        out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nResultado guardado en {out_file}")
    except ValueError as e:
        print(f"\nERROR extrayendo JSON: {e}", file=sys.stderr)
        print("Output completo guardado para inspección manual.")
        out_file.with_suffix(".txt").write_text(result.stdout, encoding="utf-8")


if __name__ == "__main__":
    main()
