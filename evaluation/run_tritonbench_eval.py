"""
Wrapper para correr evaluate_only de TritonBench4Modal y guardar resultados.

Uso:
    python evaluation/run_tritonbench_eval.py --provider qwen
    python evaluation/run_tritonbench_eval.py --provider gpt4o
    python evaluation/run_tritonbench_eval.py --provider claude

Requiere Modal configurado (modal setup / modal profile list).
No modifica extras/TritonBench4Modal-main/.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT        = Path(__file__).parent.parent
EVAL_DIR    = ROOT / "evaluation"
RESULTS_DIR = ROOT / "results"
MODAL_DIR   = ROOT / "extras/TritonBench4Modal-main"

PROVIDERS = {
    "qwen":   {"model": "qwen2.5-coder:1.5b",        "predictions": "predictions_qwen.jsonl"},
    "gpt4o":  {"model": "gpt-4o",                    "predictions": "predictions_gpt4o.jsonl"},
    "claude": {"model": "claude-haiku-4-5-20251001",  "predictions": "predictions_claude.jsonl"},
}


def run_eval(provider: str):
    cfg         = PROVIDERS[provider]
    preds_path  = EVAL_DIR / cfg["predictions"]
    output_file = RESULTS_DIR / f"tritonbench_{provider}_baseline.json"
    subdir      = f"results_{provider}"

    if not preds_path.exists():
        print(f"ERROR: no existe {preds_path}", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"=== Evaluando {provider} ({cfg['model']}) ===")
    print(f"  Predictions: {preds_path}")
    print(f"  Output:      {output_file}")
    print(f"  Modal subdir: {subdir}")
    print()

    # Correr modal run desde la carpeta de TritonBench4Modal
    cmd = [
        sys.executable, "-m", "modal", "run",
        "modal_app.py::evaluate_only",
        "--predictions", str(preds_path),
        "--output-subdir", subdir,
    ]

    result = subprocess.run(
        cmd,
        cwd=str(MODAL_DIR),
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("[stderr]", result.stderr, file=sys.stderr)

    # Extraer el JSON del output (está al final del stdout)
    summary = None
    lines = result.stdout.strip().splitlines()
    # Buscar el bloque JSON desde el final
    json_lines = []
    in_json = False
    for line in lines:
        if line.strip().startswith("{"):
            in_json = True
        if in_json:
            json_lines.append(line)

    if json_lines:
        try:
            summary = json.loads("\n".join(json_lines))
        except json.JSONDecodeError:
            print("ADVERTENCIA: no se pudo parsear el JSON del output", file=sys.stderr)

    if summary is None:
        # Guardar el output crudo si no hay JSON parseable
        summary = {"raw_output": result.stdout, "error": "JSON no encontrado en output"}

    # Agregar metadatos
    summary["provider"]  = provider
    summary["model"]     = cfg["model"]
    summary["dataset"]   = "TritonBench_T_simp_alpac_v1 + torch_code"
    summary["timestamp"] = datetime.now().isoformat()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados en: {output_file}")
    print(json.dumps(summary, indent=2))

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Correr evaluación TritonBench4Modal y guardar resultados")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        required=True,
        help="qwen | gpt4o | claude",
    )
    args = parser.parse_args()
    sys.exit(run_eval(args.provider))


if __name__ == "__main__":
    main()
