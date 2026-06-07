from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STATIC_RESULTS = ROOT / "artifacts" / "current_evaluation" / "model_eval_results.csv"
GPU_RESULTS = ROOT / "results"
REPORT = ROOT / "reports" / "reporte_estadistico" / "reporte_borrador.md"
SUMMARY = ROOT / "reports" / "reporte_estadistico" / "results" / "current_baseline_summary.csv"

MODEL_ORDER = [
    "small_qwen25_coder_1_5b",
    "frontier_openai",
    "frontier_anthropic",
]
GPU_FILES = {
    "small_qwen25_coder_1_5b": "tritonbench_qwen_baseline.json",
    "frontier_openai": "tritonbench_gpt4o_baseline.json",
    "frontier_anthropic": "tritonbench_claude_baseline.json",
}


def _rate(rows: list[dict[str, str]], key: str) -> float:
    return sum(row[key] == "True" for row in rows) / len(rows) if rows else 0.0


def build_rows() -> list[dict[str, object]]:
    with STATIC_RESULTS.open("r", encoding="utf-8") as file:
        static_rows = list(csv.DictReader(file))

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in static_rows:
        groups[row["model_id"]].append(row)

    rows = []
    for model_id in MODEL_ORDER:
        group = groups[model_id]
        gpu = json.loads((GPU_RESULTS / GPU_FILES[model_id]).read_text(encoding="utf-8"))
        rows.append(
            {
                "model_id": model_id,
                "n": len(group),
                "syntax_valid_rate": _rate(group, "syntax_valid"),
                "safety_valid_rate": _rate(group, "safety_valid"),
                "contract_valid_rate": _rate(group, "contract_valid"),
                "call_accuracy_rate": gpu["phase1_call_acc"]["rate"] / 100,
                "execution_accuracy_rate": gpu["phase2_exec_acc"]["rate"] / 100,
                "archived_speedup": gpu["phase3_efficiency"]["speedup_vs_pytorch"],
                "speedups_recorded": len(gpu["phase3_efficiency"]["per_kernel"]),
            }
        )
    return rows


def write_summary(rows: list[dict[str, object]]) -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: object) -> str:
    return f"{float(value):.1%}"


def write_report(rows: list[dict[str, object]]) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        "# Reporte de evaluación actual",
        "",
        f"Generado el {generated} desde los artefactos versionados del repositorio.",
        "",
        "## Alcance",
        "",
        "Este reporte distingue dos capas que no deben confundirse:",
        "",
        "1. Validación estática actual sobre 166 predicciones baseline por modelo.",
        "2. Métricas GPU baseline previamente producidas por TritonBench4Modal.",
        "",
        "No hay una corrida constrained actual y comparable en los artefactos disponibles.",
        "Por lo tanto, este reporte no afirma mejoras baseline-vs-constrained.",
        "",
        "## Resultados baseline",
        "",
        "| Modelo | n | Sintaxis | Seguridad | Contrato | Call accuracy GPU | Execution accuracy GPU | Speedup archivado |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_id']} | {row['n']} | {_pct(row['syntax_valid_rate'])} | "
            f"{_pct(row['safety_valid_rate'])} | {_pct(row['contract_valid_rate'])} | "
            f"{_pct(row['call_accuracy_rate'])} | {_pct(row['execution_accuracy_rate'])} | "
            f"{float(row['archived_speedup']):.2f}x |"
        )

    lines.extend(
        [
            "",
            "## Interpretación",
            "",
            "- Sintaxis válida no implica que un kernel pueda compilarse, llamarse o producir resultados correctos.",
            "- La validación de contrato comprueba firma y evidencia estructural específica de cada tarea.",
            "- `call_accuracy` y `execution_accuracy` provienen de los JSON baseline existentes en `results/`.",
            "- Los speedups son valores archivados. Deben volver a medirse con el evaluador aislado antes de usarlos como resultado final.",
            "- El evaluador actualizado solo registra speedup cuando el kernel es numéricamente correcto.",
            "",
            "## Calidad de los datos",
            "",
            "La evaluación estática encontró respuestas con errores de sintaxis, imports no permitidos, marcadores `pass` y contratos incompletos.",
            "Los archivos JSONL tienen 166 registros alineados por modelo, pero estar bien formados no demuestra calidad funcional.",
            "",
            "## Reproducibilidad",
            "",
            "Regenerar la evaluación estática:",
            "",
            "```bash",
            "python evaluation/model_evaluation.py \\",
            "  --output-dir artifacts/current_evaluation \\",
            "  --manual-outputs evaluation/predictions_qwen.jsonl \\",
            "                   evaluation/predictions_gpt4o.jsonl \\",
            "                   evaluation/predictions_claude.jsonl",
            "python benchmarks/benchmark_tritonbench_grammar.py \\",
            "  --input artifacts/current_evaluation/generated_kernels.jsonl \\",
            "  --output artifacts/current_evaluation/tritonbench_grammar_results.csv",
            "python reports/reporte_estadistico/scripts/build_current_report.py",
            "```",
            "",
            "La ejecución GPU debe hacerse en un host CUDA. Cada kernel se ejecuta en un subprocess temporal con timeout y entorno sin secretos.",
            "",
            "## Trabajo pendiente",
            "",
            "1. Repetir call accuracy, execution accuracy y eficiencia con el worker aislado.",
            "2. Generar una corrida constrained completa con el mismo dataset, hardware y parámetros.",
            "3. Comparar baseline-vs-constrained únicamente después de completar ambos grupos.",
            "4. Reportar distribución de speedups de kernels correctos, no solo un promedio agregado.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_summary(rows)
    write_report(rows)
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
