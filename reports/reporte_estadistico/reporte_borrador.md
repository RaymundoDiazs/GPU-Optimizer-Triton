# Reporte de evaluación actual

Generado el 2026-06-07 desde los artefactos versionados del repositorio.

## Alcance

Este reporte distingue dos capas que no deben confundirse:

1. Validación estática actual sobre 166 predicciones baseline por modelo.
2. Métricas GPU baseline previamente producidas por TritonBench4Modal.

No hay una corrida constrained actual y comparable en los artefactos disponibles.
Por lo tanto, este reporte no afirma mejoras baseline-vs-constrained.

## Resultados baseline

| Modelo | n | Sintaxis | Seguridad | Contrato | Call accuracy GPU | Execution accuracy GPU | Speedup archivado |
|---|---:|---:|---:|---:|---:|---:|---:|
| small_qwen25_coder_1_5b | 166 | 94.0% | 93.4% | 2.4% | 23.5% | 6.6% | 0.80x |
| frontier_openai | 166 | 100.0% | 100.0% | 65.1% | 10.8% | 7.2% | 0.45x |
| frontier_anthropic | 166 | 99.4% | 98.2% | 89.8% | 41.6% | 40.4% | 0.45x |

## Interpretación

- Sintaxis válida no implica que un kernel pueda compilarse, llamarse o producir resultados correctos.
- La validación de contrato comprueba firma y evidencia estructural específica de cada tarea.
- `call_accuracy` y `execution_accuracy` provienen de los JSON baseline existentes en `results/`.
- Los speedups son valores archivados. Deben volver a medirse con el evaluador aislado antes de usarlos como resultado final.
- El evaluador actualizado solo registra speedup cuando el kernel es numéricamente correcto.

## Calidad de los datos

La evaluación estática encontró respuestas con errores de sintaxis, imports no permitidos, marcadores `pass` y contratos incompletos.
Los archivos JSONL tienen 166 registros alineados por modelo, pero estar bien formados no demuestra calidad funcional.

## Reproducibilidad

Regenerar la evaluación estática:

```bash
python evaluation/model_evaluation.py \
  --output-dir artifacts/current_evaluation \
  --manual-outputs evaluation/predictions_qwen.jsonl \
                   evaluation/predictions_gpt4o.jsonl \
                   evaluation/predictions_claude.jsonl
python benchmarks/benchmark_tritonbench_grammar.py \
  --input artifacts/current_evaluation/generated_kernels.jsonl \
  --output artifacts/current_evaluation/tritonbench_grammar_results.csv
python reports/reporte_estadistico/scripts/build_current_report.py
```

La ejecución GPU debe hacerse en un host CUDA. Cada kernel se ejecuta en un subprocess temporal con timeout y entorno sin secretos.

## Trabajo pendiente

1. Repetir call accuracy, execution accuracy y eficiencia con el worker aislado.
2. Generar una corrida constrained completa con el mismo dataset, hardware y parámetros.
3. Comparar baseline-vs-constrained únicamente después de completar ambos grupos.
4. Reportar distribución de speedups de kernels correctos, no solo un promedio agregado.
