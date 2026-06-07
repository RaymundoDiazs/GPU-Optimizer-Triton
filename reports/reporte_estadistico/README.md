# Reporte Estadistico

Esta carpeta contiene el entregable de evaluacion experimental del proyecto.

El objetivo es separar el reporte estadistico del codigo principal para evitar confusiones entre:

- implementacion del sistema,
- datos experimentales,
- analisis estadistico,
- borrador del reporte.

## Archivos principales

- `preregistro.md`: Parte I del reporte. Define hipotesis, variables, metricas, diseno experimental, plan estadistico y amenazas antes de ejecutar el experimento final.
- `reporte_borrador.md`: Parte II del reporte. Resume los resultados preliminares actuales y deja la estructura lista para completar con mas muestras.
- `experiment_config.json`: configuracion congelada del experimento planeado.
- `reproducibilidad.md`: plantilla para documentar hardware, software, modelos y pasos exactos.
- `plan_recoleccion_final.md`: plan para obtener suficientes muestras para el reporte final.
- `tritonbench_plan.md`: plan para alinear la evaluacion final con el notebook de clase y TritonBench4Modal.
- `scripts/analyze_results.py`: script para analizar resultados CSV y generar tablas/graficas.
- `scripts/capture_environment.py`: script para capturar versiones de entorno.
- `results/`: tablas generadas por el analisis.
- `figures/`: graficas generadas por el analisis.

## Estado actual

Los artefactos actuales contienen 498 predicciones baseline:

```text
3 modelos x 166 operadores = 498 outputs
```

No existe todavía una corrida constrained completa con el mismo dataset y
hardware. Por ello no se deben presentar comparaciones baseline-vs-constrained
como resultados actuales.

## Conexion con TritonBench

La evaluacion actual mide forma y consistencia estructural de las salidas. Para el reporte final, esa capa debe conectarse con TritonBench/TritonBench4Modal para medir:

- call accuracy: si el codigo generado se puede importar y llamar con la firma esperada,
- execution accuracy: si produce el mismo resultado que la referencia PyTorch,
- speedup: si el kernel Triton es mas rapido que la version PyTorch de referencia.

El plan concreto esta en `tritonbench_plan.md`.

## Como regenerar el analisis

```bash
python reports/reporte_estadistico/scripts/analyze_results.py
python reports/reporte_estadistico/scripts/capture_environment.py
```

La evaluación actual se regenera primero con:

```bash
python evaluation/model_evaluation.py \
  --output-dir artifacts/current_evaluation \
  --manual-outputs evaluation/predictions_qwen.jsonl \
                   evaluation/predictions_gpt4o.jsonl \
                   evaluation/predictions_claude.jsonl
```

El reporte baseline actual se genera con:

```bash
python reports/reporte_estadistico/scripts/build_current_report.py
```

Los análisis anteriores basados en `evaluation/artifacts/model_eval_results.csv`
se conservan como historial, pero no son la fuente del reporte actual.
