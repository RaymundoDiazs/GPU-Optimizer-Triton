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

Los resultados actuales son preliminares. Usan 18 muestras:

```text
3 modelos x 2 modos x 3 muestras = 18 outputs
```

Esto sirve para validar el pipeline y preparar el reporte, pero no es suficiente para conclusiones estadisticas fuertes.

Para el reporte final se recomienda aumentar a minimo 30 muestras por modelo/modo, idealmente 50.

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

El script lee:

```text
evaluation/artifacts/model_eval_results.csv
```

y genera:

```text
reports/reporte_estadistico/results/summary_by_model_mode.csv
reports/reporte_estadistico/results/pairwise_proportion_tests.csv
reports/reporte_estadistico/figures/*.png
```
