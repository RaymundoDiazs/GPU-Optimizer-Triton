# Plan de Recoleccion Final

## Objetivo

Generar suficientes datos para que el reporte estadistico pueda sostener conclusiones mas fuertes que el analisis preliminar actual.

## Tamano de muestra recomendado

Minimo:

```text
30 muestras por modelo/modo
```

Ideal:

```text
50 muestras por modelo/modo
```

Con 3 modelos y 2 modos:

```text
30 muestras -> 180 outputs
50 muestras -> 300 outputs
```

## Estrategia si hay limites de costo

Opcion balanceada:

```text
Qwen local: 50 baseline + 50 constrained
GPT-4o: 20 baseline + 20 constrained
Claude: 20 baseline + 20 constrained
```

Esta opcion reduce costo, pero debe reportarse como muestra desbalanceada.

## Comandos esperados

Actualizar primero:

```text
config/model_eval.yaml
```

con:

```yaml
experiment:
  samples_per_model: 30
```

Luego correr:

```bash
python evaluation/collect_real_outputs.py --provider ollama
python evaluation/collect_real_outputs.py --provider openai
python evaluation/collect_real_outputs.py --provider anthropic
python evaluation/model_evaluation.py --manual-outputs evaluation/real_outputs.jsonl
python reports/reporte_estadistico/scripts/analyze_results.py
```

## Datos a guardar

- `evaluation/real_outputs.jsonl`
- `evaluation/artifacts/model_eval_results.csv`
- `reports/reporte_estadistico/results/summary_by_model_mode.csv`
- `reports/reporte_estadistico/results/pairwise_proportion_tests.csv`
- figuras generadas en `reports/reporte_estadistico/figures/`

## Checklist antes de correr

- Confirmar commit de Git.
- Confirmar API keys.
- Confirmar modelo local descargado en Ollama.
- Confirmar temperatura.
- Confirmar prompt.
- Confirmar que no se mezclen outputs de pruebas anteriores.

