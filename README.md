# GPU AI Optimizer

Sistema prototipo para clasificación, análisis estructurado y generación automática de kernels GPU optimizados utilizando Triton + XGrammar + Small Language Models.

## Objetivo
Automatizar la optimización de código GPU para tareas comunes:
- operaciones elementales (element-wise)
- reducciones
- multiplicación de matrices
- generación de kernels Triton adaptados al tipo de problema
- wrappers Python para lanzar kernels Triton sin escribir boilerplate de grid, salida y validación

## Estructura del proyecto
- `main.py`: pipeline principal para clasificar, parsear, convertir a XGrammar y generar código Triton.
- `models/classifier.py`: analiza el código y clasifica el tipo de problema.
- `parsing/ast_parser.py`: convierte el código en una representación AST estructurada.
- `parsing/xgrammar_converter.py`: transforma la AST en una representación estilo XGrammar.
- `generation/triton_generator.py`: selecciona el template Triton adecuado y genera kernel + wrapper.
- `generation/kernel_templates.py`: templates de kernels y wrappers para operaciones element-wise, reducciones, matriz y genérico.
- `benchmarks/`: utilidades para medir tiempos y perfilado.
- `docs/presentation_outline.md`: esquema de presentación para explicar el proyecto.

## Instalación

Entorno mínimo para ejecutar CI y las pruebas sin GPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ci.txt
pytest -q
```

Entorno completo para generación con modelos y validación CUDA:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Uso
```bash
python main.py --code "C = A + B"
```

También puedes cambiar el ejemplo de entrada:
```bash
python main.py --code "output = torch.sum(x)"
python main.py --code "C = A @ B"
```

También puedes guardar en otra ruta y lanzar benchmarks después de la generación:
```bash
python main.py --code "C = A + B" --output custom_results.txt --benchmark
```

Para validar el wrapper generado en GPU y compararlo contra una referencia PyTorch:
```bash
python main.py --code "C = A + B" --validate-gpu
python main.py --code "output = torch.sum(x)" --validate-gpu
python main.py --code "C = A @ B" --validate-gpu
```

Si falta `torch`, `triton` o una GPU CUDA, la validación responde con `status=not_run` en lugar de fallar.

## Funcionamiento
1. Clasificación del problema
2. Transformación del código a AST
3. Conversión de la AST a XGrammar
4. Generación de un kernel Triton optimizado según el tipo de problema
5. Inclusión de un wrapper `launch_*` que valida tensores CUDA, reserva la salida, calcula el grid y ejecuta el kernel
6. Validación opcional en GPU contra una referencia PyTorch

## Pruebas
```bash
pytest
```

Las pruebas funcionales GPU están marcadas como `gpu` y se saltan automáticamente si no hay `torch`, `triton` o CUDA:
```bash
pytest -m gpu
```

## Benchmarks
Genera benchmarks reproducibles de tiempo de generación:
```bash
python benchmarks/run_benchmarks.py --runs 5
```

Esto escribe `benchmarks/results.csv`. Para generar el resumen gráfico:
```bash
python benchmarks/plot_results.py
```

Esto escribe `benchmarks/results_summary.png`.

## Evaluación de modelos para generación de kernels

Las predicciones usan un esquema JSONL canónico con estos campos:

```text
schema_version
record_id
model
task
mode
sample_index
prompt
output
latency_seconds
constrained_decoding_backend
```

Los archivos históricos con `{instruction, predict}` siguen siendo compatibles:
se normalizan automáticamente al cargarlos.

Para reevaluar las 498 predicciones baseline actuales:

```bash
python evaluation/model_evaluation.py \
  --output-dir artifacts/current_evaluation \
  --manual-outputs evaluation/predictions_qwen.jsonl \
                   evaluation/predictions_gpt4o.jsonl \
                   evaluation/predictions_claude.jsonl
```

Esto genera:

- `artifacts/current_evaluation/generated_kernels.jsonl`
- `artifacts/current_evaluation/model_eval_results.csv`
- `artifacts/current_evaluation/model_eval_summary.md`

La evaluación estática separa:

- sintaxis Python válida;
- política de seguridad válida;
- contrato TritonBench válido;
- proxy estructural de corrección.

Ninguna de estas métricas sustituye compilación o equivalencia numérica GPU.

Para validar los contratos individuales y familiares:

```bash
python benchmarks/benchmark_tritonbench_grammar.py \
  --input artifacts/current_evaluation/generated_kernels.jsonl \
  --output artifacts/current_evaluation/tritonbench_grammar_results.csv
```

El comando falla si la entrada no existe o está vacía.

## Ejecución segura de kernels generados

`benchmarks/run_generated_kernels.py` no ejecuta directamente respuestas de
modelos dentro del proceso coordinador. Cada kernel se evalúa en un subprocess
temporal con:

- validación AST previa;
- lista restringida de imports;
- entorno sin API keys;
- directorio de trabajo temporal;
- timeout configurable;
- descarte completo del proceso después de cada kernel.

Ejemplo:

```bash
python benchmarks/run_generated_kernels.py \
  --input artifacts/current_evaluation/generated_kernels.jsonl \
  --worker-timeout 120
```

Este benchmark requiere CUDA. El speedup solo se registra si la salida es
numéricamente correcta.

## Generación baseline

```bash
python evaluation/generate_baseline_predictions.py --provider qwen
python evaluation/generate_baseline_predictions.py --provider gpt4o
python evaluation/generate_baseline_predictions.py --provider claude
```

`--resume` carga registros viejos o canónicos, deduplica por `task_id` y
reescribe el archivo en orden. Las API externas requieren sus variables de
entorno correspondientes.

## Perfilado GPU
El perfilado real requiere `torch`, `triton` y una GPU CUDA disponible:
```bash
python -c "from benchmarks.profiler import profile_gpu; print(profile_gpu())"
```

Si falta alguna dependencia o no hay GPU, el profiler responde con `gpu_available=False` y `status=not_run` en lugar de fallar.

## Integración LLM
La decisión de estrategia vive en `models/llm_decider.py`. Hoy usa heurísticas deterministas y la configuración de `config/settings.yaml`.

Configuración actual:
```yaml
generation:
  optimization_level: basic
```

La ruta principal conserva una heurística determinista para que `main.py`
funcione sin descargar un modelo. Los flujos experimentales admiten Ollama,
OpenAI, Anthropic y HuggingFace + XGrammar.

## CI
El workflow de GitHub Actions en `.github/workflows/ci.yml` instala dependencias y ejecuta:
```bash
pytest -q
```

Los pasos GPU se mantienen fuera de CI para que las pruebas funcionen también en runners sin acelerador.

CI instala `requirements-ci.txt`, ejecuta un smoke test de `main.py` y después
corre la suite completa.

## Estado experimental actual

Los artefactos actuales contienen 166 predicciones baseline por modelo:

- Qwen2.5-Coder-1.5B;
- GPT-4o;
- Claude Haiku 4.5.

No existe todavía una corrida constrained completa y comparable en los
artefactos versionados. El reporte actual no afirma mejoras
baseline-vs-constrained.

El reporte reproducible vive en:

```text
reports/reporte_estadistico/reporte_borrador.md
```

Se regenera con:

```bash
python reports/reporte_estadistico/scripts/build_current_report.py
```
