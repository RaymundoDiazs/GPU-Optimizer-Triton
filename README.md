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
```bash
pip install -r requirements.txt
bash setup.sh
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
Para preparar el segundo video de avance, el repo incluye un runner que compara un modelo pequeño contra dos modelos frontier en modo baseline y constrained:

```bash
python evaluation/model_evaluation.py --samples 150
```

Esto genera:
- `evaluation/artifacts/generated_kernels.jsonl`
- `evaluation/artifacts/model_eval_results.csv`
- `evaluation/artifacts/model_eval_summary.md`

También se puede reparar los kernels generados y aplicar correcciones sintácticas con:

```bash
python evaluation/repair_generated_kernels.py
```

Los modelos por defecto en `config/model_eval.yaml` usan `provider: mock` para que el flujo sea reproducible sin API keys. Antes de presentar resultados reales, reemplaza esos outputs por corridas reales del modelo pequeño y los dos modelos frontier elegidos. El plan del video está en `docs/second_video_plan.md`.

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

La interfaz ya está preparada para reemplazarse por un modelo real. Para una integración futura basada en API externa, usa una variable de entorno como `LLM_API_KEY` y conserva la firma pública `decide_strategy(code, ast_repr, grammar)`.

## CI
El workflow de GitHub Actions en `.github/workflows/ci.yml` instala dependencias y ejecuta:
```bash
pytest -q
```

Los pasos GPU se mantienen fuera de CI para que las pruebas funcionen también en runners sin acelerador.
