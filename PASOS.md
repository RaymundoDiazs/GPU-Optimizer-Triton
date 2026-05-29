# PASOS.md — Registro de implementación

Registro resumido de cada paso completado, con status y lo que se hizo.

---

## PASO 1 — Descargar dataset `TritonBench_T_simp_alpac_v1.json`
**Status: COMPLETADO**

- Descargado desde `thunlp/TritonBench` en `data/TritonBench_T_simp_alpac_v1.json`
- Verificado: 166 operadores, formato correcto con campos `instruction`, `input`, `output`

---

## PASO 2 — Crear `evaluation/generate_tritonbench_predictions.py`
**Status: COMPLETADO**

- Creado `evaluation/generate_tritonbench_predictions.py`
- Lee `data/TritonBench_T_simp_alpac_v1.json` (166 operadores, formato Alpaca)
- Usa `prompts/pytorch_to_triton_prompt.txt` como system prompt (placeholder `{instruction}` confirmado)
- Llama a Qwen 2.5-Coder 1.5B vía Ollama (temperatura 0.15, top_p 0.95, seed 42, num_predict 8192)
- `extract_code_clean()` elimina fences antes de guardar — evita SyntaxError en call@1
- Soporta `--resume` para reanudar si se interrumpe, sin perder progreso
- Guarda en `evaluation/predictions_qwen.jsonl` con formato exacto: `{"instruction": "...", "predict": "..."}`
- Verificado: prompt tiene `{instruction}`, dataset tiene 166 entradas, campo `input` está vacío

**Por qué script nuevo y no modificar `collect_real_outputs.py`:**
El script existente guarda campos extra (`model`, `task`, `mode`, `sample_index`). `evaluate_only` espera exactamente `{"instruction", "predict"}`. Modificarlo rompería los experimentos internos ya corridos.

---

## PASO 3 — Actualizar `config/model_eval.yaml`
**Status: COMPLETADO**

- Reemplazados los parámetros viejos (`samples_per_model: 20`, `seed: 7`, dataset de ejemplos propios) por los del benchmark estándar
- Valores finales: `samples_per_operator: 1`, `seed: 42`, `temperature: 0.15`, `top_p: 0.95`, `max_tokens: 8192`, `dataset: data/TritonBench_T_simp_alpac_v1.json`

**Para qué sirve y por qué es necesario:**
Este archivo es la fuente de verdad de los parámetros de generación del proyecto. Los valores anteriores eran de los experimentos iniciales propios (20 muestras por modelo, seed 7, dataset de 6 ejemplos). El benchmark requiere exactamente 1 muestra por operador con los parámetros del notebook de clase (seed 42, temp 0.15) para que los resultados sean reproducibles y comparables con los demás equipos. Sin esta actualización, cualquier script que lea el YAML usaría configuración incorrecta.

---

## PASO 4 — Generar `predictions_qwen.jsonl`
**Status: COMPLETADO**

- Corrido `evaluation/generate_tritonbench_predictions.py` contra los 166 operadores de TritonBench-T
- 166/166 generados sin ningún FALLO — todos respondieron OK
- Archivo resultante: `evaluation/predictions_qwen.jsonl`, 166 líneas, formato `{"instruction": "...", "predict": "..."}`

**Para qué sirve y por qué es necesario:**
Este es el entregable principal de Dilan para German. Contiene los 166 kernels Triton que Qwen generó a partir de las instrucciones del benchmark. German lo pasa a `modal_app.py::evaluate_only` para correr las 3 fases de evaluación en GPU real (call@1, exe@1, speedup). Sin este archivo no hay nada que evaluar.

---

## PASO 5 — Configurar Modal
**Status: COMPLETADO**

- Modal v1.4.3 instalado y autenticado (cuenta: deoh02)
- Secret `tritonbench-llm` creado con las API keys de OpenAI y Anthropic

**Para qué sirve y por qué es necesario:**
Modal es la plataforma GPU donde se evalúan los kernels. El secret `tritonbench-llm` es el nombre exacto que `modal_app.py` busca para obtener las API keys al generar con GPT-4o y Claude Haiku. Sin cuenta autenticada y sin ese secret, ninguna corrida de Modal funciona.

---
## PASO 6 — Crear `scripts/run_modal.py`
**Status: COMPLETADO**

- Creado `scripts/run_modal.py` (carpeta `scripts/` creada también)
- Soporta dos modos: `--evaluate-only --predictions <archivo>` para Qwen, y `--provider --model` para GPT-4o y Claude
- Corre Modal como subproceso, muestra logs en tiempo real, extrae el JSON final y lo guarda en `results/` automáticamente
- Si no encuentra JSON válido en el output, guarda el output completo como `.txt` para inspección manual

**Para qué sirve y por qué es necesario:**
`modal run` mezcla logs de progreso con el JSON de resultados en stdout — sin este helper habría que copiar el JSON manualmente cada vez. El script automatiza eso: extrae solo el bloque JSON y lo guarda con el nombre correcto en `results/`. También centraliza los comandos de Modal para que no haya que recordar la sintaxis exacta cada vez.

---
## PASO 7 — Pendiente
## PASO 8 — Pendiente
## PASO 9 — Pendiente
