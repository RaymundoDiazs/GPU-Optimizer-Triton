# context.md — Contexto para próxima conversación
## Tarea inmediata: generación de predictions para TritonBench4Modal

---

## Lo que hay que hacer ahora

Adaptar el pipeline de generación de Dilan para producir los archivos que necesita TritonBench4Modal.

**Entregables concretos:**
1. `evaluation/predictions_qwen.jsonl` — 166 líneas, generado con Ollama local
2. `evaluation/predictions_gpt4o.jsonl` — generado con `modal_app.py::main` directamente
3. `evaluation/predictions_claude.jsonl` — generado con `modal_app.py::main` directamente

---

## Formato exacto requerido

Cada línea del JSONL debe ser:
```json
{"instruction": "<texto IDÉNTICO al campo instruction del Alpaca dataset>", "predict": "```python\n<código generado>\n```"}
```

**CRÍTICO:** El `instruction` debe ser copia exacta del dataset — el evaluador lo usa para encontrar el test driver (busca substring entre `"Functional Description: "` y `"Wrapper Entry Information:"`).

---

## Dos flujos distintos

### GPT-4o y Claude Haiku → usar modal_app.py directamente
`modal_app.py` ya tiene generación integrada para Anthropic y OpenAI. No hay que tocar nada — solo correrlo:
```bash
# Desde extras/TritonBench4Modal-main/
modal run modal_app.py::main --provider openai --model gpt-4o
modal run modal_app.py::main --provider anthropic --model claude-haiku-4-5-20251001
```
Esto genera Y evalúa en una sola corrida. Los resultados quedan en el Modal Volume.

⚠️ **`extras/TritonBench4Modal-main/` NO SE MODIFICA NUNCA** — es la herramienta estándar de toda la escuela. Cualquier cambio rompe la comparabilidad con otros equipos.

### Qwen → generar localmente, evaluar con evaluate_only
Modal no puede llamar a Ollama local. Por eso:
1. Generar con `evaluation/collect_real_outputs.py` (Ollama)
2. Producir `evaluation/predictions_qwen.jsonl`
3. German corre: `modal run modal_app.py::evaluate_only --predictions ./predictions_qwen.jsonl`

---

## Qué hay que modificar en collect_real_outputs.py

El script actual lee `data/pytorch_examples.json` (6 ejemplos propios) y produce un formato interno.

Hay que agregar un modo nuevo que:
1. Lea `TritonBench_T_simp_alpac_v1.json` — cada entrada tiene `instruction` y `input` (puede estar vacío)
2. Construya el mensaje al modelo: `instruction` + `input` si no está vacío
3. Extraiga el bloque `\`\`\`python ... \`\`\`` de la respuesta
4. Guarde en formato exacto: `{"instruction": "<original>", "predict": "<código con fences>"}`
5. Output: `evaluation/predictions_qwen.jsonl`

El prompt del sistema que usa modal_app.py (para ser consistentes):
```
You are an expert in Triton programming, capable of writing Triton kernels
and wrapper functions based on functional descriptions and function
parameters. The wrapper function must fully match the provided function
signature.

Output a single, self-contained Python module containing: (a) the necessary
imports (torch, triton, triton.language as tl), (b) the Triton kernel(s),
and (c) the wrapper function that the description specifies. Wrap the
entire module in one ```python ... ``` fenced code block. Do NOT include
any test code or example calls — tests will be appended separately.
```

---

## Parámetros de generación

| Parámetro | Valor |
|---|---|
| temperature | 0.15 |
| top_p | 0.95 |
| seed | 42 |
| samples_per_operator | 1 |
| max_tokens | 8192 |

---

## Dataset

`TritonBench_T_simp_alpac_v1.json` vive dentro del contenedor Modal en `/opt/TritonBench/data/`.

Para generación local con Qwen hay que descargarlo del repo upstream:
```
https://github.com/thunlp/TritonBench — carpeta data/
```

O usar el comando de Modal para obtener solo el dataset:
```bash
modal run modal_app.py::generate_only --limit 0  # genera 0 kernels, solo sirve para ver las instructions
```

---

## Archivos relevantes del proyecto

```
evaluation/collect_real_outputs.py   ← MODIFICAR — agregar modo Alpaca para Qwen
evaluation/predictions_qwen.jsonl    ← CREAR — output del script modificado
config/model_eval.yaml               ← configuración de modelos
prompts/pytorch_to_triton_prompt.txt ← revisar si es compatible o usar el prompt de modal_app.py
extras/TritonBench4Modal-main/modal_app.py  ← NO TOCAR — referencia para el prompt del sistema
```

---

## Modelos configurados

| Modelo | Provider | Cómo se genera |
|---|---|---|
| Qwen2.5-Coder-1.5B | Ollama local (gratis) | `collect_real_outputs.py` modificado |
| GPT-4o | OpenAI API | `modal_app.py::main --provider openai` |
| Claude Haiku 4.5 | Anthropic API | `modal_app.py::main --provider anthropic` |

Costos estimados para 166 operadores:
- GPT-4o: ~$2-5
- Claude Haiku: ~$0.50-1
- Modal GPU (evaluación): ~$2-3 por run
- Qwen: $0

---

## Lo que NO hay que hacer

- No modificar `extras/TritonBench4Modal-main/` bajo ninguna circunstancia
- No inventar un formato propio de JSONL — usar exactamente el que espera `evaluate_only`
- No generar múltiples muestras por operador — 1 es suficiente para call@1/exe@1/speedup
- No preocuparse por constrained decoding ahora — eso es trabajo de Charlie después

---

## Contexto del proyecto completo

Ver `CLAUDE.md` en la raíz del proyecto para el contexto completo: tareas de cada persona, estado actual, flujo de datos, y reglas del benchmark.
