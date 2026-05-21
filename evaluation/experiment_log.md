# Experiment Log — Kernel Generation Evaluation
**Responsable:** Dylan (Ocampo)  
**Proyecto:** GPU Optimizer Triton  
**Deadline video:** 23 de mayo de 2026

---

## Experimento 1 — Recolección baseline + constrained

**Fecha de ejecución:** 2026-05-20  
**Tarea evaluada:** `vector_add` (Z = X + Y para N elementos)

### Modelos usados

| ID | Nombre | Provider | Versión exacta | Tier |
|---|---|---|---|---|
| `small_qwen25_coder_1_5b` | Qwen2.5-Coder-1.5B | Ollama local | qwen2.5-coder:1.5b | small |
| `frontier_openai` | GPT-4o | OpenAI API | gpt-4o | frontier |
| `frontier_anthropic` | Claude Haiku 4.5 | Anthropic API | claude-haiku-4-5-20251001 | frontier |

### Prompt usado

Archivo base: `prompts/kernel_generation_prompt.txt`  
Sin modificaciones por modelo — mismo prompt para los tres.

Modo baseline: prompt base sin agregar nada adicional.  
Modo constrained: prompt base + bloque de "Constrained decoding contract" al final.

¿Hubo ajustes por modelo? No — se usó el mismo prompt base para los tres modelos sin variaciones.

### Parámetros de generación

| Parámetro | Valor |
|---|---|
| `temperature` | 0.2 |
| `seed` | 7 |
| `samples_per_model` | 3 |
| `max_tokens` (Anthropic) | 1024 |

---

## Observaciones de los outputs

### Qwen2.5-Coder-1.5B (local)
- **Errores frecuentes observados:**
  - No usa `tl.program_id` ni `BLOCK_SIZE` — procesa todos los elementos como un bloque único en lugar de dividir el trabajo en bloques paralelos.
  - No implementa correctamente la expresión `x + y` en el cuerpo del kernel (usa variables con nombres distintos como `x_data + y_data`, que el evaluador no reconoce).
  - En modo constrained, mejora la estructura pero sigue sin pasar `correctness_proxy` por el mismo problema de nombres de variables.
  - Envuelve el output en bloque markdown (` ```python `) — el evaluador lo extrae correctamente.
- ¿Pasó syntax check? 100% (todas las muestras tienen sintaxis Python válida)
- ¿Pasó kernel_shape_valid? 33% en baseline, 100% en constrained (el prompt constrained ayuda a incluir `@triton.jit`, `tl.load`, `tl.store`)
- ¿Pasó correctness_proxy? 0% (nunca incluye la expresión literal `x + y`)
- Latencia promedio: ~4.8s en baseline, ~3.7s en constrained

### GPT-4o (OpenAI)
- **Errores frecuentes observados:** Ninguno — genera kernels correctos en todos los intentos.
- Usa `tl.program_id`, `BLOCK_SIZE: tl.constexpr`, offsets correctos, máscara correcta.
- Incluye `x + y` explícitamente (pasa el correctness_proxy).
- Envuelve el output en bloque markdown — el evaluador lo extrae correctamente.
- ¿Pasó syntax check? 100%
- ¿Pasó kernel_shape_valid? 100%
- ¿Pasó correctness_proxy? 100%
- Latencia promedio: ~2.9s en baseline, ~1.6s en constrained

### Claude Haiku 4.5 (Anthropic)
- **Errores frecuentes observados:** Ninguno — genera kernels correctos en todos los intentos.
- Estructura idéntica a GPT-4o: `program_id`, `BLOCK_SIZE`, offsets, máscara, `x + y`.
- Envuelve el output en bloque markdown — el evaluador lo extrae correctamente.
- ¿Pasó syntax check? 100%
- ¿Pasó kernel_shape_valid? 100%
- ¿Pasó correctness_proxy? 100%
- Latencia promedio: ~4.0s en baseline, ~1.7s en constrained

---

## Resultados de evaluación

| Modelo | Modo | Muestras | Sintaxis válida | Kernel shape válido | Correctness proxy |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | baseline | 3 | 100% | 33% | 0% |
| Qwen2.5-Coder-1.5B | constrained | 3 | 100% | 100% | 0% |
| GPT-4o | baseline | 3 | 100% | 100% | 100% |
| GPT-4o | constrained | 3 | 100% | 100% | 100% |
| Claude Haiku 4.5 | baseline | 3 | 100% | 100% | 100% |
| Claude Haiku 4.5 | constrained | 3 | 100% | 100% | 100% |

---

## Trade-offs observados

| Aspecto | Qwen local | GPT-4o | Claude Haiku 4.5 |
|---|---|---|---|
| Costo por corrida | $0 | ~$0.01–0.03 | ~$0.002–0.01 |
| Latencia baseline (promedio) | ~4.8s | ~2.9s | ~4.0s |
| Latencia constrained (promedio) | ~3.7s | ~1.6s | ~1.7s |
| Calidad subjetiva | Baja — no entiende el patrón de kernel paralelo | Alta — estructura correcta siempre | Alta — estructura correcta siempre |
| Instruction following | Parcial — sigue algunas instrucciones pero ignora el patrón de paralelismo | Excelente | Excelente |
| Privacidad de datos | Total — corre local, nada sale del equipo | Datos van a OpenAI | Datos van a Anthropic |

*Costo estimado basado en tokens aprox. 500 input + 300 output por llamada. Claude Haiku es significativamente más barato que GPT-4o con calidad equivalente en esta tarea.*

---

## Hallazgo principal para el video

**El modelo pequeño (Qwen) falla en correctness_proxy aunque tenga sintaxis válida.**  
El problema no es que genere código inválido — es que no entiende el patrón de kernel paralelo Triton.  
Qwen escribe kernels que parecen correctos superficialmente (tienen `@triton.jit`, `tl.load`, `tl.store`) pero no usan `tl.program_id` ni `BLOCK_SIZE`, que son los mecanismos clave de paralelismo en Triton.

El modo constrained mejora el `kernel_shape_valid` de Qwen (de 33% a 100%) porque el prompt adicional lo guía a incluir los elementos estructurales obligatorios. Sin embargo, no resuelve el problema de fondo: Qwen sigue sin escribir la expresión `x + y` de forma que el evaluador reconozca como correcta.

Los modelos frontier (GPT-4o y Claude Haiku) pasan todos los checks en todos los intentos, tanto en baseline como en constrained.

---

## Comandos ejecutados

```bash
# Instalar dependencias adicionales
pip install openai anthropic requests

# Descargar modelo local
ollama pull qwen2.5-coder:1.5b

# Recolectar outputs reales
python evaluation/collect_real_outputs.py --provider ollama
python evaluation/collect_real_outputs.py --provider openai
python evaluation/collect_real_outputs.py --provider anthropic

# Correr evaluación con datos reales
python evaluation/model_evaluation.py --manual-outputs evaluation/real_outputs.jsonl
```

---

## Artefactos generados

- `evaluation/real_outputs.jsonl` — 18 outputs crudos de los modelos (6 por modelo: 3 baseline + 3 constrained)
- `evaluation/artifacts/generated_kernels.jsonl` — kernels extraídos y evaluados
- `evaluation/artifacts/model_eval_results.csv` — métricas por modelo y modo con latencias reales
- `evaluation/artifacts/model_eval_summary.md` — resumen para el video
