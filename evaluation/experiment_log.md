# Experiment Log — Traducción PyTorch → Triton
**Responsable:** Dilan Ocampo  
**Proyecto:** GPU Optimizer Triton  

---

## Experimento — Traducción PyTorch → Triton con LLM

**Objetivo:** Evaluar qué tan bien cada modelo traduce código PyTorch a kernels Triton optimizados.  
**Tarea:** Traducir 3 operaciones PyTorch (vector_add, vector_relu, vector_scale) a kernels Triton.

### Modelos usados

| ID | Nombre | Provider | Versión | Tier |
|---|---|---|---|---|
| `small_qwen25_coder_1_5b` | Qwen2.5-Coder-1.5B | Ollama local | qwen2.5-coder:1.5b | small |
| `frontier_openai` | GPT-4o | OpenAI API | gpt-4o | frontier |
| `frontier_anthropic` | Claude Haiku 4.5 | Anthropic API | claude-haiku-4-5-20251001 | frontier |

### Ejemplos PyTorch traducidos

| ID | Código PyTorch | Operación |
|---|---|---|
| `vector_add` | `z = x + y` | Suma elemento a elemento |
| `vector_relu` | `z = torch.relu(x)` | Activación ReLU |
| `vector_scale` | `z = x * alpha` | Multiplicación por escalar |

### Prompt usado

Archivo: `prompts/pytorch_to_triton_prompt.txt`  
Mismo prompt para los 3 modelos — condición fundamental de comparación justa.  
El prompt recibe `{pytorch_code}` y `{operation_description}` como variables.

### Parámetros de generación

| Parámetro | Valor |
|---|---|
| `temperature` | 0.2 |
| `seed` | 7 |
| `samples_per_model` | 3 |
| `max_tokens` (Anthropic) | 1024 |

---

## Comandos para reproducir

```bash
# Recolectar outputs de traducción (3 modelos × 3 ejemplos × 3 muestras = 27 llamadas)
python evaluation/collect_real_outputs.py --all --task-type translation

# Evaluar outputs y generar artefactos
python evaluation/model_evaluation.py --manual-outputs evaluation/translation_outputs.jsonl
```

---

## Artefactos generados

- `evaluation/translation_outputs.jsonl` — 27 outputs crudos de los modelos
- `evaluation/artifacts/generated_kernels.jsonl` — kernels extraídos en código limpio + métricas heurísticas
- `evaluation/artifacts/model_eval_results.csv` — métricas por modelo, ejemplo y muestra
- `evaluation/artifacts/model_eval_summary.md` — resumen para presentar

---

## Kernels listos para TritonBench (German)

Archivo: `evaluation/artifacts/generated_kernels.jsonl`  
Total de kernels esperados: 27 (3 modelos × 3 ejemplos × 3 muestras)

Formato de cada registro:
- `extracted_code` — código Python limpio (sin markdown)
- `model_id`, `mode`, `task_id`, `sample_index` — identificadores
- `syntax_valid`, `kernel_shape_valid`, `correctness_proxy` — métricas heurísticas previas

German toma estos kernels y los evalúa en TritonBench: compilación, corrección numérica y speedup vs PyTorch.
