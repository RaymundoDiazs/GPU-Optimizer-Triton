# CLAUDE.md — GPU Optimizer Triton
## Guía de contexto completo del proyecto para Claude

---

## Objetivo del reto

Evaluar qué tan bien distintos modelos de lenguaje pueden **traducir código PyTorch a kernels Triton**, midiendo si el código generado funciona, es numéricamente correcto, y es más rápido que PyTorch.

**Pregunta central:** ¿Qué técnicas adicionales (constrained decoding) mejoran la traducción PyTorch→Triton?

El reto tiene 4 partes:

| Parte | Descripción |
|---|---|
| **1. Generación baseline** | 3 modelos generan kernels Triton a partir de los 166 operadores de TritonBench-T. Evaluar con TritonBench4Modal. |
| **2. HumanEval** | Medir capacidad general de programación Python de cada modelo (pass@1, pass@k). Comparar si esa habilidad predice el desempeño en Triton. |
| **3. Constrained decoding** | Usar XGrammar para restringir tokens durante generación. Comparar resultados vs baseline. |
| **4. Análisis estadístico** | Comparar call@1, exe@1, speedup por modelo y por modo (baseline vs constrained). |

---

## Benchmark de evaluación: TritonBench4Modal

Código en `extras/TritonBench4Modal-main/modal_app.py` (copia local del repo `salvahin/TritonBench4Modal`).

### ⚠️ REGLA CRÍTICA — SOLO TOCAR `PROMPT_HEADER`
**La carpeta `extras/TritonBench4Modal-main/` solo se puede modificar para cambiar `PROMPT_HEADER` en `modal_app.py` — nada más.**
Todos los equipos de la escuela usan el mismo pipeline de evaluación. `PROMPT_HEADER` es nuestro diferenciador legítimo: cambia cómo se genera el código, no cómo se evalúa. Cualquier otra modificación rompe la comparabilidad con los demás equipos.

### Los 166 operadores
- Fuente: TritonBench Track T — mismo benchmark que usan todos los equipos de la escuela
- Dataset: `TritonBench_T_simp_alpac_v1.json` (versión simple) — vive dentro del contenedor Modal en `/opt/TritonBench/data/`
- Formato Alpaca: cada entrada tiene `instruction` (descripción funcional + firma) y `output` vacío

### Las 3 métricas (jerarquía obligatoria)
| Fase | Métrica | Qué mide |
|---|---|---|
| 1 | **call@1** | ¿El código generado corre sin error? (concatena predicción + test driver, ejecuta) |
| 2 | **exe@1** | ¿El output es numéricamente correcto vs PyTorch? (torch.allclose) |
| 3 | **speedup** | ¿Es más rápido que PyTorch? (solo si pasa exe@1) |

**Punto pedagógico clave** (del notebook de clase): un kernel puede pasar call@1 y fallar exe@1 (ej: `sum` sin dividir entre N pasa sintaxis pero da resultado incorrecto). Por eso la jerarquía importa.

### Cómo se usa para evaluar kernels propios (`evaluate_only`)

```bash
# Desde extras/TritonBench4Modal-main/
pip install -r requirements-local.txt
modal setup
# Subir predictions.jsonl y evaluar en GPU (no se necesita API key para esto)
modal run modal_app.py::evaluate_only --predictions ./predictions_qwen.jsonl
```

### Formato exacto del archivo `predictions.jsonl`

Cada línea es un JSON con exactamente estos dos campos:
```json
{"instruction": "<texto exacto del campo instruction del Alpaca>", "predict": "```python\n<código generado>\n```"}
```

**CRÍTICO:** El campo `instruction` debe ser **idéntico** al del dataset original — el evaluador lo usa para encontrar el test driver correspondiente (busca el substring entre `"Functional Description: "` y `"Wrapper Entry Information:"`).

---

## División de tareas del equipo

### Ocampo (Dilan) — generación + HumanEval
- **GPT-4o y Claude Haiku:** usar `modal_app.py::main` directamente — Modal llama las APIs y genera los 166 kernels. Costo: APIs de OpenAI y Anthropic + GPU Modal para evaluación.
- **Qwen:** generar localmente con `collect_real_outputs.py` (Ollama, sin costo) → producir `predictions_qwen.jsonl` → pasar a German para que corra `evaluate_only` en Modal.
- Los 3 archivos resultantes tienen 166 líneas cada uno, formato exacto `{"instruction": "...", "predict": "..."}`.
- HumanEval (pass@1 y pass@k) para los 3 modelos — ideal pero no bloqueante.
- Mantener evaluación heurística interna como capa previa (sintaxis, kernel shape, correctness proxy).

### Ray — gramática formal
- Definir una gramática EBNF general que describe qué es un kernel Triton válido (no necesita ser específica a cada uno de los 166 operadores — una gramática general es suficiente)
- Validar que los 100 kernels de `extras/student_package/datasets/adversarial_100.jsonl` sean **rechazados** por la gramática (son kernels intencionalmente inválidos)
- Validar que los 100 kernels de `extras/student_package/datasets/curated_100.jsonl` sean **aceptados**
- Completar `get_valid_next_tokens()` en `parsing/triton_grammar_rules.py` (interfaz para Charlie)

### Charlie — constrained decoding
- Integrar XGrammar real con Qwen (único modelo local compatible — los modelos de API no permiten restringir tokens)
- Compilar la gramática de Ray con XGrammar y usarla para restringir tokens durante generación
- Producir `predictions_qwen_constrained.jsonl` — mismo formato que los de Dilan, para los mismos 166 operadores (o subset)
- Este archivo va a German para evaluación comparativa

### German — evaluación GPU con TritonBench4Modal
- Configurar Modal (cuenta + GPU T4)
- Correr `evaluate_only` sobre los 4 archivos: `predictions_qwen.jsonl`, `predictions_gpt4o.jsonl`, `predictions_claude.jsonl`, `predictions_qwen_constrained.jsonl`
- Reportar call@1 %, exe@1 %, geometric speedup por modelo y modo
- Documentar errores frecuentes de compilación (especialmente de Qwen)

---

## Flujo de datos completo

```
⚠️  extras/TritonBench4Modal-main/  →  SOLO TOCAR PROMPT_HEADER
    modal_app.py es la herramienta de evaluación estándar de toda la escuela
    Solo se modificó PROMPT_HEADER (diferenciador del proyecto) — el pipeline de evaluación no se toca

──────────────────────────────────────────────────────────
GENERACIÓN
──────────────────────────────────────────────────────────

GPT-4o y Claude Haiku:
  modal run modal_app.py::main --provider openai --model gpt-4o
  modal run modal_app.py::main --provider anthropic --model claude-haiku-4-5-20251001
  ↓ Modal llama las APIs, genera 166 kernels, evalúa en GPU automáticamente
  ↓ resultados directamente en Modal Volume

Qwen (local, Ollama — Modal no puede llamarlo):
  [DILAN] evaluation/collect_real_outputs.py  ← genera localmente
  ↓
  evaluation/predictions_qwen.jsonl           ← {"instruction": "...", "predict": "..."}
  ↓ (evaluación heurística interna — opcional, no es el benchmark oficial)
  evaluation/artifacts/model_eval_results.csv
  ↓
  [GERMAN] modal run modal_app.py::evaluate_only --predictions ./predictions_qwen.jsonl

[CHARLIE] Qwen + XGrammar (constrained decoding, solo posible en modelo local):
  generation/xgrammar_llm_decoder.py  ← Qwen restringido con gramática de Ray
  ↓
  evaluation/predictions_qwen_constrained.jsonl
  ↓
  [GERMAN] modal run modal_app.py::evaluate_only --predictions ./predictions_qwen_constrained.jsonl

──────────────────────────────────────────────────────────
RESULTADOS (todos en GPU real vía Modal)
──────────────────────────────────────────────────────────

results/tritonbench_gpt4o_baseline.json
results/tritonbench_claude_baseline.json
results/tritonbench_qwen_baseline.json
results/tritonbench_qwen_constrained.json

  → call@1 %  |  exe@1 %  |  geometric speedup vs PyTorch
```

---

## Estado actual por persona

### Dilan — 60% ⚠️

| Componente | Estado | Notas |
|---|---|---|
| 3 modelos configurados (Qwen, GPT-4o, Claude Haiku) | ✅ | `config/model_eval.yaml` |
| Prompt de traducción | ✅ | `prompts/pytorch_to_triton_prompt.txt` — revisar compatibilidad con formato Alpaca |
| Script `collect_real_outputs.py` funcionando | ✅ | Implementado, corre los 3 modelos |
| 360 outputs reales (6 ejemplos propios × 3 modelos × 20 muestras) | ✅ | Prueba de concepto válida, no es el benchmark final |
| Evaluación heurística (`model_evaluation.py`) | ✅ | Funciona como capa previa |
| HumanEval de Qwen (~44% pass@1) documentado | ✅ | `evaluation/humaneval_context.md` |
| Tests unitarios | ✅ | 32 tests pasan |
| **Adaptar `collect_real_outputs.py` para leer `TritonBench_T_simp_alpac_v1.json`** | ❌ | Cambio principal |
| **Producir `predictions_<modelo>.jsonl` en formato correcto para `evaluate_only`** | ❌ | Entregable real |
| **HumanEval con pass@k para GPT-4o y Claude Haiku** | ❌ | Solo Qwen está documentado |

**Pendiente:**
- [ ] Obtener `TritonBench_T_simp_alpac_v1.json` (está dentro del contenedor Modal; se puede descargar directamente del repo `thunlp/TritonBench`)
- [ ] Adaptar `collect_real_outputs.py` para leer formato Alpaca y producir `predictions_<modelo>.jsonl`
- [ ] Correr los 166 operadores completos (sin subset)
- [ ] Correr HumanEval pass@1 y pass@k para GPT-4o y Claude Haiku (misma metodología que el notebook de clase)

---

### Ray — 60% ⚠️

| Componente | Estado | Notas |
|---|---|---|
| Gramática EBNF para `vector_add` | ✅ | `grammars/vector_add.ebnf` — base, pero es específica a un operador |
| Contrato de diseño | ✅ | `grammars/vector_add_contract.md` |
| Validador regex post-generación | ✅ | `parsing/triton_grammar_rules.py` |
| **Gramática EBNF general** (no específica a vector_add) | ❌ | Es lo que realmente necesita Charlie |
| **Validar `adversarial_100.jsonl`** (100 kernels inválidos deben ser rechazados) | ❌ | Dataset está en `extras/student_package/datasets/` |
| **Validar `curated_100.jsonl`** (100 kernels válidos deben ser aceptados) | ❌ | Dataset está en `extras/student_package/datasets/` |
| `get_valid_next_tokens()` completo | ❌ | Stub vacío — bloquea a Charlie |

**Pendiente:**
- [ ] Generalizar la gramática: de vector_add específico → gramática general de "kernel Triton válido"
- [ ] Correr el validador contra `extras/student_package/datasets/adversarial_100.jsonl` y `curated_100.jsonl` — documentar precisión (% correctamente clasificados)
- [ ] Completar `get_valid_next_tokens()` para que Charlie pueda integrarlo con XGrammar

---

### Charlie — 30% ❌

| Componente | Estado | Notas |
|---|---|---|
| Estructura de `XGrammarLLMDecoder` | ✅ | `generation/xgrammar_llm_decoder.py` — definida pero no conectada |
| `ConstrainedDecoder` prototipo | ⚠️ | Valida post-hoc, NO restringe tokens durante generación |
| XGrammar en `requirements.txt` | ❌ | Falta agregar |
| Integración con Ollama/Qwen durante generación | ❌ | No conectado |
| Usa gramática general de Ray | ❌ | Usa `DEFAULT_TRITON_EBNF` propio |
| `predictions_qwen_constrained.jsonl` producido | ❌ | No existe |

**Pendiente:**
- [ ] Agregar `xgrammar` a `requirements.txt`
- [ ] Esperar gramática general de Ray (`get_valid_next_tokens()` completo)
- [ ] Conectar XGrammar con Ollama/Qwen: restringir logits en tiempo real durante generación
- [ ] Producir `evaluation/predictions_qwen_constrained.jsonl` — formato idéntico al de Dilan

**Nota crítica:** XGrammar solo funciona con modelos locales (Ollama/HuggingFace). Las APIs de OpenAI y Anthropic no permiten intervenir en la generación de tokens. Por eso constrained decoding solo aplica a Qwen.

---

### German — 20% ❌

| Componente | Estado | Notas |
|---|---|---|
| TritonBench manual en 6 ejemplos propios de Qwen | ✅ | 27 registros — referencia, no es el benchmark final |
| **Cuenta Modal configurada con GPU T4** | ❌ | Primer paso |
| **`evaluate_only` corrido sobre `predictions_qwen.jsonl`** | ❌ | Esperando archivo de Dilan |
| **`evaluate_only` corrido sobre `predictions_gpt4o.jsonl`** | ❌ | Esperando archivo de Dilan |
| **`evaluate_only` corrido sobre `predictions_claude.jsonl`** | ❌ | Esperando archivo de Dilan |
| **`evaluate_only` corrido sobre `predictions_qwen_constrained.jsonl`** | ❌ | Esperando archivo de Charlie |

**Pendiente:**
- [ ] Crear cuenta Modal y configurar GPU (T4 = $0.59/hr — cuesta pocos dólares por run completo)
- [ ] Cuando Dilan entregue los `predictions_<modelo>.jsonl`, correr `evaluate_only` para cada uno
- [ ] Guardar los JSON de resultados en `results/` con naming claro
- [ ] Documentar errores frecuentes de compilación de Qwen

---

## Resultados actuales (referencia — 6 ejemplos propios, no el benchmark final)

### Heurísticos (Dilan)

| Modelo | Muestras | Sintaxis válida | Kernel shape válido | Correctness proxy |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B | 120 | 100% | 32% | 0% |
| GPT-4o | 120 | 100% | 100% | 100% |
| Claude Haiku 4.5 | 120 | 100% | 98% | 100% |

### GPU real (German, solo Qwen × 27 muestras)
- call@1: **0%** — ningún kernel compila en Triton
- Causa: no usa `tl.load`/`tl.store`, no usa `tl.program_id`, loops secuenciales, sin boundary mask

### HumanEval (pass@1)
| Modelo | pass@1 |
|---|---|
| Qwen2.5-Coder-1.5B | ~44% |
| GPT-4o | ~90% |
| Claude Haiku 4.5 | ~88% |

La brecha HumanEval (44% vs 90%) predice directamente la brecha en Triton.

---

## Archivos clave

```
DILAN:
  config/model_eval.yaml                          ← 3 modelos, temperatura 0.15, seed 42
  prompts/pytorch_to_triton_prompt.txt            ← template compatible con formato Alpaca
  evaluation/collect_real_outputs.py              ← script de generación (adaptar para Alpaca)
  evaluation/predictions_qwen.jsonl               ← ENTREGABLE para German (pendiente)
  evaluation/predictions_gpt4o.jsonl              ← ENTREGABLE para German (pendiente)
  evaluation/predictions_claude.jsonl             ← ENTREGABLE para German (pendiente)
  evaluation/artifacts/model_eval_results.csv     ← métricas heurísticas internas
  evaluation/artifacts/model_eval_summary.md      ← tabla resumen
  evaluation/humaneval_context.md                 ← HumanEval Qwen documentado

RAY:
  grammars/triton_general.ebnf                    ← gramática general (pendiente — hoy solo vector_add)
  grammars/vector_add.ebnf                        ← gramática específica (base)
  grammars/vector_add_contract.md                 ← documentación
  parsing/triton_grammar_rules.py                 ← validador (completar get_valid_next_tokens)
  extras/student_package/datasets/                ← datasets de validación (curated_100, adversarial_100)

CHARLIE:
  generation/xgrammar_llm_decoder.py              ← decoder (conectar con Qwen + gramática de Ray)
  generation/constrained_decoder.py               ← prototipo post-hoc (no es constrained real)
  evaluation/predictions_qwen_constrained.jsonl   ← ENTREGABLE para German (pendiente)

GERMAN:
  extras/TritonBench4Modal-main/modal_app.py      ← herramienta de evaluación
  results/tritonbench_qwen_baseline.json          ← pendiente
  results/tritonbench_gpt4o_baseline.json         ← pendiente
  results/tritonbench_claude_baseline.json        ← pendiente
  results/tritonbench_qwen_constrained.json       ← pendiente

REFERENCIA (ya existe, no es el benchmark final):
  evaluation/translation_outputs.jsonl            ← 360 outputs (6 ejemplos × 3 modelos × 20 muestras)
  results/generated_kernels_tritonbenchG.jsonl    ← 27 kernels evaluados manualmente (Qwen)
```

---

## Parámetros de generación

| Parámetro | Valor | Razón |
|---|---|---|
| temperature | 0.15 | Del notebook de clase — baja para código determinista |
| top_p | 0.95 | Del notebook de clase |
| seed | 42 | Del notebook de clase — reproducibilidad |
| samples_per_operator | 1 | Con 166 operadores, 1 muestra por operador es suficiente para call@1/exe@1 |
| dataset | `TritonBench_T_simp_alpac_v1.json` | Versión simple — instrucciones más directas |
| modelos | Qwen2.5-Coder-1.5B (Ollama local), GPT-4o (OpenAI), Claude Haiku 4.5 (Anthropic) | Small local + 2 frontier |

> **Nota sobre samples:** Para HumanEval se necesitan N≥3 muestras por problema para calcular pass@k con el estimador unbiasado. Para TritonBench, 1 muestra por operador basta para call@1/exe@1/speedup.

---

## Problemas críticos a resolver

| Prioridad | Problema | Quién |
|---|---|---|
| 🔴 CRÍTICO | Adaptar `collect_real_outputs.py` al formato Alpaca y producir `predictions_<modelo>.jsonl` | Dilan |
| 🔴 CRÍTICO | XGrammar no conectado — no hay constrained decoding real | Charlie |
| 🔴 CRÍTICO | German no tiene cuenta Modal configurada | German |
| 🟠 ALTO | Gramática de Ray solo cubre vector_add — necesita ser general | Ray |
| 🟠 ALTO | `get_valid_next_tokens()` vacío bloquea a Charlie | Ray |
| 🟠 ALTO | HumanEval de GPT-4o y Claude no documentado | Dilan |
| 🟡 MEDIO | Análisis estadístico final (comparar baseline vs constrained) | Todos |
| 🟡 MEDIO | Video de presentación en inglés (5–8 min, todos hablan) | Todos |

---

## Recursos de clase disponibles en `extras/`

| Archivo | Qué es | Para quién |
|---|---|---|
| `extras/tritonbench_t_didactic-nvidia.ipynb` | Notebook didáctico completo del pipeline: prompt → LLM → call@1 → exe@1 → speedup | Todos — es el blueprint del reto |
| `extras/gemma2_humaneval_lab.ipynb` | Notebook de HumanEval con Gemma 2B: cómo calcular pass@1 y pass@k | Dilan — replicar para los 3 modelos |
| `extras/TritonBench4Modal-main/modal_app.py` | Herramienta de evaluación en GPU | German — correr `evaluate_only` |
| `extras/student_package/datasets/curated_100.jsonl` | 100 kernels Triton válidos y bien escritos | Ray — deben ser aceptados por la gramática |
| `extras/student_package/datasets/adversarial_100.jsonl` | 100 kernels Triton intencionalmente inválidos (categorizados por tipo de error) | Ray — deben ser rechazados por la gramática |
| `extras/student_package/examples/valid_kernel_*.py` | Ejemplos de kernels válidos (vector_add, softmax) | Ray/Charlie — referencia |
| `extras/student_package/examples/invalid_kernel_*.py` | Ejemplos de kernels inválidos (missing colon, bad token, bad assignment) | Ray — casos de prueba |

---

## Cómo presentar cada parte (para el video en inglés)

**Dilan — Generation:**
> "My part is generation. I designed a prompt compatible with the TritonBench-T Alpaca format, configured three models — Qwen locally, GPT-4o, and Claude Haiku — and generated Triton translations for the 166 benchmark operators. I also ran HumanEval to measure general Python coding ability and compared whether that score predicts Triton translation quality."

**Ray — Grammar:**
> "My part defines formally what a valid Triton kernel looks like. The EBNF grammar specifies required structures: the jit decorator, program_id, tl.load and tl.store with masks. I validated it against 100 curated valid kernels and 100 adversarial invalid ones. This grammar is what Charlie compiles with XGrammar to constrain generation."

**Charlie — Constrained Decoding:**
> "My part takes Ray's grammar, compiles it with XGrammar, and uses it to restrict which tokens the model can generate in real time — not validating after the fact, but steering during generation. This reduces syntax errors and increases the probability of the kernel compiling on the first try."

**German — TritonBench Evaluation:**
> "My part evaluates all generated kernels on a real GPU using TritonBench4Modal on Modal. I run three phases: call accuracy — does the code run at all? Execution accuracy — does it produce the same numbers as PyTorch? And speedup — is it actually faster? We use the same benchmark as every other team in the class, so our numbers are directly comparable."
