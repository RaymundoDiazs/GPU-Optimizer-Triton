# CLAUDE.md — GPU-Optimizer-Triton

Instrucciones y contexto persistente para Claude Code. Lee esto al inicio de cada sesión.

---

## Proyecto

Proyecto académico que compara modelos LLM generando kernels Triton desde código PyTorch.
- **Modelos:** Qwen2.5-Coder-1.5B-Instruct (local), GPT-4o, Claude
- **Diferenciador experimental:** Qwen corre con **constrained decoding (XGrammar)** — esto es requisito del proyecto y NO se puede cambiar ni quitar
- **Modelo NO se puede cambiar** — debe ser Qwen2.5-Coder-1.5B-Instruct
- Evaluación en 3 fases: call_acc → execution_acc → speedup vs PyTorch
- Baseline de Qwen sin constrained: `results/tritonbench_qwen_baseline.json`

---

## Entorno

| Componente | Valor |
|---|---|
| Python generación | `.venv311/Scripts/python.exe` (3.11 + torch 2.5.1+cu124 + xgrammar + transformers 4.46.3) |
| Python proyecto | Python 3.14 — **NO tiene CUDA, no usar para generación** |
| GPU | NVIDIA RTX 2060 6GB |
| Modal | Evaluación corre en GPU T4 en la nube |

---

## Archivos clave

| Archivo | Descripción |
|---|---|
| `grammars/tritonbench_t/general_kernel_family.ebnf` | Gramática GBNF activa (versión actual: v6) |
| `evaluation/generate_constrained_predictions.py` | Runner de las 166 predicciones |
| `evaluation/predictions_qwen_constrained.jsonl` | Salida de predicciones (166 líneas al finalizar) |
| `generation/tritonbench_constrained_decoding.py` | Prompt builder — une instruction + PyTorch + few-shot |
| `grammars/few_shot_examples.py` | Ejemplos few-shot por familia (elementwise, reduction, matmul) |
| `extras/TritonBench4Modal-main/modal_app.py` | Evaluador Modal — usar `evaluate_only` |
| `UPDATE.md` | Donde pegar outputs del evaluador para analizar |
| `data/tritonbench_t_simp_subset166.json` | Dataset de los 166 operadores |

---

## Flujo de trabajo

```bash
# 1. Generar predicciones (166 operadores)
.venv311/Scripts/python.exe evaluation/generate_constrained_predictions.py \
  --mode family \
  --grammar-file grammars/tritonbench_t/general_kernel_family.ebnf \
  --max-new-tokens 1536

# 2. Monitorear progreso
wc -l evaluation/predictions_qwen_constrained.jsonl

# 3. Evaluar en Modal
modal run extras/TritonBench4Modal-main/modal_app.py::evaluate_only \
  --predictions evaluation/predictions_qwen_constrained.jsonl

# 4. Pegar output en UPDATE.md, analizar, iterar gramática
```

**Reglas del flujo:**
- NO usar `--resume` si la gramática cambió (el archivo se sobreescribe limpio)
- NO tocar nada en `extras/` (TritonBench4Modal-main, TritonBench-main)
- NO agregar few-shot examples sin validar que corren primero

---

## Prompt que usa el modelo

Definido en `generation/tritonbench_constrained_decoding.py:build_tritonbench_constrained_spec()`:

```
{instruction Alpaca}

PyTorch reference implementation:
```python
{código PyTorch del operador}
```

Here is a correct Triton kernel example for a similar operator ({family}):
```python
{few-shot example según familia}
```

CRITICAL RULES:
1. Kernel name MUST end with '_kernel'. Wrapper name MUST NOT contain '_kernel'.
2. Wrapper MUST call <name>_kernel[grid](...) — the name must match.
3. NEVER use .data_ptr() — pass tensors directly to the kernel launch.
4. Use input.numel() for 1D element count. Do NOT assume 2D shapes unless guaranteed.
5. Use tl.math.* for transcendental functions. Do NOT use tl.tanh, tl.sigmoid — they don't exist.
6. tl.arange arguments must be tl.constexpr (use BLOCK_SIZE, not runtime variables).

Now generate the Triton kernel:
```

---

## Historial de la gramática — para mejorarla inteligentemente

> **Al mejorar la gramática, leer esta sección completa primero.**
> Cada versión documenta qué cambió, por qué, y qué evidencia tuvo.

### v3 — baseline inicial
- Gramática básica: imports + kernel + launcher
- **Resultado:** 3/22 call_acc = **13.6%**
- **Errores principales:**
  - NameError (4/22): wrapper y kernel tenían el mismo nombre → `fused_bmm`, `solve_multiple_lu`, etc.
  - `.data_ptr()` en launcher (varios): Triton rechaza punteros int64
  - SyntaxError por truncación de líneas largas
  - `tl.tanh`, `tl.math.bessel_i0` no existen en Triton API

### v4 — fixes NameError y firma corrupta
- **Cambios:**
  - Kernel forzado a terminar en `_kernel`
  - `launcher_param_char` excluye `)` — evita firma corrupta con inline comments
  - `mid` body reducido a {0,45} — más tokens para el launcher
  - `launcher_body` reducido a {1,30}
- **Resultado:** 3/27 call_acc = **11%** (bajó — muestra que el modelo a veces necesita más líneas en launcher)
- **Lección:** Reducir `launcher_body` demasiado rompe wrappers legítimos

### v5 — fixes SyntaxError por truncación
- **Cambios:**
  - `stmt_text` reducido de 160 a 130 chars — intentaba forzar saltos de línea
  - `launcher_body` aumentado a {1,35}
- **Resultado:** 2/24 call_acc = **8.3%** (bajó más)
- **Lección crítica:** Reducir `stmt_text` no ayudó — los SyntaxErrors son por truncación, no por líneas largas. El modelo necesita MÁS chars para cerrar paréntesis/strings, no menos.

### v6 — fix real para truncación (versión activa)
- **Cambios:**
  - `stmt_text` aumentado de 130 a **200 chars** — da espacio para cerrar expresiones
  - `launcher_body` aumentado a {1,40}
  - Prompt reforzado con 6 reglas explícitas (anti-.data_ptr(), nombres, shapes, tl.math)
  - Few-shot: comentario `# Pass tensors directly — NEVER use .data_ptr()` en los 4 wrappers
- **Resultado:** pendiente (corrida en curso)
- **Hipótesis:** debería resolver ~9/22 fallos (4 .data_ptr + 5 SyntaxError truncación)

### Errores que la gramática NO puede resolver (limitación del modelo 1.5B)
Estos errores son de **capacidad del modelo**, no de gramática. No perder tiempo intentando arreglarlos con GBNF:
- `tl.tanh`, `tl.math.i0`, `tl.math.bessel_i0` — APIs inexistentes que el modelo alucina
- Shape unpacking incorrecto (`M, N = x.shape` en tensor 1D)
- Lógica errónea en el wrapper (args en orden incorrecto, assertion sobre tipos)
- `tl.exp` aplicado sobre int64

### Guía para la próxima mejora de gramática

Antes de tocar la gramática, hacer este análisis sobre `UPDATE.md`:

1. **Contar SyntaxError** → si hay líneas truncadas (unclosed `(`, unterminated string), aumentar `stmt_text`
2. **Contar NameError/UnboundLocal** → si el wrapper llama una función que no existe, el problema es que el launcher_body no tiene suficientes líneas o el modelo confunde nombres
3. **Contar `ptr type int64`** → el modelo sigue usando `.data_ptr()` — reforzar en prompt, la gramática no puede prohibirlo directamente
4. **Contar IndentationError** → la gramática fuerza indent de 4 espacios, si aparece es un bug de la gramática misma
5. **NO tocar** `mid {0,45}` — está calibrado para el máximo real del dataset (44 líneas)
6. **NO bajar** `stmt_text` — la lección de v5 es que menos chars = más truncaciones

---

## Clasificación de familias (para few-shot)

| Familia | Few-shot usado | Operadores típicos |
|---|---|---|
| `elementwise` | Tutorial 01 — vector add | relu, sigmoid, tanh, sqrt, abs |
| `elementwise_fusion` | Tutorial 01 — vector add | add+gelu, mul+relu, etc. |
| `indexing_interpolation` | neg kernel (numel pattern) | grid_sample, gather |
| `complex_fallback` | neg kernel | cualquier cosa no clasificada |
| `reduction` | Tutorial 02 — softmax | argmax, logsumexp, std, sum |
| `fusion_matmul` | Tutorial 03 — matmul | matmul, bmm, addmm |
| `convolution_fusion` | Tutorial 03 — matmul | conv2d fusions |

---

## Restricciones del proyecto

- **Constrained decoding con XGrammar es obligatorio** — no se puede quitar
- **Modelo fijo:** Qwen2.5-Coder-1.5B-Instruct — no se puede cambiar
- No tocar `extras/` (código del benchmark oficial de la escuela)
