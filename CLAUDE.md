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
| `grammars/tritonbench_t/general_kernel_family.ebnf` | Gramática GBNF activa — actualmente v6 (correcta, NO tocar) |
| `evaluation/generate_constrained_predictions.py` | Runner de las 166 predicciones |
| `evaluation/predictions_qwen_constrained.jsonl` | Salida de predicciones (166 líneas al finalizar) |
| `generation/tritonbench_constrained_decoding.py` | Prompt builder — une instruction + PyTorch + few-shot |
| `grammars/few_shot_examples.py` | Ejemplos few-shot por familia — **actualmente v3 (2 ejemplos por familia, sin comentarios)** |
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
- **ANTES de evaluar en Modal:** validar JSONL con script de limpieza (ver abajo)
  - El runner de predicciones a veces corrompe líneas con null bytes (`\x00`) o fragmentos de código sueltos
  - El evaluador de Modal truena con `JSONDecodeError` si hay UNA línea mala
  - **TODO para siguiente iteración:** agregar validación automática al final de `generate_constrained_predictions.py` que limpie null bytes y líneas no-JSON antes de escribir el archivo final

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

Rules: kernel name ends with _kernel, wrapper calls it by name.
Pass tensors directly to kernel (not .data_ptr()).
Use input.numel() for element count.
Use tl.math.exp, tl.math.log for math ops (not tl.tanh, tl.sigmoid).
tl.arange needs tl.constexpr args (BLOCK_SIZE).
Wrapper function MUST be named: {wrapper_name}
Kernel function MUST be named: {wrapper_name}_kernel

Now generate the Triton kernel:
```

---

## Estado actual — v8 EN PROGRESO (corrida activa)

**Corrida en background activa.** Las predicciones están generándose ahora mismo.
**NO correr otra instancia del runner** — sobreescribiría el archivo.

### Qué está activo ahora (v8)
Todos los cambios ya están en disco y corriendo:

**Gramática:** `general_kernel_family.ebnf` = v6 (simple, launcher libre {1,40} líneas) ✓

**Token bans en `generation/xgrammar_llm_decoder.py` y `generate_constrained_predictions.py`:**
```python
BAD_WORDS_IDS = [
    [2196, 4348],   # .data_ptr  → x_ptr/output_ptr NO afectados (prefijo diferente)
    [2196, 5348],   # .dataPtr   → evasión camelCase del ban anterior
    [11544, 84058], # tl.sigmoid → tl.math.sigmoid NO afectado (.math en medio)
]
```

**Prompt injection en `generation/tritonbench_constrained_decoding.py`:**
```
Wrapper function MUST be named: {wrapper_name}
Kernel function MUST be named: {wrapper_name}_kernel
```
El nombre se extrae de la instrucción (regex sobre "Wrapper Entry Information") — disponible en 166/166 operadores. No es trampa — es info del input, no del output.

**Few-shot:** v3 sin cambios (2 ejemplos por familia, sin comentarios)

### Resultado parcial visto hasta ahora (52/166 evaluados)
- call_acc: 6/52 = **11.5%**
- exe_acc: 6/6 = **100%** — todo lo que pasa call_acc es correcto
- Survivors: solve_multiple_lu, sub, fused_mv_logsoftmax_dropout, max, log1p, sqrt_exp

### Flujo para cuando termine la corrida
```bash
# 1. Verificar que terminó
wc -l evaluation/predictions_qwen_constrained.jsonl  # debe ser 166

# 2. Limpiar null bytes (pueden aparecer si el runner se interrumpió)
python -c "
with open('evaluation/predictions_qwen_constrained.jsonl', 'rb') as f: data = f.read()
data = data.replace(b'\x00', b'')
with open('evaluation/predictions_qwen_constrained.jsonl', 'wb') as f: f.write(data)
"

# 3. Validar que todas las líneas son JSON válido
python -c "
import json
with open('evaluation/predictions_qwen_constrained.jsonl') as f:
    for i, line in enumerate(f, 1):
        json.loads(line.strip())
print('All OK')
"

# 4. Evaluar en Modal
cd extras/TritonBench4Modal-main
modal run modal_app.py::evaluate_only \
  --predictions ../../evaluation/predictions_qwen_constrained.jsonl \
  --output-subdir results_constrained_partial

# 5. Pegar output en UPDATE.md y analizar
```

### Siguientes mejoras propuestas (después de ver resultados completos)
1. **Banear más evasiones** si aparecen (`.data`, `.numpy()`, etc.) — bajo riesgo
2. **Few-shot con flatten/reshape** para atacar shape mismatch (13% de errores) — riesgo medio
3. **Analizar si wrapper name injection funcionó** — NameError era 22% de errores

---

## Historial de la gramática — leer COMPLETO antes de tocar

> **IMPORTANTE:** Cada versión tiene evidencia empírica. No repetir errores pasados.

### v3 — baseline inicial
- Gramática básica: imports + kernel + launcher
- **Resultado:** 3/22 call_acc = **13.6%**
- **Errores:** NameError mismo nombre (4), `.data_ptr()` (varios), SyntaxError truncación, APIs inexistentes

### v4 — fixes NameError y firma corrupta
- Kernel forzado a `_kernel`, `launcher_param_char` excluye `)`, `mid` {0,45}, `launcher_body` {1,30}
- **Resultado:** 3/27 call_acc = **11%** (bajó)
- **Lección:** `launcher_body` {1,30} es MUY poco — rompe wrappers legítimos

### v5 — intento de fix SyntaxError
- `stmt_text` reducido de 160 a 130 chars, `launcher_body` {1,35}
- **Resultado:** 2/24 call_acc = **8.3%** (bajó más)
- **Lección CRÍTICA:** Reducir `stmt_text` EMPEORA las cosas — más truncaciones. NUNCA bajar de 200.

### v6 — mejor resultado hasta ahora
- `stmt_text` 200 chars, `launcher_body` {1,40}, prompt con reglas, few-shot con comentario anti-.data_ptr()
- **Resultado:** 4/30 call_acc = **13.3%** — MEJOR resultado
- **Análisis detallado de los 40 predicciones:**
  - 34/40 pasan syntax check (85%)
  - 22/40 (55%) usan `.data_ptr()` en wrapper — ERROR #1 POR LEJOS
  - 17 de esos 22 copian el comentario "Pass tensors directly" y LUEGO usan `.data_ptr()` igual
  - **El modelo NO lee comentarios ni instrucciones. Solo copia patrones visuales del few-shot.**
- **Clasificación de errores evaluados (30):**
  - `.data_ptr()` → ptr type int64 (4/30): div, add, rsqrt, tanh
  - SyntaxError truncación (5/30): sigmoid_conv2d, svd, relu_sqrt, grid_sample, silu_batch_norm, sigmoid_argmax
  - NameError wrapper no definido (3/30): fused_silu_layer_norm_conv2d, fused_lu_solve, determinant_via_qr
  - API inexistente (2/30): tl.math.tanh, tl.language.random.uniform
  - Shape/lógica incorrecta (5/30): relu M,N on 1D, assertion errors, wrong args
  - Triton compile error (3/30): arange sin constexpr, exp on int64, ptr type
  - **PASARON:** sqrt, sub, i0, fused_tile_exp (4/30)

### v7 — structured kernel call (FRACASO — no usar)
- `launcher_body` dividido en `setup_lines` + `call_line` (structured) + `post_lines`
- `call_line` bloqueaba `.data_ptr()` estructuralmente en los args del kernel call
- **Resultado:** 0/20 call_acc = **0%** — DESTRUYÓ todo
- **Por qué falló:**
  1. La estructura rígida `setup → call → post` confundió al modelo
  2. `setup_lines{0,25}` + `post_lines{0,15}` daban demasiado espacio libre → basura repetitiva
  3. El modelo generaba test code, verificaciones, segundo kernel call, `<|fim_middle|>` tokens
  4. `.data_ptr()` reaparecía en las `launcher_free_line` (setup/post), no en el `call_line`
  5. Donde SÍ funcionó (pred 2: div): `div_kernel[grid](input, other, output, n_elements, BLOCK_SIZE=1024)` — sin `.data_ptr()`! Pero falló por otro motivo (ptr type fp32 en kernel body)
- **Lección CRÍTICA:** GBNF no puede prohibir `.data_ptr()` selectivamente sin romper todo. La gramática debe ser SIMPLE — el modelo 1.5B no tolera estructuras complejas. Cualquier restricción que divida el launcher en bloques genera basura.

### v8 — gramática v6 + bad_words_ids + wrapper name injection (CORRIDA EN PROGRESO)
- Gramática idéntica a v6 (simple, launcher libre)
- `bad_words_ids` banea: `.data_ptr` [2196,4348], `.dataPtr` [2196,5348], `tl.sigmoid` [11544,84058]
- Prompt inyecta nombre exacto del wrapper extraído de la instrucción (166/166 operadores tienen el nombre)
- **Resultado parcial (52/166):** 6/52 call_acc = **11.5%**, exe_acc **100%**
- **Survivors parciales:** solve_multiple_lu, sub, fused_mv_logsoftmax_dropout, max, log1p, sqrt_exp
- **Ban funcionó:** .data_ptr bajó de 55% a 6% de predicciones
- **Error #1 ahora:** NameError — modelo renombra wrapper (22% de fallos) — atacado con wrapper name injection
- **Error #2:** Shape mismatch 1D/3D/4D (13%) — sin fix todavía
- **Error #3:** SyntaxError/truncación (11%)
- **NOTA:** porcentaje bajo con 52 muestras NO es alarma — la muestra crece, los operadores difíciles también

---

## Lecciones duras — NO REPETIR

1. **NUNCA bajar `stmt_text`** por debajo de 200 — siempre empeora (v5)
2. **NUNCA bajar `launcher_body`** por debajo de {1,35} — rompe wrappers (v4)
3. **Comentarios en few-shot NO funcionan** — el modelo los copia sin entenderlos (v6)
4. **Reglas en prompt NO eliminan `.data_ptr()`** — el modelo no las lee (v6)
5. **Gramática estructurada compleja DESTRUYE la generación** — el modelo 1.5B se pierde con bloques rígidos (v7)
6. **GBNF no puede hacer negative lookahead** — no se puede prohibir una secuencia específica sin romper todo
7. **El modelo copia ESTRUCTURA VISUAL del few-shot**, no instrucciones textuales

## Lo que SÍ funciona

1. **Few-shot con 2 ejemplos cortos por familia** (v3) — refuerza patrón visual
2. **`_kernel` suffix forzado** — elimina NameError de nombres duplicados (v4+)
3. **`stmt_text` largo (200+)** — reduce truncaciones (v6)
4. **`launcher_body` generoso ({1,40})** — da espacio al modelo (v6)
5. **Gramática SIMPLE y libre en el launcher** — el modelo funciona mejor con menos restricciones
6. **`bad_words_ids` para banear secuencias de tokens** — elimina `.data_ptr()` sin romper nada (v8)
7. **100% exe_acc en survivors** — lo que pasa call_acc está bien escrito (v8)

## Problemas resueltos y abiertos

### `.data_ptr()` — RESUELTO en v8
- Era error #1 en v6 (55% de predicciones). Resuelto con `bad_words_ids`.
- El modelo evade con `.dataPtr()` camelCase → también baneado [2196, 5348].
- NO es post-procesamiento — es restricción de generación como XGrammar.

### NameError (wrapper renombrado) — PARCIALMENTE ATACADO en v8
- 22% de fallos en v8 parcial. El modelo genera `tile_exp` en vez de `fused_tile_exp`.
- Fix en v8: inyectar `Wrapper function MUST be named: X` en el prompt.
- Pendiente confirmar si funciona en resultados completos.

### Shape mismatch — ABIERTO
- 13% de fallos. El modelo asume `M, N = input.shape` pero input puede ser 1D/3D/4D.
- Opciones: few-shot con `.view(-1)` antes del kernel call — riesgo medio (puede romper 2D).
- NO atacar con gramática — cualquier restricción de shape rompe todo (lección v7).

### SyntaxError / truncación — PARCIALMENTE MITIGADO
- 11% de fallos. Nombres de variables degenerados muy largos, paréntesis sin cerrar.
- `stmt_text{1,200}` en gramática ya lo limita. No hay más que hacer sin arriesgar.

### APIs inexistentes (`tl.tanh`, `tl.math.mean`) — PARCIALMENTE ATACADO
- `tl.sigmoid` baneado. `tl.tanh` NO se puede banear (comparte tokens con `tl.math.tanh`).
- Alternativa: mejorar regla en prompt (efecto limitado en modelo 1.5B).

---

## Clasificación de familias (para few-shot)

| Familia | Few-shot usado | Operadores típicos |
|---|---|---|
| `elementwise` | add + neg (2 ejemplos) | relu, sigmoid, tanh, sqrt, abs |
| `elementwise_fusion` | add + neg (2 ejemplos) | add+gelu, mul+relu, etc. |
| `indexing_interpolation` | add + neg (mismo) | grid_sample, gather |
| `complex_fallback` | add + neg (mismo) | cualquier cosa no clasificada |
| `reduction` | softmax + sum (2 ejemplos) | argmax, logsumexp, std, sum |
| `fusion_matmul` | matmul (1 ejemplo) | matmul, bmm, addmm |
| `convolution_fusion` | matmul (1 ejemplo) | conv2d fusions |

---

## Restricciones del proyecto

- **Constrained decoding con XGrammar es obligatorio** — no se puede quitar
- **Modelo fijo:** Qwen2.5-Coder-1.5B-Instruct — no se puede cambiar
- **Post-procesamiento de outputs NO es aceptable** — se considera trampa
- No tocar `extras/` (código del benchmark oficial de la escuela)
