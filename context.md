# Contexto de sesión — GPU-Optimizer-Triton

## Objetivo actual
Generar las 166 predicciones de Qwen2.5-Coder-1.5B-Instruct con **constrained decoding (XGrammar)** para el benchmark TritonBench-T, luego evaluarlas con el evaluador oficial en Modal.

---

## Estado actual

### Generación en curso
- **Script:** `evaluation/generate_constrained_predictions.py --mode family`
- **Proceso:** Corriendo en background con Python 3.11 (`venv311`) en la máquina local
- **Salida:** `evaluation/predictions_qwen_constrained.jsonl` (se llena de a uno)
- **Monitorear:** `wc -l evaluation/predictions_qwen_constrained.jsonl`

### Gramática activa
- **Archivo:** `grammars/tritonbench_t/general_kernel_family.ebnf`
- **Versión:** v4 (última, la correcta a usar)
- **Cambios clave en v4:**
  - Kernel forzado a terminar en `_kernel` → evita NameError por nombres duplicados
  - `launcher_param_char` excluye `)` → evita firma corrupta con inline comments
  - `mid` body reducido a {0,45} → más tokens para el launcher
  - `launcher_body` reducido a {1,30} → evita test-code basura

---

## Flujo de trabajo establecido

1. Esperar ~20-25 predicciones en el `.jsonl`
2. Correr evaluación en Modal con `evaluate_only` del `modal_app.py`:
   ```bash
   modal run extras/TritonBench4Modal-main/modal_app.py::evaluate_only --predictions evaluation/predictions_qwen_constrained.jsonl
   ```
3. Pegar output del evaluador en `UPDATE.md`
4. Analizar errores e iterar la gramática si hace falta
5. Cuando call_acc > 50% → lanzar los 166 completos

---

## Entorno

- **Python para las predicciones:** `.venv311/Scripts/python.exe` (Python 3.11 + torch 2.5.1+cu124 + xgrammar 0.2.1 + transformers 4.46.3)
- **Python del proyecto (otros scripts):** Python 3.14 (sin CUDA — NO usar para generación)
- **GPU:** NVIDIA RTX 2060 6GB
- **Modal:** sesión activa, evaluación corre en GPU T4

---

## Clasificación de errores del evaluador (22 predicciones v3, 3/22 call_acc = 13.6%)

### Errores que la gramática v4 ya resuelve
- **NameError wrapper/kernel mismo nombre** (4/22): `fused_bmm_rmsnorm_gelu_dropout_sub`, `solve_multiple_lu`, `fused_mv_logsoftmax_dropout`, `fused_lu_solve` → **FIX: `_kernel` forzado**

### Errores que la gramática aún no resuelve (capacidad del modelo)
- **Triton API inexistente** (2/22): `tl.tanh` no existe, `tl.math.bessel_i0` no existe
- **Lógica incorrecta** (2/22): shape unpacking erróneo, `.data_ptr()` en int
- **Compile error** (2/22): ptr type incorrecto en `tl.load`, arange sin constexpr

### Errores de sintaxis pendientes
- **SyntaxError unbalanced brackets** (1/22): `normalize_pairwise_distance` — launcher genera lista rota
- **IndentationError** (1/22): `grid_sample` — launcher con indent incorrecto
- **`def linalg.svd`** (1/22): `svd` — modelo pone punto en nombre de función (ilegítimo)

---

## Archivos clave

| Archivo | Descripción |
|---|---|
| `grammars/tritonbench_t/general_kernel_family.ebnf` | Gramática activa v4 |
| `evaluation/generate_constrained_predictions.py` | Script de generación (--mode family, --max-new-tokens 1536) |
| `evaluation/predictions_qwen_constrained.jsonl` | Predicciones generándose ahora |
| `generation/tritonbench_constrained_decoding.py` | Prompt builder (instruction + PyTorch code + few-shot) |
| `grammars/few_shot_examples.py` | Ejemplos few-shot por familia (elementwise, reduction, matmul) |
| `extras/TritonBench4Modal-main/modal_app.py` | Evaluador en Modal (usar `evaluate_only`) |
| `UPDATE.md` | Donde pegar outputs del evaluador oficial |
| `.venv311/` | Entorno virtual con CUDA para generación |

---

## Prompt que usa el modelo

Estructura de cada prompt (en `generation/tritonbench_constrained_decoding.py`):
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

Now generate the Triton kernel:
```

---

## Qué NO hacer
- No tocar nada en `extras/` (TritonBench4Modal-main, TritonBench-main)
- No usar Python 3.14 para generación (no tiene CUDA)
- No correr `--resume` si la gramática cambió (el archivo se sobreescribe limpio)
- No agregar few-shot examples sin ejecutarlos primero para validar que corren

---

## Contexto del proyecto
- Proyecto académico comparando modelos (Qwen local, GPT-4o, Claude) generando kernels Triton
- Baseline de Qwen ya existe: `results/tritonbench_qwen_baseline.json`
- El constrained decoding es el diferenciador experimental vs el baseline
- Evaluación en 3 fases: call_acc → execution_acc → speedup vs PyTorch
