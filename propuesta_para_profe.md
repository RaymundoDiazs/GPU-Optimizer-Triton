# Plan de implementación — PyTorch → Triton Translation Benchmark

**Equipo:** Dilan Ocampo · Ray · Charlie · German  
**Fecha:** 2026-05-28  
**Versión:** para revisión del profesor y coordinación interna del equipo

---

## Qué estamos construyendo y por qué

El reto pregunta: **¿qué tan bien pueden los LLMs traducir código PyTorch a kernels Triton, y qué técnicas mejoran eso?**

Para responderlo necesitamos producir exactamente **4 archivos** y evaluarlos en GPU:

```
predictions_qwen_baseline.jsonl        ← Qwen sin restricciones
predictions_gpt4o_baseline.jsonl       ← GPT-4o sin restricciones
predictions_claude_baseline.jsonl      ← Claude sin restricciones
predictions_qwen_constrained.jsonl     ← Qwen con XGrammar activo
```

Cada archivo tiene 166 líneas — una por operador del benchmark TritonBench-T. Cada línea es:
```json
{"instruction": "<descripción del operador>", "predict": "<kernel Triton generado>"}
```

Los 4 archivos se evalúan con el mismo pipeline en GPU real (Modal T4), produciendo 3 métricas por archivo:
- **call@1** — ¿el kernel compila y corre sin error?
- **exe@1** — ¿produce el mismo resultado numérico que PyTorch?
- **speedup** — ¿es más rápido que PyTorch? (solo si pasa exe@1)

Con esos números respondemos las 3 preguntas del análisis final:
1. ¿Qué modelo genera mejor código Triton? (comparar los 3 baselines)
2. ¿El constrained decoding mejora la calidad? (Qwen baseline vs Qwen constrained)
3. ¿La habilidad general de programación (HumanEval) predice el desempeño en Triton?

---

## Los 3 niveles de "restricción" que vamos a comparar

Esto es importante que todo el equipo entienda antes de empezar:

### Nivel 1 — Generación libre (baseline)
El modelo genera lo que quiere. Sin restricciones. Así funcionan los 3 modelos en su baseline.

### Nivel 2 — Structured prompting (GPT-4o y Claude)
No es constrained decoding real. Es pedirle al modelo mediante el prompt que siga una estructura. El modelo *intenta* cumplirla pero puede fallar. Se logra con few-shot examples, instrucciones explícitas en el system prompt, o prefill del assistant turn (solo Claude).

**Por qué no se puede hacer constrained decoding real con GPT-4o y Claude:** estas son APIs externas. El modelo vive en los servidores de OpenAI/Anthropic y solo te devuelven texto. No hay forma de acceder a los logits internos, que es lo que XGrammar necesita para aplicar la máscara.

### Nivel 3 — Constrained decoding real (solo Qwen)
XGrammar intercepta los logits en cada paso de decodificación y pone `-inf` a todos los tokens que violarían la gramática. El modelo **físicamente no puede** generar un token inválido — no es que intente seguir las reglas, es que las reglas están hardcodeadas en el proceso de generación.

```
Generación normal:    logits → softmax → token (puede ser inválido)
Con XGrammar:         logits → máscara grammar → softmax → token (siempre válido)
```

Esto solo es posible con Qwen porque lo cargamos localmente con HuggingFace, lo que nos da acceso directo al vector de logits.

**La pregunta de investigación interesante que emerge:** ¿un modelo pequeño (Qwen 1.5B) con constrained decoding real supera a modelos grandes (GPT-4o, Claude) que solo tienen structured prompting?

---

## Quién hace qué y cómo se conecta todo

### Dilan — Generación baseline

**Responsabilidad:** producir los 3 archivos baseline con el formato exacto que espera el evaluador.

**Qwen (local):**
- Script: `evaluation/generate_tritonbench_predictions.py`
- Modelo: `qwen2.5-coder:1.5b` vía Ollama
- Dataset: `data/TritonBench_T_simp_alpac_v1.json` (166 operadores)
- Output: `evaluation/predictions_qwen.jsonl` ✅ Ya generado

**GPT-4o y Claude (Modal):**
- Herramienta: `extras/TritonBench4Modal-main/modal_app.py`
- Genera Y evalúa en una sola corrida en GPU
- Comandos:
  ```bash
  python scripts/run_modal.py --provider openai --model gpt-4o
  python scripts/run_modal.py --provider anthropic --model claude-haiku-4-5-20251001
  ```
- Output: `evaluation/predictions_gpt4o.jsonl` y `evaluation/predictions_claude.jsonl`

**HumanEval (complementario):**
- Medir pass@1 para los 3 modelos siguiendo `extras/gemma2_humaneval_lab.ipynb`
- Qwen ya documentado (~44%). GPT-4o y Claude pendientes.

**⚠️ Regla crítica:** `extras/TritonBench4Modal-main/` no se toca. Es la herramienta estándar que usa toda la escuela. El único cambio permitido es `PROMPT_HEADER` — nuestro diferenciador legítimo.

---

### Ray — Gramática general de Triton

**Responsabilidad:** entregar un string EBNF que describa qué es un kernel Triton válido. Este string es lo único que Charlie necesita para conectar XGrammar.

**Lo que la gramática DEBE capturar (estructural, no específico a un operador):**
- `@triton.jit` presente
- `def <cualquier_nombre>(<cualquier_parámetro>):`
- Al menos un `tl.program_id(...)`
- Al menos un `tl.load(...)`
- Al menos un `tl.store(...)`
- Python sintácticamente válido (paréntesis balanceados, indentación correcta, sin tokens ilegales)

**Lo que la gramática NO debe hardcodear:**
- El nombre de la función
- Los nombres de parámetros
- Las variables internas
- Qué operaciones `tl.*` específicas se usan (hay kernels válidos que usan `tl.sum`, `tl.where`, `tl.broadcast_to`, etc.)

**Cómo verificar que funciona — los datasets ya están en el repo:**
```
extras/student_package/datasets/curated_100.jsonl    ← 100 kernels válidos → todos deben ser ACEPTADOS
extras/student_package/datasets/adversarial_100.jsonl ← 100 kernels inválidos → todos deben ser RECHAZADOS
```

Los inválidos del adversarial tienen errores como: tokens ilegales (`$`, `?`), paréntesis sin cerrar, números malformados. La gramática debe detectarlos.

**Base de partida:** `grammars/vector_add.ebnf` ya existe pero es demasiado específica. Hay que generalizar quitando los nombres hardcodeados.

**Entregable concreto:** un string EBNF en `grammars/triton_general.ebnf` que pase los 200 casos de validación.

---

### Charlie — Constrained decoding con XGrammar

**Responsabilidad:** conectar XGrammar con Qwen usando HuggingFace y producir `predictions_qwen_constrained.jsonl`.

**Cómo funciona según el notebook de clase (`extras/xgrammar_hands_on.ipynb`):**

```python
import xgrammar as xgr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# 1. Cargar Qwen con HuggingFace (NO con Ollama)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")

# 2. Compilar la gramática de Ray
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
compiler = xgr.GrammarCompiler(tokenizer_info)
compiled = compiler.compile_grammar(TRITON_EBNF)  # string de Ray

# 3. Generar con restricción activa — una línea extra
logits_processor = xgr.contrib.hf.LogitsProcessor(compiled)
output = model.generate(**inputs, logits_processor=[logits_processor])
```

**Por qué HuggingFace y no Ollama:** Ollama es un servidor REST — te devuelve texto, no logits. XGrammar necesita modificar los logits antes del softmax. Con Ollama eso es imposible. Con HuggingFace tienes acceso directo al proceso de generación.

**Lo que ya existe:** `generation/xgrammar_llm_decoder.py` ya tiene la estructura correcta con `LogitsProcessor`. El cambio principal es reemplazar la llamada a Ollama por `AutoModelForCausalLM.from_pretrained`.

**Estrategia para no quedarse bloqueado esperando a Ray:** empezar con una gramática mínima propia (solo `@triton.jit` + `def` + `tl.program_id`) para verificar que el pipeline funciona end-to-end. Reemplazar por la gramática completa de Ray cuando esté lista.

**Entregable concreto:** `evaluation/predictions_qwen_constrained.jsonl` — 166 líneas, mismo formato que los otros archivos.

---

### German — Evaluación en GPU

**Responsabilidad:** convertir cualquier archivo `.jsonl` de predicciones en métricas reales (call@1, exe@1, speedup) usando GPU T4 en Modal.

**Setup inicial (hacer esto primero, el día 1):**
```bash
pip install modal
modal setup          # abre browser para autenticarse
modal secret list    # verificar que tritonbench-llm está creado
```

**Cómo evaluar un archivo:**
```bash
# Desde extras/TritonBench4Modal-main/
modal run modal_app.py::evaluate_only \
    --predictions ../../evaluation/predictions_qwen.jsonl
```

O con el helper que ya existe:
```bash
python scripts/run_modal.py --evaluate-only \
    --predictions evaluation/predictions_qwen.jsonl
```

**Los 4 archivos a evaluar (en orden de llegada):**
1. `predictions_qwen.jsonl` — disponible ahora
2. `predictions_gpt4o.jsonl` — cuando Dilan lo genere
3. `predictions_claude.jsonl` — cuando Dilan lo genere
4. `predictions_qwen_constrained.jsonl` — cuando Charlie lo genere

**Resultado esperado por archivo:**
```json
{
  "phase1_call_acc": {"rate": 0.XX},
  "phase2_exec_acc": {"rate": 0.XX},
  "phase3_efficiency": {"speedup_vs_pytorch": X.XX}
}
```

Guardar cada resultado en `results/` con nombre claro: `tritonbench_qwen_baseline.json`, etc.

---

## Plan de trabajo día a día

La prioridad es eliminar dependencias en cadena. Cada persona debe poder avanzar sin esperar a otra.

### Día 1 — Todo el mundo en paralelo

| Persona | Tarea | Por qué es urgente |
|---|---|---|
| **German** | Configurar Modal y correr `evaluate_only` con `predictions_qwen.jsonl` | Sin esto no hay ningún resultado real. Es el cuello de botella de todo |
| **Dilan** | Correr GPT-4o y Claude en Modal | Mientras German configura, Dilan genera los otros 2 baselines |
| **Ray** | Leer `curated_100.jsonl` y `adversarial_100.jsonl`, identificar patrones | Entender qué tienen en común los 100 válidos antes de escribir la gramática |
| **Charlie** | Cargar Qwen con HuggingFace y verificar que genera texto | No esperar a Ray — primero confirmar que el modelo carga y el pipeline funciona |

### Día 2

| Persona | Tarea |
|---|---|
| **German** | Evalúa `predictions_gpt4o.jsonl` y `predictions_claude.jsonl` |
| **Ray** | Primera versión de gramática general, validar contra `curated_100.jsonl` |
| **Charlie** | Integrar XGrammar con gramática mínima propia, verificar que restringe tokens |
| **Dilan** | HumanEval para GPT-4o y Claude |

### Día 3

| Persona | Tarea |
|---|---|
| **German** | Evalúa `predictions_qwen_constrained.jsonl` cuando llegue |
| **Ray** | Validar gramática contra `adversarial_100.jsonl`, ajustar hasta 100/100 |
| **Charlie** | Reemplazar gramática mínima por la de Ray, correr 166 operadores completos |
| **Dilan** | Armar tabla comparativa con todos los resultados |

### Día 4 — Análisis y video

Con los 4 JSONs de resultados en `results/`, todos contribuyen al análisis:
- Tabla: call@1 / exe@1 / speedup × 4 condiciones
- ¿Mejora el constrained decoding el call@1 de Qwen?
- ¿Predice HumanEval el desempeño en Triton?
- Video en inglés, ~2 min por persona

---

## Dependencias críticas

```
German configura Modal
    └─→ primeros números reales de Qwen (desbloqueado hoy)

Dilan corre GPT-4o y Claude
    └─→ German evalúa esos 2 archivos

Ray entrega gramática general
    └─→ Charlie conecta con XGrammar
        └─→ predictions_qwen_constrained.jsonl
            └─→ German evalúa
                └─→ comparación baseline vs constrained

Todo lo anterior
    └─→ análisis estadístico + video
```

**El único bloqueo real:** si Ray tarda, Charlie puede empezar con gramática mínima propia. Si German no configura Modal el día 1, todo el análisis final se atrasa.

---

## Preguntas que necesitamos que el profesor aclare

1. **Dataset:** el input de `TritonBench_T_simp_alpac_v1.json` es una descripción funcional en texto, no código PyTorch fuente. ¿Es el dataset correcto o deberíamos usar la versión completa con código fuente?

2. **Constrained decoding en GPT-4o y Claude:** técnicamente no es posible hacer constrained decoding real (token masking) con APIs externas porque no exponen logits. Lo que sí se puede hacer es structured prompting (few-shot, prefill). ¿Queremos incluir eso como una condición adicional de comparación, o el alcance es solo Qwen con XGrammar?

3. **HumanEval:** ¿es obligatorio calcularlo para los 3 modelos o podemos citar valores de literatura para GPT-4o y Claude?

4. **Fine-tuning:** vemos que otros equipos lo están usando. Según entendemos el reto, la técnica adicional a explorar es constrained decoding. ¿El fine-tuning está dentro del scope esperado?
