# PLAN.md — Generación baseline completa + evaluación con TritonBench4Modal

Fecha: 2026-05-28  
Responsable: Dilan (Ocampo)  
Objetivo: Producir los 3 archivos `predictions_<modelo>.jsonl`, evaluarlos con TritonBench4Modal en GPU, y guardar los resultados.

---

## Contexto: dos flujos, misma evaluación

| Modelo | Generación | Evaluación |
|---|---|---|
| Qwen 1.5B | Local con Ollama → `predictions_qwen.jsonl` | `modal_app.py::evaluate_only --predictions ...` — Modal sube el archivo y corre las 3 fases |
| GPT-4o | `modal_app.py::main --provider openai` — genera y evalúa en una sola corrida | incluida en la misma corrida |
| Claude Haiku | `modal_app.py::main --provider anthropic` — genera y evalúa en una sola corrida | incluida en la misma corrida |

**Por qué Qwen tiene generación separada:** `modal_app.py` llama APIs en la nube. Ollama corre en tu máquina local y Modal no puede alcanzarlo. No hay alternativa mientras uses Ollama. La evaluación GPU es idéntica para los 3 — `modal_app.py` recibe el `.jsonl`, lo sube automáticamente al Volume con `_upload_local_predictions()`, y corre las 3 fases.

**Por qué el script de generación de Qwen no puede reusar `_build_messages()` de `modal_app.py`:** ese archivo tiene decoradores `@app.function` que asumen el entorno Modal — no se puede importar localmente. El script replica exactamente la misma lógica (3 líneas) con el mismo system prompt.

**Regla de oro:** `extras/TritonBench4Modal-main/` **solo se toca para modificar `PROMPT_HEADER` — nada más.**

---

## Checklist completa

### PASO 1 — Descargar dataset `TritonBench_T_simp_alpac_v1.json`

- [x] Descargar el dataset en `data/TritonBench_T_simp_alpac_v1.json`
- [x] Verificar que contiene 166 entradas

```bash
# Desde la raíz del proyecto
curl -L "https://raw.githubusercontent.com/thunlp/TritonBench/main/data/TritonBench_T_simp_alpac_v1.json" \
     -o data/TritonBench_T_simp_alpac_v1.json

# Verificar
python -c "
import json
data = json.load(open('data/TritonBench_T_simp_alpac_v1.json'))
print(f'Total operadores: {len(data)}')  # debe ser 166
print('Primer instruction:', data[0]['instruction'][:80])
"
```

---

### PASO 2 — Crear `evaluation/generate_tritonbench_predictions.py`

- [x] Crear el script con el código de abajo
- [x] Verificar que el script lee `prompts/pytorch_to_triton_prompt.txt` como system prompt

**Por qué un script nuevo y no modificar `collect_real_outputs.py`:** el script existente guarda formato interno con campos extra (`model`, `task`, `mode`, `sample_index`, etc.). `evaluate_only` espera exactamente `{"instruction": "...", "predict": "..."}` y nada más. Modificar el script existente rompe los experimentos propios que ya corren con ese formato.

**Sobre el prompt:** `prompts/pytorch_to_triton_prompt.txt` es el prompt maestro que usamos para los 3 modelos. Incluye reglas obligatorias de Triton, referencia de la API, y dos ejemplos few-shot (vector add y softmax). El placeholder `{instruction}` es donde se inserta cada entrada del dataset Alpaca. Este prompt es el diferenciador del proyecto vs los demás equipos que usan el prompt genérico del benchmark.

**Sobre el campo `input` del dataset:** está vacío en las 166 entradas. Toda la información necesaria para generar el kernel está dentro de `instruction`: una descripción funcional en texto + la firma completa de la función (nombre, parámetros, tipos, shapes). No hay código PyTorch fuente — el modelo debe implementar el kernel Triton desde cero basándose solo en esa descripción. El benchmark evalúa si el kernel producido es equivalente al comportamiento de PyTorch, no si es una traducción literal de código existente.

**CRÍTICO — formato del campo `predict`:** el evaluador de TritonBench (`evaluate_only`) espera **código Python limpio sin fences** en el campo `predict`. Así lo confirma `_extract_code()` en `modal_app.py` (línea 276): extrae el código dentro de los fences y guarda solo el contenido. Si se guardan los fences (` ```python...``` `), el evaluador los interpretará como código Python y fallará con `SyntaxError` en fase 1 (call@1). La función del script debe llamarse `extract_code_clean` y devolver código sin fences.

```python
"""
generate_tritonbench_predictions.py
------------------------------------
Genera predictions_qwen.jsonl para TritonBench4Modal.

Lee TritonBench_T_simp_alpac_v1.json, llama a Qwen vía Ollama,
y guarda en formato exacto que espera evaluate_only.

Uso:
    python evaluation/generate_tritonbench_predictions.py          # todos (166)
    python evaluation/generate_tritonbench_predictions.py --resume # reanudar si se interrumpe
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "TritonBench_T_simp_alpac_v1.json"
OUTPUT_PATH  = REPO_ROOT / "evaluation" / "predictions_qwen.jsonl"
PROMPT_PATH  = REPO_ROOT / "prompts" / "pytorch_to_triton_prompt.txt"
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL_NAME   = "qwen2.5-coder:1.5b"


def build_prompt(item: dict, template: str) -> str:
    """Inserta instruction del dataset en el placeholder {instruction} del prompt maestro."""
    instr = item["instruction"]
    inp   = (item.get("input") or "").strip()
    full_instr = f"{instr}\n\n{inp}" if inp else instr
    return template.replace("{instruction}", full_instr)


def extract_code_clean(text: str) -> str:
    """Extrae código Python limpio SIN fences — formato que espera evaluate_only.

    modal_app.py hace lo mismo en _extract_code(): si se guardan fences,
    el evaluador los interpreta como código y falla con SyntaxError en call@1.
    """
    s = text.strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", s, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    # Sin fence de cierre (respuesta truncada) — quitar solo la apertura
    s = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    return s.strip() + "\n"


def call_ollama(prompt: str) -> str:
    payload = json.dumps({
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,
            "top_p": 0.95,
            "seed": 42,
            "num_predict": 8192,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8")).get("response", "")
    except Exception as exc:
        print(f"\n  ERROR Ollama: {exc}", file=sys.stderr)
        return ""


def load_already_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line.strip())["instruction"])
                except Exception:
                    pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar desde donde se quedó (no sobreescribe)")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"ERROR: No se encuentra {DATASET_PATH}", file=sys.stderr)
        print("Descarga el dataset primero — ver PLAN.md Paso 1", file=sys.stderr)
        sys.exit(1)

    template     = PROMPT_PATH.read_text(encoding="utf-8")
    items        = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    already_done = load_already_done(OUTPUT_PATH) if args.resume else set()

    if args.resume and already_done:
        print(f"Reanudando: {len(already_done)} ya generados, saltando...")

    pending = [it for it in items if it["instruction"] not in already_done]
    total   = len(items)
    done_n  = len(already_done)

    print(f"Dataset: {total} operadores | Pendientes: {len(pending)} | Modelo: {MODEL_NAME}\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and already_done else "w"

    with OUTPUT_PATH.open(mode, encoding="utf-8") as out_f:
        for i, item in enumerate(pending, start=1):
            prompt = build_prompt(item, template)
            print(f"[{done_n + i}/{total}] {item['instruction'][:60]}... ", end="", flush=True)
            t0  = time.perf_counter()
            raw = call_ollama(prompt)
            elapsed = time.perf_counter() - t0

            if not raw:
                predict = "# generation failed\n"
                print(f"FALLO ({elapsed:.1f}s)")
            else:
                predict = extract_code_clean(raw)
                print(f"OK ({elapsed:.1f}s)")

            out_f.write(json.dumps({"instruction": item["instruction"], "predict": predict}) + "\n")
            out_f.flush()

    print(f"\nListo. {done_n + len(pending)} registros en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

---

### PASO 3 — Actualizar `config/model_eval.yaml`

- [x] Actualizar parámetros al estándar del benchmark

```yaml
experiment:
  samples_per_operator: 1
  seed: 42
  temperature: 0.15
  top_p: 0.95
  max_tokens: 8192
  dataset: data/TritonBench_T_simp_alpac_v1.json
```

---

### PASO 4 — Generar `predictions_qwen.jsonl` (166 operadores, Ollama local)

- [x] Verificar que Ollama está corriendo: `ollama list` debe mostrar `qwen2.5-coder:1.5b`
- [x] Correr el script completo
- [x] Verificar conteo final: 166 líneas

```bash
# Correr todos los 166 (toma ~30-90 min con Qwen 1.5B)
python evaluation/generate_tritonbench_predictions.py

# Si se interrumpe, reanudar sin perder progreso:
python evaluation/generate_tritonbench_predictions.py --resume

# Verificar conteo
python -c "print(sum(1 for _ in open('evaluation/predictions_qwen.jsonl')))"
# Debe imprimir: 166
```

---

### PASO 5 — Configurar Modal y crear el secret con las API keys

**Cuándo hacerlo:** mientras el Paso 4 (generación de Qwen) corre en paralelo — toma tiempo y puedes aprovechar para dejar Modal listo.

- [ ] Crear cuenta en modal.com si no tienes una
- [ ] Instalar Modal: `pip install modal`
- [ ] Autenticar: `modal setup` (abre browser para login/registro)
- [ ] Crear el secret con ambas API keys
- [ ] Verificar que el secret existe: `modal secret list`

```bash
pip install modal
modal setup

# Crear el secret con el nombre exacto que espera modal_app.py
modal secret create tritonbench-llm \
    OPENAI_API_KEY=sk-... \
    ANTHROPIC_API_KEY=sk-ant-...

# Verificar
modal secret list
```

**Costo estimado:**
- GPT-4o (166 operadores): ~$2–5 en API de OpenAI
- Claude Haiku (166 operadores): ~$0.50–1 en API de Anthropic
- GPU T4 en Modal (evaluación, ~1h por run): ~$0.59/h → ~$2–3 por corrida

---

### PASO 6 — Crear `scripts/run_modal.py` (helper para capturar resultados)

- [ ] Crear el script con el código de abajo

`modal run` mezcla logs de progreso con el JSON final en stdout. Este script corre Modal como subproceso, extrae el JSON del output, y lo guarda automáticamente en `results/`.

```python
"""
scripts/run_modal.py
---------------------
Corre modal_app.py y guarda el summary JSON en results/ automáticamente.

Uso:
    python scripts/run_modal.py --provider openai --model gpt-4o
    python scripts/run_modal.py --provider anthropic --model claude-haiku-4-5-20251001
    python scripts/run_modal.py --evaluate-only --predictions evaluation/predictions_qwen.jsonl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[1]
MODAL_DIR  = REPO_ROOT / "extras" / "TritonBench4Modal-main"
RESULTS_DIR = REPO_ROOT / "results"


def extract_json(output: str) -> dict:
    """Extrae el último bloque JSON válido del output de modal run."""
    lines = output.strip().splitlines()
    # El JSON empieza en la primera línea que es "{"
    json_lines = []
    in_json = False
    for line in lines:
        if line.strip() == "{":
            in_json = True
            json_lines = []
        if in_json:
            json_lines.append(line)
        if in_json and line.strip() == "}":
            try:
                return json.loads("\n".join(json_lines))
            except json.JSONDecodeError:
                json_lines = []
                in_json = False
    raise ValueError("No se encontró JSON válido en el output de Modal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "anthropic"])
    parser.add_argument("--model")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--predictions")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.evaluate_only:
        assert args.predictions, "--predictions requerido con --evaluate-only"
        predictions_path = REPO_ROOT / args.predictions
        cmd = [
            "modal", "run", "modal_app.py::evaluate_only",
            "--predictions", str(predictions_path),
        ]
        out_file = RESULTS_DIR / "tritonbench_qwen_baseline.json"
    else:
        assert args.provider and args.model, "--provider y --model requeridos"
        cmd = [
            "modal", "run", "modal_app.py", "--",
            "--provider", args.provider,
            "--model", args.model,
        ]
        name = "gpt4o" if "gpt" in args.model else "claude"
        out_file = RESULTS_DIR / f"tritonbench_{name}_baseline.json"

    print(f"Corriendo: {' '.join(cmd)}")
    print(f"Guardando resultado en: {out_file}\n")

    result = subprocess.run(
        cmd, cwd=str(MODAL_DIR),
        capture_output=False,   # muestra progreso en tiempo real
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Mostrar output completo en consola
    print(result.stdout)

    # Extraer y guardar solo el JSON
    try:
        summary = extract_json(result.stdout)
        out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nResultado guardado en {out_file}")
    except ValueError as e:
        print(f"\nERROR extrayendo JSON: {e}", file=sys.stderr)
        print("Output completo guardado para inspección manual.")
        out_file.with_suffix(".txt").write_text(result.stdout, encoding="utf-8")


if __name__ == "__main__":
    main()
```

---

### PASO 7 — Generar y evaluar GPT-4o

- [ ] Correr `run_modal.py` con provider openai
- [ ] Verificar que `results/tritonbench_gpt4o_baseline.json` fue creado
- [ ] Descargar `predictions_gpt4o.jsonl` del Modal Volume

```bash
# Desde la raíz del proyecto — genera 166 kernels Y evalúa en GPU, guarda resultado automáticamente
python scripts/run_modal.py --provider openai --model gpt-4o

# Descargar predictions del volume
modal volume get tritonbench-t-data predictions/openai_gpt-4o_simp.jsonl \
    evaluation/predictions_gpt4o.jsonl
```

---

### PASO 8 — Generar y evaluar Claude Haiku

- [ ] Correr `run_modal.py` con provider anthropic
- [ ] Verificar que `results/tritonbench_claude_baseline.json` fue creado
- [ ] Descargar `predictions_claude.jsonl` del Modal Volume

```bash
python scripts/run_modal.py --provider anthropic --model claude-haiku-4-5-20251001

modal volume get tritonbench-t-data predictions/anthropic_claude-haiku-4-5-20251001_simp.jsonl \
    evaluation/predictions_claude.jsonl
```

---

### PASO 9 — Evaluar Qwen con `evaluate_only`

- [ ] Correr `run_modal.py --evaluate-only` con el predictions de Qwen
- [ ] Verificar que `results/tritonbench_qwen_baseline.json` fue creado

```bash
python scripts/run_modal.py --evaluate-only \
    --predictions evaluation/predictions_qwen.jsonl
```

**Resultado esperado en los 3 archivos JSON:**
- `phase1_call_acc.rate` — % de kernels que compilaron sin error
- `phase2_exec_acc.rate` — % de kernels numéricamente correctos vs PyTorch
- `phase3_efficiency.speedup_vs_pytorch` — geometric mean del speedup

---

### PASO 9 — HumanEval pass@k para los 3 modelos

- [ ] Generar N muestras por problema para GPT-4o (`temperature=0.8` para diversidad)
- [ ] Generar N muestras por problema para Claude Haiku
- [ ] Calcular pass@1 y pass@k con el estimador unbiasado (ver `extras/gemma2_humaneval_lab.ipynb`)
- [ ] Documentar en `evaluation/humaneval_context.md` junto al resultado de Qwen (~44%)

```bash
pip install human-eval
# Ver extras/gemma2_humaneval_lab.ipynb para la metodología exacta
# Necesitas N≥k muestras por problema para el estimador unbiasado
```

**Resultado esperado:**
| Modelo | pass@1 |
|---|---|
| Qwen 1.5B | ~44% (ya documentado) |
| GPT-4o | ~90% |
| Claude Haiku 4.5 | ~88% |

---

## Estado final cuando todo esté completo

Con los pasos 1–8 terminados tendrás:
- `evaluation/predictions_qwen.jsonl` — 166 kernels generados por Qwen
- `evaluation/predictions_gpt4o.jsonl` — 166 kernels generados por GPT-4o
- `evaluation/predictions_claude.jsonl` — 166 kernels generados por Claude Haiku
- `results/tritonbench_qwen_baseline.json` — call@1, exe@1, speedup para Qwen
- `results/tritonbench_gpt4o_baseline.json` — call@1, exe@1, speedup para GPT-4o
- `results/tritonbench_claude_baseline.json` — call@1, exe@1, speedup para Claude Haiku

Esto cubre el **100% de la parte de generación y evaluación baseline**. Lo que queda fuera de este plan (gramática de Ray, constrained decoding de Charlie) es trabajo de los otros integrantes del equipo.

---

## Lo que queda fuera de este plan

| Parte | Responsable |
|---|---|
| Gramática EBNF general | Ray |
| `get_valid_next_tokens()` | Ray |
| Constrained decoding con XGrammar | Charlie |
| `predictions_qwen_constrained.jsonl` | Charlie |
