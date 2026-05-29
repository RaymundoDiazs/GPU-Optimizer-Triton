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


def load_already_done(out_path: Path) -> set:
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

    if not PROMPT_PATH.exists():
        print(f"ERROR: No se encuentra {PROMPT_PATH}", file=sys.stderr)
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
