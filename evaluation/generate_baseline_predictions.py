"""
Paso 2 — Script maestro de generación baseline.

Uso:
    python evaluation/generate_baseline_predictions.py --provider qwen
    python evaluation/generate_baseline_predictions.py --provider gpt4o
    python evaluation/generate_baseline_predictions.py --provider claude
    python evaluation/generate_baseline_predictions.py --provider qwen --resume

Genera evaluation/predictions_{provider}.jsonl con 166 entradas.
Cada entrada: {"instruction": "...", "predict": "<código limpio sin fences>"}
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.prediction_schema import (
    PROVIDER_MODELS,
    load_prediction_jsonl,
    make_prediction_record,
    write_prediction_jsonl,
)

SIMP_PATH  = ROOT / "extras/TritonBench-main/data/TritonBench_T_simp_alpac_v1.json"
T_PATH     = ROOT / "extras/TritonBench-main/data/TritonBench_T_v1.jsonl"
PY_DIR     = ROOT / "extras/TritonBench-main/data/TritonBench_T_v1"
OUTPUT_DIR = ROOT / "evaluation"

PROVIDERS = {
    "qwen":   {"model": "qwen2.5-coder:1.5b"},
    "gpt4o":  {"model": "gpt-4o"},
    "claude": {"model": "claude-haiku-4-5-20251001"},
}

GEN_PARAMS = {
    "temperature": 0.15,
    "top_p":       0.95,
    "seed":        42,
    "max_tokens":  4096,
}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_datasets():
    with open(SIMP_PATH, encoding="utf-8") as f:
        simp_data = json.load(f)
    with open(T_PATH, encoding="utf-8") as f:
        t_data = json.load(f)

    if len(simp_data) != len(t_data):
        print(f"ERROR: longitudes distintas ({len(simp_data)} vs {len(t_data)})", file=sys.stderr)
        sys.exit(1)

    return simp_data, t_data


# ---------------------------------------------------------------------------
# Construcción de prompt
# ---------------------------------------------------------------------------

def load_pytorch_func(file: str) -> str:
    """Lee el archivo .py del operador y devuelve solo la función (sin test driver)."""
    path = PY_DIR / file
    with open(path, encoding="utf-8") as f:
        full = f.read()
    return re.split(r'#{10,}', full)[0].strip()


def build_prompt(instruction: str, file: str) -> str:
    func_code = load_pytorch_func(file)
    return (
        instruction
        + "\n\nPyTorch reference implementation:\n```python\n"
        + func_code
        + "\n```\n\nNow generate the Triton kernel:"
    )


# ---------------------------------------------------------------------------
# Extracción de código limpio
# ---------------------------------------------------------------------------

def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    # Sin fence de cierre (respuesta truncada) — quitar solo la apertura
    text = re.sub(r"^```(?:python|py)?\s*\n?", "", text.strip())
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# Clientes de modelo
# ---------------------------------------------------------------------------

def call_qwen(prompt: str) -> str:
    import urllib.request

    payload = json.dumps({
        "model":  PROVIDERS["qwen"]["model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": GEN_PARAMS["temperature"],
            "top_p":       GEN_PARAMS["top_p"],
            "seed":        GEN_PARAMS["seed"],
            "num_predict": GEN_PARAMS["max_tokens"],
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["response"]


def call_gpt4o(prompt: str, client) -> str:
    response = client.chat.completions.create(
        model=PROVIDERS["gpt4o"]["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=GEN_PARAMS["temperature"],
        top_p=GEN_PARAMS["top_p"],
        seed=GEN_PARAMS["seed"],
        max_tokens=GEN_PARAMS["max_tokens"],
    )
    return response.choices[0].message.content


def call_claude(prompt: str, client) -> str:
    # Anthropic no permite especificar temperature y top_p simultáneamente
    response = client.messages.create(
        model=PROVIDERS["claude"]["model"],
        max_tokens=GEN_PARAMS["max_tokens"],
        temperature=GEN_PARAMS["temperature"],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Lógica de resume
# ---------------------------------------------------------------------------

def load_existing_records(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}
    return {record["task"]["id"]: record for record in load_prediction_jsonl(output_path)}


# ---------------------------------------------------------------------------
# Generación principal
# ---------------------------------------------------------------------------

def generate(provider: str, resume: bool):
    output_path = OUTPUT_DIR / f"predictions_{provider}.jsonl"

    if not resume and output_path.exists():
        print(f"ADVERTENCIA: {output_path} ya existe. Usar --resume para continuar o eliminarlo manualmente.")
        sys.exit(1)

    simp_data, t_data = load_datasets()

    records_by_task: dict[str, dict] = {}
    if resume:
        records_by_task = load_existing_records(output_path)
        print(f"Resume: {len(records_by_task)} operadores ya procesados, saltando...")

    # Inicializar cliente según provider
    client = None
    if provider == "gpt4o":
        try:
            from openai import OpenAI
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("ERROR: OPENAI_API_KEY no encontrada en .env", file=sys.stderr)
                sys.exit(1)
            client = OpenAI(api_key=api_key)
        except ImportError:
            print("ERROR: pip install openai python-dotenv", file=sys.stderr)
            sys.exit(1)

    elif provider == "claude":
        try:
            import anthropic
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("ERROR: ANTHROPIC_API_KEY no encontrada en .env", file=sys.stderr)
                sys.exit(1)
            client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("ERROR: pip install anthropic python-dotenv", file=sys.stderr)
            sys.exit(1)

    elif provider == "qwen":
        # Verificar que Ollama está activo
        import urllib.request
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        except Exception:
            print("ERROR: Ollama no responde en http://localhost:11434. ¿Está corriendo?", file=sys.stderr)
            sys.exit(1)

    total     = len(simp_data)
    processed = 0
    skipped   = 0
    errors    = 0

    for i, (s, t) in enumerate(zip(simp_data, t_data), start=1):
        task_id = f"tritonbench_t_{i:03d}"
        instruction = s["instruction"]

        if task_id in records_by_task:
            skipped += 1
            continue

        prompt = build_prompt(instruction, t["file"])

        try:
            if provider == "qwen":
                raw = call_qwen(prompt)
            elif provider == "gpt4o":
                raw = call_gpt4o(prompt, client)
                time.sleep(0.5)
            elif provider == "claude":
                raw = call_claude(prompt, client)
                time.sleep(0.5)

            code = extract_code(raw)

            task = {
                "id": task_id,
                "benchmark": "tritonbench_t",
                "name": t["name"],
                "file": t["file"],
                "instruction": instruction,
                "expected_output": s.get("output", ""),
            }
            records_by_task[task_id] = make_prediction_record(
                model=PROVIDER_MODELS[provider],
                task=task,
                output=code,
                mode="baseline",
                sample_index=1,
                prompt=prompt,
                source_index=i,
            )
            write_prediction_jsonl(
                output_path,
                [records_by_task[key] for key in sorted(records_by_task)],
            )

            processed += 1
            print(f"[{i:3d}/{total}] OK — {t['name']}")

        except Exception as e:
            errors += 1
            print(f"[{i:3d}/{total}] ERROR — {t['name']}: {e}", file=sys.stderr)

    if records_by_task:
        write_prediction_jsonl(
            output_path,
            [records_by_task[key] for key in sorted(records_by_task)],
        )

    print(f"\nFinalizado — provider={provider}")
    print(f"  Procesados:  {processed}")
    print(f"  Saltados:    {skipped}")
    print(f"  Errores:     {errors}")
    print(f"  Output:      {output_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generar predicciones baseline para TritonBench")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        required=True,
        help="Modelo a usar: qwen | gpt4o | claude",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanudar generación desde donde se interrumpió",
    )
    args = parser.parse_args()
    generate(args.provider, args.resume)


if __name__ == "__main__":
    main()
