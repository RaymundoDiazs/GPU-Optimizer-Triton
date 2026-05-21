"""
collect_real_outputs.py
-----------------------
Recopila outputs reales de los modelos configurados en config/model_eval.yaml
y los guarda en evaluation/real_outputs.jsonl en el formato esperado por
model_evaluation.py --manual-outputs.

Uso:
    python evaluation/collect_real_outputs.py --provider ollama
    python evaluation/collect_real_outputs.py --provider openai
    python evaluation/collect_real_outputs.py --provider anthropic
    python evaluation/collect_real_outputs.py --all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "model_eval.yaml"
PROMPT_TEMPLATE = REPO_ROOT / "prompts" / "kernel_generation_prompt.txt"
OUTPUT_PATH = REPO_ROOT / "evaluation" / "real_outputs.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tasks(task_file: str) -> list:
    path = Path(task_file)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(task: dict, mode: str) -> str:
    template = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    prompt = template.format(task_prompt=task["prompt"])
    if mode == "constrained":
        prompt += (
            "\n\nConstrained decoding contract:\n"
            "Generate only code matching the project grammar for a vector-add Triton kernel.\n"
            "Allowed kernel body pattern: compute offsets, mask, two tl.load calls, one tl.store call.\n"
        )
    return prompt


def get_env(key: str) -> str:
    """Lee una variable de entorno. Intenta cargar .env si no existe."""
    val = os.environ.get(key)
    if val:
        return val
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip()
    return ""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def call_ollama(model_name: str, prompt: str) -> tuple[str, float]:
    """Llama al modelo local via Ollama HTTP API."""
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        print("urllib no disponible", file=sys.stderr)
        sys.exit(1)

    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "seed": 7},
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            latency = time.perf_counter() - start
            return data.get("response", ""), latency
    except Exception as exc:
        print(f"  ERROR Ollama: {exc}", file=sys.stderr)
        print("  ¿Está corriendo Ollama? Prueba: ollama serve", file=sys.stderr)
        sys.exit(1)


def call_openai(model_name: str, prompt: str) -> tuple[str, float]:
    """Llama al modelo via OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Falta openai. Instala con: pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = get_env("OPENAI_API_KEY")
    if not api_key:
        print("No se encontró OPENAI_API_KEY en el entorno ni en .env", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        seed=7,
    )
    latency = time.perf_counter() - start
    return response.choices[0].message.content or "", latency


def call_anthropic(model_name: str, prompt: str) -> tuple[str, float]:
    """Llama al modelo via Anthropic API."""
    try:
        import anthropic
    except ImportError:
        print("Falta anthropic. Instala con: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    api_key = get_env("ANTHROPIC_API_KEY")
    if not api_key:
        print("No se encontró ANTHROPIC_API_KEY en el entorno ni en .env", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    start = time.perf_counter()
    message = client.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - start
    return message.content[0].text if message.content else "", latency


PROVIDER_FN = {
    "ollama": call_ollama,
    "openai": call_openai,
    "anthropic": call_anthropic,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def collect_for_model(model: dict, tasks: list, modes: list, samples: int, out_file) -> int:
    """Genera y guarda outputs para un modelo. Devuelve cantidad de registros escritos."""
    provider = model["provider"]
    if provider not in PROVIDER_FN:
        print(f"  Provider '{provider}' no soportado. Saltando {model['id']}.")
        return 0

    call_fn = PROVIDER_FN[provider]
    model_name = model.get("model_name", model["id"])
    count = 0

    for task in tasks:
        for mode in modes:
            for sample_index in range(1, samples + 1):
                prompt = build_prompt(task, mode)
                print(f"  [{model['id']}] task={task['id']} mode={mode} sample={sample_index} ... ", end="", flush=True)
                output, latency = call_fn(model_name, prompt)
                print(f"OK ({latency:.1f}s)")

                record = {
                    "model": {
                        "id": model["id"],
                        "display_name": model.get("display_name", model["id"]),
                        "tier": model.get("tier", "unknown"),
                        "provider": provider,
                    },
                    "task": task,
                    "mode": mode,
                    "sample_index": sample_index,
                    "prompt": prompt,
                    "output": output,
                    "latency_seconds": round(latency, 6),
                }
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
                count += 1

    return count


def run(providers_filter: list[str] | None = None) -> None:
    config = load_config()
    experiment = config.get("experiment", {})
    tasks = load_tasks(experiment.get("task_file", "data/kernel_generation_tasks.json"))
    models = config.get("models", [])
    modes = config.get("modes", ["baseline", "constrained"])
    samples = int(experiment.get("samples_per_model", 3))

    if providers_filter:
        models = [m for m in models if m.get("provider") in providers_filter]

    if not models:
        print("No hay modelos que coincidan con el filtro de provider.")
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Modo append: si el archivo ya existe, agrega sin borrar lo anterior
    mode_open = "a" if OUTPUT_PATH.exists() else "w"
    total = 0

    with OUTPUT_PATH.open(mode_open, encoding="utf-8") as out_file:
        for model in models:
            print(f"\n>> Modelo: {model.get('display_name', model['id'])} [{model['provider']}]")
            count = collect_for_model(model, tasks, modes, samples, out_file)
            total += count

    print(f"\nListo. {total} registros guardados en {OUTPUT_PATH}")
    print("Siguiente paso:")
    print("  python evaluation/model_evaluation.py --manual-outputs evaluation/real_outputs.jsonl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recopila outputs reales de los modelos y los guarda en real_outputs.jsonl"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--provider", choices=["ollama", "openai", "anthropic"],
                       help="Corre solo el provider indicado")
    group.add_argument("--all", action="store_true",
                       help="Corre todos los providers configurados")

    args = parser.parse_args()

    if args.all:
        run(providers_filter=None)
    else:
        run(providers_filter=[args.provider])


if __name__ == "__main__":
    main()
