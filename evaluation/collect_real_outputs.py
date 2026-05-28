"""
collect_real_outputs.py
-----------------------
Recopila outputs reales de los modelos configurados en config/model_eval.yaml.

Modos de tarea:
  --task-type generation   Pide al modelo generar un kernel Triton desde cero.
                           Usa prompts/kernel_generation_prompt.txt
                           Guarda en evaluation/real_outputs.jsonl

  --task-type translation  Pide al modelo traducir código PyTorch a Triton.
                           Usa prompts/pytorch_to_triton_prompt.txt
                           Lee ejemplos de data/pytorch_examples.json
                           Guarda en evaluation/translation_outputs.jsonl

Uso:
    python evaluation/collect_real_outputs.py --provider ollama --task-type translation
    python evaluation/collect_real_outputs.py --all --task-type translation
    python evaluation/collect_real_outputs.py --all --task-type generation
    python evaluation/collect_real_outputs.py --all   (equivalente a generation)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from generation.hf_xgrammar_provider import call_hf_xgrammar

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "model_eval.yaml"

# Modo generation
PROMPT_TEMPLATE  = REPO_ROOT / "prompts" / "kernel_generation_prompt.txt"
OUTPUT_PATH      = REPO_ROOT / "evaluation" / "real_outputs.jsonl"

# Modo translation
TRANSLATION_PROMPT   = REPO_ROOT / "prompts" / "pytorch_to_triton_prompt.txt"
TRANSLATION_EXAMPLES = REPO_ROOT / "data" / "pytorch_examples.json"
TRANSLATION_OUTPUT   = REPO_ROOT / "evaluation" / "translation_outputs.jsonl"


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


def load_pytorch_examples() -> list:
    """Carga los ejemplos de código PyTorch desde data/pytorch_examples.json."""
    with TRANSLATION_EXAMPLES.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(task: dict, mode: str) -> str:
    """Construye el prompt de generación desde cero (kernel_generation_prompt.txt)."""
    template = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    prompt = template.format(task_prompt=task["prompt"])
    if mode == "constrained":
        prompt += (
            "\n\nConstrained decoding contract:\n"
            "Generate only code matching the project grammar for a vector-add Triton kernel.\n"
            "Allowed kernel body pattern: compute offsets, mask, two tl.load calls, one tl.store call.\n"
        )
    return prompt


def build_translation_prompt(example: dict) -> str:
    """Construye el prompt de traducción PyTorch→Triton (pytorch_to_triton_prompt.txt)."""
    template = TRANSLATION_PROMPT.read_text(encoding="utf-8")
    return template.format(
        pytorch_code=example["pytorch_code"],
        operation_description=example["operation_description"],
    )


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

def collect_for_model_generation(model: dict, tasks: list, modes: list, samples: int, out_file) -> int:
    """Modo generation: genera un kernel Triton desde cero para cada tarea.
    Devuelve cantidad de registros escritos.

    En modo constrained para Qwen local, usa XGrammar real mediante
    HuggingFace + logits processor. En los demás casos usa el provider normal.
    """
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
                print(
                    f"  [{model['id']}] task={task['id']} mode={mode} sample={sample_index} ... ",
                    end="",
                    flush=True,
                )

                uses_real_xgrammar = (
                    mode == "constrained"
                    and model["id"] == "small_qwen25_coder_1_5b"
                )

                if uses_real_xgrammar:
                    output, latency = call_hf_xgrammar(prompt)
                else:
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
                    "constrained_decoding_backend": (
                        "xgrammar_hf" if uses_real_xgrammar else "provider_prompt"
                    ),
                }
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
                count += 1

    return count

def collect_for_model_translation(model: dict, examples: list, samples: int, out_file) -> int:
    """Modo translation: pide al modelo traducir cada ejemplo PyTorch a Triton.
    Solo usa modo baseline (sin constrained) porque el prompt ya es específico.
    Devuelve cantidad de registros escritos."""
    provider = model["provider"]
    if provider not in PROVIDER_FN:
        print(f"  Provider '{provider}' no soportado. Saltando {model['id']}.")
        return 0

    call_fn = PROVIDER_FN[provider]
    model_name = model.get("model_name", model["id"])
    count = 0

    for example in examples:
        for sample_index in range(1, samples + 1):
            prompt = build_translation_prompt(example)
            print(f"  [{model['id']}] example={example['id']} sample={sample_index} ... ", end="", flush=True)
            output, latency = call_fn(model_name, prompt)
            print(f"OK ({latency:.1f}s)")

            record = {
                "model": {
                    "id": model["id"],
                    "display_name": model.get("display_name", model["id"]),
                    "tier": model.get("tier", "unknown"),
                    "provider": provider,
                },
                "task": {
                    "id": example["id"],
                    "pytorch_code": example["pytorch_code"],
                    "operation_description": example["operation_description"],
                    "expected_kernel_name": example.get("expected_kernel_name", ""),
                    "expected_terms": example.get("expected_terms", []),
                },
                "mode": "translation",
                "sample_index": sample_index,
                "prompt": prompt,
                "output": output,
                "latency_seconds": round(latency, 6),
            }
            out_file.write(json.dumps(record) + "\n")
            out_file.flush()
            count += 1

    return count


def run(providers_filter: list[str] | None = None, task_type: str = "generation") -> None:
    config = load_config()
    experiment = config.get("experiment", {})
    models = config.get("models", [])
    modes = config.get("modes", ["baseline", "constrained"])
    samples = int(experiment.get("samples_per_model", 3))

    if providers_filter:
        models = [m for m in models if m.get("provider") in providers_filter]

    if not models:
        print("No hay modelos que coincidan con el filtro de provider.")
        sys.exit(1)

    if task_type == "translation":
        # Prefiere translation_examples_file del yaml si existe, si no usa la constante por defecto
        examples_file = experiment.get("translation_examples_file")
        if examples_file:
            from pathlib import Path as _P
            _ep = _P(examples_file)
            if not _ep.is_absolute():
                _ep = REPO_ROOT / _ep
            with _ep.open("r", encoding="utf-8") as _f:
                examples = json.load(_f)
        else:
            examples = load_pytorch_examples()
        out_path = TRANSLATION_OUTPUT
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mode_open = "a" if out_path.exists() else "w"
        total = 0

        print(f"\n=== Modo: TRADUCCION PyTorch -> Triton ===")
        print(f"Ejemplos: {[e['id'] for e in examples]}")
        print(f"Modelos:  {[m['id'] for m in models]}")
        print(f"Muestras por modelo por ejemplo: {samples}\n")

        with out_path.open(mode_open, encoding="utf-8") as out_file:
            for model in models:
                print(f"\n>> Modelo: {model.get('display_name', model['id'])} [{model['provider']}]")
                count = collect_for_model_translation(model, examples, samples, out_file)
                total += count

        print(f"\nListo. {total} registros guardados en {out_path}")
        print("Siguiente paso:")
        print("  python evaluation/model_evaluation.py --manual-outputs evaluation/translation_outputs.jsonl")

    else:  # generation
        tasks = load_tasks(experiment.get("task_file", "data/kernel_generation_tasks.json"))
        out_path = OUTPUT_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mode_open = "a" if out_path.exists() else "w"
        total = 0

        print(f"\n=== Modo: GENERACIÓN desde cero ===")
        print(f"Tareas:  {[t['id'] for t in tasks]}")
        print(f"Modelos: {[m['id'] for m in models]}")
        print(f"Modos:   {modes}")
        print(f"Muestras por modelo: {samples}\n")

        with out_path.open(mode_open, encoding="utf-8") as out_file:
            for model in models:
                print(f"\n>> Modelo: {model.get('display_name', model['id'])} [{model['provider']}]")
                count = collect_for_model_generation(model, tasks, modes, samples, out_file)
                total += count

        print(f"\nListo. {total} registros guardados en {out_path}")
        print("Siguiente paso:")
        print("  python evaluation/model_evaluation.py --manual-outputs evaluation/real_outputs.jsonl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recopila outputs reales de los modelos.\n"
            "  --task-type generation  → genera kernels Triton desde cero\n"
            "  --task-type translation → traduce código PyTorch a Triton (objetivo del reto)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--provider", choices=["ollama", "openai", "anthropic"],
                       help="Corre solo el provider indicado")
    group.add_argument("--all", action="store_true",
                       help="Corre todos los providers configurados")

    parser.add_argument(
        "--task-type",
        choices=["generation", "translation"],
        default="generation",
        help=(
            "'generation': pide al modelo generar un kernel desde cero (default). "
            "'translation': pide al modelo traducir código PyTorch a Triton."
        ),
    )

    args = parser.parse_args()
    providers = None if args.all else [args.provider]
    run(providers_filter=providers, task_type=args.task_type)


if __name__ == "__main__":
    main()
