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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.hf_xgrammar_provider import call_hf_xgrammar
from generation.tritonbench_constrained_decoding import build_tritonbench_constrained_spec
from evaluation.prediction_schema import make_prediction_record

CONFIG_PATH = REPO_ROOT / "config" / "model_eval.yaml"

# Modo generation
PROMPT_TEMPLATE  = REPO_ROOT / "prompts" / "kernel_generation_prompt.txt"
OUTPUT_PATH      = REPO_ROOT / "evaluation" / "real_outputs.jsonl"
QWEN_CONSTRAINED_OUTPUT_PATH = REPO_ROOT / "evaluation" / "predictions_qwen_constrained.jsonl"

# Modo translation
TRANSLATION_PROMPT   = REPO_ROOT / "prompts" / "pytorch_to_triton_prompt.txt"
TRANSLATION_EXAMPLES = REPO_ROOT / "data" / "pytorch_examples.json"
TRANSLATION_OUTPUT   = REPO_ROOT / "evaluation" / "translation_outputs.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    import yaml

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
    """Construye el prompt de traducción PyTorch→Triton.

    Soporta dos formatos:
    - ejemplos locales con pytorch_code/operation_description;
    - ejemplos TritonBench-T Alpaca con instruction/input/output.
    """
    template = TRANSLATION_PROMPT.read_text(encoding="utf-8")
    if "instruction" in example:
        instruction = example["instruction"].strip()
        extra_input = (example.get("input") or "").strip()
        if extra_input:
            instruction = f"{instruction}\n\n{extra_input}"
        return template.format(instruction=instruction)

    instruction = (
        f"Functional description: {example['operation_description']}\n\n"
        "PyTorch reference implementation:\n"
        f"```python\n{example['pytorch_code']}\n```"
    )
    return template.format(
        instruction=instruction,
        pytorch_code=example["pytorch_code"],
        operation_description=example["operation_description"],
    )


def build_tritonbench_constrained_prompt(task_id: str, mode: str = "family") -> tuple[str, str]:
    """Build a TritonBench prompt and grammar contract for constrained Qwen generation."""
    spec = build_tritonbench_constrained_spec(task_id, mode=mode)
    return spec.prompt, spec.grammar_text


def normalize_translation_task(example: dict, index: int) -> dict:
    """Normaliza metadatos de ejemplos locales y TritonBench-T para el jsonl."""
    if "instruction" in example:
        return {
            "id": example.get("id", f"tritonbench_t_{index:03d}"),
            "benchmark": "tritonbench_t",
            "instruction": example["instruction"],
            "input": example.get("input", ""),
            "expected_output": example.get("output", ""),
        }

    return {
        "id": example["id"],
        "benchmark": "local",
        "pytorch_code": example["pytorch_code"],
        "operation_description": example["operation_description"],
        "expected_kernel_name": example.get("expected_kernel_name", ""),
        "expected_terms": example.get("expected_terms", []),
    }


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


def _is_qwen_constrained(model: dict, mode: str) -> bool:
    return mode == "constrained" and model.get("id") == "small_qwen25_coder_1_5b"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def collect_for_model_generation(
    model: dict,
    tasks: list,
    modes: list,
    samples: int,
    out_file,
    qwen_out_file=None,
) -> int:
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
                uses_real_xgrammar = _is_qwen_constrained(model, mode)
                print(
                    f"  [{model['id']}] task={task['id']} mode={mode} sample={sample_index} ... ",
                    end="",
                    flush=True,
                )

                if uses_real_xgrammar:
                    prompt, grammar_text = build_tritonbench_constrained_prompt(
                        task["id"], mode="family"
                    )
                    output, latency = call_hf_xgrammar(
                        prompt,
                        grammar_text=grammar_text,
                        max_new_tokens=768,
                    )
                else:
                    prompt = build_prompt(task, mode)
                    output, latency = call_fn(model_name, prompt)

                print(f"OK ({latency:.1f}s)")

                record = make_prediction_record(
                    model={**model, "provider": provider},
                    task=task,
                    mode=mode,
                    sample_index=sample_index,
                    prompt=prompt,
                    output=output,
                    latency_seconds=latency,
                    constrained_decoding_backend=(
                        "xgrammar_hf" if uses_real_xgrammar else "provider_prompt"
                    ),
                )

                target_file = qwen_out_file if uses_real_xgrammar else out_file
                target_file.write(json.dumps(record) + "\n")
                target_file.flush()
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

    for example_index, example in enumerate(examples, start=1):
        for sample_index in range(1, samples + 1):
            prompt = build_translation_prompt(example)
            task = normalize_translation_task(example, example_index)
            print(f"  [{model['id']}] example={task['id']} sample={sample_index} ... ", end="", flush=True)
            output, latency = call_fn(model_name, prompt)
            print(f"OK ({latency:.1f}s)")

            record = make_prediction_record(
                model={**model, "provider": provider},
                task=task,
                mode="translation",
                sample_index=sample_index,
                prompt=prompt,
                output=output,
                latency_seconds=latency,
                source_index=example_index,
            )
            out_file.write(json.dumps(record) + "\n")
            out_file.flush()
            count += 1

    return count


def run(
    providers_filter: list[str] | None = None,
    task_type: str = "generation",
    examples_file_override: str | None = None,
    samples_override: int | None = None,
) -> None:
    config = load_config()
    experiment = config.get("experiment", {})
    models = config.get("models", [])
    modes = config.get("modes", ["baseline", "constrained"])
    samples = int(samples_override or experiment.get("samples_per_model", 3))

    if providers_filter:
        models = [m for m in models if m.get("provider") in providers_filter]

    if not models:
        print("No hay modelos que coincidan con el filtro de provider.")
        sys.exit(1)

    if task_type == "translation":
        # Prefiere translation_examples_file del yaml si existe, si no usa la constante por defecto
        examples_file = examples_file_override or experiment.get("translation_examples_file")
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

        qwen_out_file = None
        if any(m.get("id") == "small_qwen25_coder_1_5b" for m in models) and "constrained" in modes:
            qwen_path = QWEN_CONSTRAINED_OUTPUT_PATH
            qwen_path.parent.mkdir(parents=True, exist_ok=True)
            qwen_mode_open = "a" if qwen_path.exists() else "w"
            qwen_out_file = qwen_path.open(qwen_mode_open, encoding="utf-8")

        print(f"\n=== Modo: GENERACIÓN desde cero ===")
        print(f"Tareas:  {[t['id'] for t in tasks]}")
        print(f"Modelos: {[m['id'] for m in models]}")
        print(f"Modos:   {modes}")
        print(f"Muestras por modelo: {samples}\n")

        with out_path.open(mode_open, encoding="utf-8") as out_file:
            for model in models:
                print(f"\n>> Modelo: {model.get('display_name', model['id'])} [{model['provider']}]")
                count = collect_for_model_generation(
                    model,
                    tasks,
                    modes,
                    samples,
                    out_file,
                    qwen_out_file=qwen_out_file,
                )
                total += count

        if qwen_out_file is not None:
            qwen_out_file.close()

        print(f"\nListo. {total} registros guardados en {out_path}")
        if qwen_out_file is not None:
            print(f"  Qwen constrained records también guardados en {QWEN_CONSTRAINED_OUTPUT_PATH}")
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
    parser.add_argument(
        "--examples-file",
        help="Archivo JSON de ejemplos para --task-type translation. Ejemplo: data/tritonbench_t_simp_subset5.json",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Sobrescribe samples_per_model del YAML para pruebas pequeñas.",
    )

    args = parser.parse_args()
    providers = None if args.all else [args.provider]
    run(
        providers_filter=providers,
        task_type=args.task_type,
        examples_file_override=args.examples_file,
        samples_override=args.samples,
    )


if __name__ == "__main__":
    main()
