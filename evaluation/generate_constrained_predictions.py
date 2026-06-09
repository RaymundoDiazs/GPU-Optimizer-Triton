"""Paso 2 (constrained) — Runner de predicciones con XGrammar.


IMPORTANTE:
- El constrained decoding via logits processor solo funciona con un modelo
  HF local (Qwen). GPT-4o / Claude por API NO exponen acceso a logits, asi
  que NO se pueden correr en modo constrained real.
- Requiere `xgrammar`, `transformers` y `torch` instalados y una GPU para que
  sea practico. Sin esas dependencias, usar --dry-run para validar el cableado
  (prompts + gramaticas + contratos de los 166 tasks) sin cargar el modelo.

Uso:
    # Validar el cableado de los 166 sin modelo (rapido, sin GPU):
    python evaluation/generate_constrained_predictions.py --dry-run

    # Smoke test real sobre 3 tasks (requiere GPU + deps):
    python evaluation/generate_constrained_predictions.py --limit 3

    # Corrida completa (166 tasks, Qwen local + XGrammar):
    python evaluation/generate_constrained_predictions.py

    # Reanudar una corrida interrumpida:
    python evaluation/generate_constrained_predictions.py --resume

Genera: evaluation/predictions_qwen_constrained.jsonl  (166 lineas)
Despues, pasar ese jsonl por el MISMO evaluador que produjo
results/tritonbench_qwen_baseline.json -> tritonbench_qwen_constrained.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Asegura que `generation/` y `parsing/` sean importables sin importar el CWD.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.tritonbench_constrained_decoding import (  # noqa: E402
    build_tritonbench_constrained_spec,
    generate_with_tritonbench_xgrammar,
)
from parsing.tritonbench_grammar_rules import load_tritonbench_contracts  # noqa: E402

DATASET_PATH = ROOT / "data" / "tritonbench_t_simp_subset166.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "predictions_qwen_constrained.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def _task_id_to_index(task_id: str) -> int:
    """tritonbench_t_001 -> 0 (alineado con el dataset por posicion)."""
    return int(task_id.removeprefix("tritonbench_t_")) - 1


def _load_dataset() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _ordered_task_ids() -> list[str]:
    """Lista ordenada tritonbench_t_001 .. _166 a partir de los contratos."""
    return sorted(load_tritonbench_contracts().keys())


def _count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def dry_run(mode: str, limit: int) -> int:
    """Construye el spec (prompt + gramatica + contrato) de cada task SIN modelo.

    Valida que las 166 gramaticas EBNF y contratos existan y carguen, y que el
    prompt constrained se arme. No genera codigo. Devuelve el numero de fallos.
    """
    task_ids = _ordered_task_ids()
    if limit > 0:
        task_ids = task_ids[:limit]

    failures: list[tuple[str, str]] = []
    empty_grammar: list[str] = []

    for task_id in task_ids:
        try:
            spec = build_tritonbench_constrained_spec(task_id, mode=mode)
            if not spec.grammar_text.strip():
                empty_grammar.append(task_id)
            if not spec.prompt.strip():
                failures.append((task_id, "prompt vacio"))
        except Exception as exc:  # noqa: BLE001
            failures.append((task_id, f"{type(exc).__name__}: {exc}"))

    ok = len(task_ids) - len(failures)
    print(f"[dry-run] mode={mode}  tasks={len(task_ids)}  ok={ok}  fallos={len(failures)}")
    if empty_grammar:
        print(f"[dry-run] AVISO gramatica vacia en {len(empty_grammar)}: {empty_grammar[:5]}...")
    for task_id, why in failures[:20]:
        print(f"  FALLO {task_id}: {why}")
    if not failures and not empty_grammar:
        print("[dry-run] OK: las 166 specs (prompt+gramatica+contrato) se arman sin errores.")
        print("[dry-run] El cableado esta correcto; solo falta correr con GPU + deps.")
    return len(failures)


def run(
    model_name: str,
    mode: str,
    max_new_tokens: int,
    limit: int,
    resume: bool,
    seed: int,
    output: Path,
    grammar_file: Path | None = None,
    no_grammar: bool = False,
) -> int:
    """Corrida real: carga Qwen local una vez y genera con XGrammar por task.

    Si se pasa grammar_file, se usa ESA gramatica GBNF (valida para XGrammar)
    para los 166 tasks, en vez de las EBNF individuales del repo (que estan en
    formato ISO y XGrammar no puede compilar). El prompt sigue siendo el de cada
    task; solo cambia la gramatica que restringe el decoding.
    """
    # Carga e import perezoso para que --dry-run no exija las dependencias.
    from generation.hf_xgrammar_provider import load_hf_model

    custom_grammar = None
    if grammar_file is not None:
        custom_grammar = Path(grammar_file).read_text(encoding="utf-8")
        print(f"Usando gramatica unica: {grammar_file}")
        from generation.xgrammar_llm_decoder import XGrammarLLMDecoder
        from generation.tritonbench_constrained_decoding import (
            build_tritonbench_constrained_spec,
        )

    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001
        pass

    print(f"Cargando modelo local {model_name} (esto puede tardar)...")
    model, tokenizer = load_hf_model(model_name)
    print("Modelo cargado.")

    dataset = _load_dataset()
    task_ids = _ordered_task_ids()
    if limit > 0:
        task_ids = task_ids[:limit]

    start = _count_existing(output) if resume else 0
    if resume and start:
        print(f"Reanudando: ya hay {start} predicciones en {output.name}, se omiten.")
    pending = task_ids[start:]

    file_mode = "a" if (resume and start) else "w"
    written = 0
    errors = 0
    t0 = time.perf_counter()

    with output.open(file_mode, encoding="utf-8") as out:
        for i, task_id in enumerate(pending, start=start + 1):
            index = _task_id_to_index(task_id)
            instruction = dataset[index]["instruction"]
            try:
                if no_grammar:
                    # Solo diferenciador: prompt con few-shot, sin XGrammar.
                    from generation.tritonbench_constrained_decoding import build_tritonbench_constrained_spec
                    from transformers import AutoModelForCausalLM, AutoTokenizer as HFTokenizer
                    import torch
                    spec = build_tritonbench_constrained_spec(task_id, mode=mode)
                    text = spec.prompt
                    try:
                        if getattr(tokenizer, "chat_template", None):
                            text = tokenizer.apply_chat_template(
                                [{"role": "user", "content": spec.prompt}],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                    except Exception:
                        pass
                    inputs = tokenizer(text, return_tensors="pt")
                    try:
                        device = next(model.parameters()).device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    except Exception:
                        pass
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
                    generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    from evaluation.code_safety import validate_generated_code_safety
                    safety = validate_generated_code_safety(generated_code)
                    from generation.xgrammar_llm_decoder import XGrammarGenerationResult
                    result = XGrammarGenerationResult(
                        prompt=spec.prompt,
                        generated_code=generated_code,
                        accepted=safety.safe,
                        errors=safety.errors,
                        warnings=[],
                        used_xgrammar=False,
                    )
                elif custom_grammar is not None:
                    spec = build_tritonbench_constrained_spec(task_id, mode=mode)
                    decoder = XGrammarLLMDecoder(
                        model=model, tokenizer=tokenizer, grammar_text=custom_grammar
                    )
                    result = decoder.generate(
                        prompt=spec.prompt, max_new_tokens=max_new_tokens
                    )
                else:
                    result = generate_with_tritonbench_xgrammar(
                        task_id=task_id,
                        model=model,
                        tokenizer=tokenizer,
                        mode=mode,
                        max_new_tokens=max_new_tokens,
                    )
                predict = result.generated_code
                tag = "ok" if result.accepted else "gen(no-valida)"
            except Exception as exc:  # noqa: BLE001
                predict = ""  # se mantiene la linea para no romper el alineado
                errors += 1
                tag = f"ERROR {type(exc).__name__}"

            out.write(json.dumps({"instruction": instruction, "predict": predict}) + "\n")
            out.flush()
            written += 1
            print(f"[{i}/{len(task_ids)}] {task_id} -> {tag}")

    dt = time.perf_counter() - t0
    print(f"\nFinalizado — escritas {written} (errores={errors}) en {dt:.1f}s")
    print(f"Salida: {output}")
    print(
        "Siguiente paso: pasar este jsonl por el mismo evaluador que genero "
        "results/tritonbench_qwen_baseline.json para obtener "
        "tritonbench_qwen_constrained.json."
    )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generar predicciones constrained (XGrammar) para TritonBench-T."
    )
    parser.add_argument("--provider", default="qwen", choices=["qwen"],
                        help="Solo qwen: el constrained real requiere modelo HF local.")
    parser.add_argument("--mode", default="individual", choices=["individual", "family"],
                        help="individual = EBNF por operador; family = gramatica de familia.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = los 166; >0 = primeros N (smoke test).")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Valida prompts+gramaticas de los 166 sin cargar el modelo.")
    parser.add_argument("--grammar-file", type=Path, default=None,
                        help="Ruta a UNA gramatica GBNF valida para usar en los 166.")
    parser.add_argument("--no-grammar", action="store_true",
                        help="Genera sin XGrammar: solo prompt con diferenciador few-shot. "
                             "Mas rapido, sin riesgo de bucles de gramatica.")
    args = parser.parse_args()

    if args.dry_run:
        failures = dry_run(mode=args.mode, limit=args.limit)
        sys.exit(1 if failures else 0)

    errors = run(
        model_name=args.model_name,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        resume=args.resume,
        seed=args.seed,
        output=args.output,
        grammar_file=args.grammar_file,
        no_grammar=args.no_grammar,
    )
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
