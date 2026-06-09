"""TritonBench-T constrained-decoding helpers.

Conecta la gramática general con el decoder XGrammar/HF.
Usa únicamente la gramática general (general_kernel_family.ebnf) para los
166 operadores. El prompt es idéntico al del baseline (instruction + código
PyTorch de referencia) — la única diferencia es la restricción gramatical.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from generation.xgrammar_llm_decoder import XGrammarLLMDecoder
from grammars.few_shot_examples import FEW_SHOT_EXAMPLES
from parsing.tritonbench_grammar_rules import load_tritonbench_contracts

ROOT             = Path(__file__).resolve().parents[1]
SIMP_PATH        = ROOT / "extras/TritonBench-main/data/TritonBench_T_simp_alpac_v1.json"
T_PATH           = ROOT / "extras/TritonBench-main/data/TritonBench_T_v1.jsonl"
PY_DIR           = ROOT / "extras/TritonBench-main/data/TritonBench_T_v1"
FAMILY_GRAMMAR_PATH = ROOT / "grammars" / "tritonbench_t" / "general_kernel_family.ebnf"

if not FAMILY_GRAMMAR_PATH.exists():
    raise FileNotFoundError(
        f"Gramática general no encontrada en {FAMILY_GRAMMAR_PATH}"
    )


@dataclass
class TritonBenchConstrainedSpec:
    task_id: str
    grammar_text: str
    prompt: str
    contract: dict


def _load_datasets() -> tuple[list[dict], list[dict]]:
    with SIMP_PATH.open("r", encoding="utf-8") as fh:
        simp_data = json.load(fh)
    with T_PATH.open("r", encoding="utf-8") as fh:
        t_data = json.load(fh)
    return simp_data, t_data


def _load_pytorch_func(file: str) -> str:
    """Lee el archivo .py del operador y devuelve solo la función (sin test driver)."""
    path = PY_DIR / file
    with open(path, encoding="utf-8") as fh:
        full = fh.read()
    return re.split(r"#{10,}", full)[0].strip()


def _task_index(task_id: str) -> int:
    prefix = "tritonbench_t_"
    if not task_id.startswith(prefix):
        raise ValueError(f"task_id esperado como tritonbench_t_001, recibido {task_id!r}")
    return int(task_id.removeprefix(prefix)) - 1


def build_tritonbench_constrained_spec(
    task_id: str,
    mode: str = "family",
) -> TritonBenchConstrainedSpec:
    """Construye el prompt + gramática para un operador de TritonBench-T.

    El prompt es idéntico al del baseline: instruction + código PyTorch de
    referencia. La única diferencia con el baseline es que la generación
    estará restringida por la gramática GBNF en tiempo de decodificación.
    """
    contracts = load_tritonbench_contracts()
    if task_id not in contracts:
        raise KeyError(f"Contrato no encontrado para {task_id}")

    simp_data, t_data = _load_datasets()
    index = _task_index(task_id)

    if index < 0 or index >= len(simp_data):
        raise IndexError(f"{task_id} fuera de rango")

    contract = contracts[task_id]
    instruction = simp_data[index]["instruction"]
    pytorch_code = _load_pytorch_func(t_data[index]["file"])
    family = contract.get("family", "complex_fallback")
    example = FEW_SHOT_EXAMPLES.get(family, FEW_SHOT_EXAMPLES["complex_fallback"])

    prompt = (
        instruction
        + "\n\nPyTorch reference implementation:\n```python\n"
        + pytorch_code
        + "\n```"
        + "\n\nHere is a correct Triton kernel example for a similar operator"
        + f" ({family}):\n```python\n"
        + example
        + "\n```"
        + "\n\nCRITICAL RULES:"
        + "\n1. Kernel name MUST end with '_kernel'. Wrapper name MUST NOT contain '_kernel'."
        + "\n2. Wrapper MUST call <name>_kernel[grid](...) — the name must match."
        + "\n3. NEVER use .data_ptr() — pass tensors directly to the kernel launch."
        + "\n4. Use input.numel() for 1D element count. Do NOT assume 2D shapes (M, N = x.shape) unless the input is guaranteed 2D."
        + "\n5. Use tl.math.* for transcendental functions (tl.math.exp, tl.math.log, tl.math.sqrt). Do NOT use tl.tanh, tl.sigmoid — they do not exist."
        + "\n6. tl.arange arguments must be tl.constexpr (use BLOCK_SIZE, not runtime variables)."
        + "\n\nNow generate the Triton kernel:"
    )

    grammar_text = FAMILY_GRAMMAR_PATH.read_text(encoding="utf-8")

    return TritonBenchConstrainedSpec(
        task_id=task_id,
        grammar_text=grammar_text,
        prompt=prompt,
        contract=contract,
    )


def generate_with_tritonbench_xgrammar(
    task_id: str,
    model,
    tokenizer,
    mode: str = "family",
    max_new_tokens: int = 768,
):
    """Genera código para un operador de TritonBench-T usando XGrammar.

    Requiere modelo HuggingFace local (Qwen) y xgrammar instalado.
    """
    spec = build_tritonbench_constrained_spec(task_id, mode=mode)

    decoder = XGrammarLLMDecoder(
        model=model,
        tokenizer=tokenizer,
        grammar_text=spec.grammar_text,
    )

    return decoder.generate(prompt=spec.prompt, max_new_tokens=max_new_tokens)
