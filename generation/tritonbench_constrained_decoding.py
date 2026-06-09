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
    # Override: operators classified as fusion_matmul but whose description says
    # "element-wise" (without mentioning matrix/matmul/linear/dot) are really
    # elementwise ops. Giving them the matmul few-shot causes M,N=x.shape on 1D
    # tensors. The elementwise few-shot uses numel() which is shape-agnostic.
    # "linear unit(s)" is stripped because ReLU/SELU/GELU contain "linear" in
    # their activation name, not because they do a linear transformation.
    # Ops with "dim" in wrapper signature are reductions, not pure elementwise.
    if family == "fusion_matmul":
        desc = contract.get("description", "").lower()
        wrapper_sig = contract.get("wrapper", "").lower()
        desc_cleaned = re.sub(r"linear unit[s]?", "", desc)
        is_elementwise = "element" in desc
        is_matmul = any(kw in desc_cleaned for kw in ["matrix", "matmul", "linear", "dot product", "gemm"])
        has_dim = "dim" in wrapper_sig
        if is_elementwise and not is_matmul and not has_dim:
            family = "elementwise"
    example = FEW_SHOT_EXAMPLES.get(family, FEW_SHOT_EXAMPLES["complex_fallback"])

    # Extract the expected wrapper function name from the instruction.
    # It's always present as "Wrapper Entry Information: [def] name("
    import re as _re
    _wrapper_match = _re.search(
        r'Wrapper Entry Information:.*?(?:def\s+)?(\w+)\s*\(', instruction
    )
    wrapper_name = _wrapper_match.group(1) if _wrapper_match else None

    prompt = (
        instruction
        + "\n\nPyTorch reference implementation:\n```python\n"
        + pytorch_code
        + "\n```"
        + "\n\nHere is a correct Triton kernel example for a similar operator"
        + f" ({family}):\n```python\n"
        + example
        + "\n```"
        + "\n\nRules: kernel name ends with _kernel, wrapper calls it by name."
        + " Pass tensors directly to kernel (not .data_ptr())."
        + " Use input.numel() for element count."
        + " Use tl.math.exp, tl.math.log for math ops (not tl.tanh, tl.sigmoid, tl.square)."
        + " Use tl.math.log(1+x) instead of tl.math.log1p (not available)."
        + " tl.arange needs tl.constexpr args (BLOCK_SIZE)."
    )
    if wrapper_name:
        prompt += (
            f"\nWrapper function MUST be named: {wrapper_name}"
            f"\nKernel function MUST be named: {wrapper_name}_kernel"
        )
    prompt += "\n\nNow generate the Triton kernel:"

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
