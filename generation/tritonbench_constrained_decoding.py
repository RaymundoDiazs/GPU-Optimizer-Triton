"""TritonBench-T constrained-decoding helpers.

This module connects the 166 generated grammar contracts to the XGrammar/HF
decoder path. In environments without XGrammar/transformers installed, callers
can still use it to build the exact constrained prompt and selected grammar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from generation.xgrammar_llm_decoder import XGrammarLLMDecoder
from parsing.tritonbench_grammar_rules import load_tritonbench_contracts


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "tritonbench_t_simp_subset166.json"
FAMILY_GRAMMAR_PATH = ROOT / "grammars" / "tritonbench_t" / "general_kernel_family.ebnf"


@dataclass
class TritonBenchConstrainedSpec:
    task_id: str
    grammar_mode: str
    grammar_text: str
    prompt: str
    contract: dict


def _load_dataset() -> list[dict]:
    import json

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _task_index(task_id: str) -> int:
    prefix = "tritonbench_t_"
    if not task_id.startswith(prefix):
        raise ValueError(f"Expected task id like tritonbench_t_001, got {task_id!r}")
    return int(task_id.removeprefix(prefix)) - 1


def select_tritonbench_grammar(task_id: str, mode: str = "individual") -> str:
    """Return the grammar text selected for a TritonBench-T task.

    `individual` uses the task-specific EBNF contract. `family` uses the
    broader reusable family grammar.
    """
    contracts = load_tritonbench_contracts()
    if task_id not in contracts:
        raise KeyError(f"No TritonBench-T contract found for {task_id}")

    if mode == "individual":
        grammar_path = ROOT / contracts[task_id]["ebnf"]
    elif mode == "family":
        grammar_path = FAMILY_GRAMMAR_PATH
    else:
        raise ValueError("mode must be 'individual' or 'family'")

    return grammar_path.read_text(encoding="utf-8")


def build_tritonbench_constrained_spec(
    task_id: str,
    mode: str = "individual",
) -> TritonBenchConstrainedSpec:
    """Build the prompt + grammar pair used by constrained decoding."""
    contracts = load_tritonbench_contracts()
    if task_id not in contracts:
        raise KeyError(f"No TritonBench-T contract found for {task_id}")

    dataset = _load_dataset()
    index = _task_index(task_id)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"{task_id} is outside {DATASET_PATH.name}")

    contract = contracts[task_id]
    grammar_text = select_tritonbench_grammar(task_id, mode=mode)
    instruction = dataset[index]["instruction"]

    prompt = f"""{instruction}

Constrained decoding contract:
- Target task id: {task_id}
- Required wrapper: def {contract['wrapper']}
- Grammar mode: {mode}
- Family: {contract['family']}

Return only one Python module. Do not include explanations.
"""

    return TritonBenchConstrainedSpec(
        task_id=task_id,
        grammar_mode=mode,
        grammar_text=grammar_text,
        prompt=prompt.strip(),
        contract=contract,
    )


def generate_with_tritonbench_xgrammar(
    task_id: str,
    model,
    tokenizer,
    mode: str = "individual",
    max_new_tokens: int = 768,
):
    """Generate code for one TritonBench-T task using XGrammar constraints.

    This is the real token-level integration point. It requires a local
    HuggingFace-compatible model/tokenizer and `xgrammar` installed.
    """
    spec = build_tritonbench_constrained_spec(task_id, mode=mode)
    decoder = XGrammarLLMDecoder(
        model=model,
        tokenizer=tokenizer,
        grammar_text=spec.grammar_text,
    )
    return decoder.generate(
        prompt=spec.prompt,
        max_new_tokens=max_new_tokens,
    )
