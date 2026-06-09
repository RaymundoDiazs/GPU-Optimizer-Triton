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

# Few-shot header: ensena al modelo la API REAL de Triton con ejemplos
# resueltos, para que no alucine funciones inexistentes. Es el diferenciador.
TRITON_FEWSHOT_HEADER = """You are an expert in Triton GPU programming. Translate a PyTorch operation into a high-performance Triton kernel.

Output ONE self-contained Python module with: (a) imports (torch, triton, triton.language as tl), (b) the @triton.jit kernel(s), (c) the wrapper function matching the given signature. No test code, no explanations.

MANDATORY RULES:
1. PARALLELISM: use tl.program_id, never Python for-loops.
2. BLOCK SIZE: must be a tl.constexpr parameter.
3. MEMORY: always tl.load / tl.store, never direct indexing.
4. MASK: guard every load/store with mask = offsets < n_elements.
5. OFFSETS: offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).
6. GRID: wrapper computes grid and launches: kernel[grid](...).

FORBIDDEN INSIDE @triton.jit (these cause failures — NEVER do them):
- NO Python for/while loops over elements.
- NO if/raise/ValueError/assert or shape checks.
- NO len(), .shape, .dtype, range(), print() inside the kernel.
- NO undefined variables. Every variable must be a kernel argument or computed with tl.* .
- The kernel body must ONLY: compute offsets, mask, tl.load, do tl.* math, tl.store.
- Put ALL Python logic (shape checks, dtype, grid) in the WRAPPER, never in the kernel.

API REFERENCE (use ONLY these, do not invent):
tl.program_id(axis), tl.arange(start,end), tl.constexpr, triton.cdiv(a,b),
tl.load(ptr,mask,other), tl.store(ptr,value,mask), tl.sum/max/min(x,axis),
tl.exp, tl.log, tl.sqrt, tl.where(cond,x,y), tl.dot(a,b).

COMPLETE VALID EXAMPLE — vector addition:
```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

COMPLETE VALID EXAMPLE — row softmax:
```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    output = numerator / tl.sum(numerator, axis=0)
    tl.store(output_ptr + row_idx * n_cols + col_offsets, output, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    softmax_kernel[(n_rows,)](x, output, n_cols, BLOCK_SIZE=triton.next_power_of_2(n_cols))
    return output
```"""
FAMILY_GRAMMAR_PATH = ROOT / "grammars" / "tritonbench_t" / "general_kernel_family.ebnf"

if not FAMILY_GRAMMAR_PATH.exists():
    raise FileNotFoundError(
        f"Expected the general Triton grammar at {FAMILY_GRAMMAR_PATH}, but it was not found."
    )


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

    prompt = f"""{TRITON_FEWSHOT_HEADER}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{instruction}

Required wrapper signature: def {contract['wrapper']}

Now write the complete Triton module for THIS task, following the rules and
examples above. Use tl.program_id, tl.arange, tl.load, tl.store with masks.
Do NOT use Python for-loops. Do NOT invent functions. Return only the module.
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