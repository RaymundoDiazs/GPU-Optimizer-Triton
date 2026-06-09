"""Generate TritonBench-T grammar contracts from the upstream simple dataset.

The generated contracts are intentionally structural. They are not a full
semantic compiler for all 166 operators; they define the wrapper, required
module structure, operation family, and semantic evidence expected from a
generated Triton candidate. Existing hand-written EBNF files are preserved.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "external" / "TritonBench" / "data" / "TritonBench_T_simp_alpac_v1.json"
DEFAULT_SUBSET = ROOT / "data" / "tritonbench_t_simp_subset166.json"
DEFAULT_CONTRACTS = ROOT / "grammars" / "tritonbench_t_contracts.json"
GRAMMAR_DIR = ROOT / "grammars" / "tritonbench_t"


FAMILY_KEYWORDS = {
    "convolution_fusion": [
        "conv2d",
        "convolution",
        "batch norm",
        "batch_norm",
        "layer norm",
        "layer_norm",
    ],
    "fusion_matmul": [
        "matmul",
        "matrix multiplication",
        "bmm",
        "mm",
        "linear",
        "dot",
        "mv",
        "gemm",
    ],
    "reduction": [
        "sum",
        "mean",
        "max",
        "min",
        "argmax",
        "argmin",
        "norm",
        "softmax",
        "log_softmax",
        "logsumexp",
        "distance",
    ],
    "indexing_interpolation": [
        "index",
        "gather",
        "scatter",
        "grid",
        "sample",
        "fill",
        "tile",
        "repeat",
        "where",
        "select",
        "mask",
    ],
    "complex_fallback": [
        "svd",
        "qr",
        "lu",
        "solve",
        "determinant",
        "inverse",
        "cholesky",
        "eigen",
        "eig",
        "fft",
        "fourier",
        "least squares",
    ],
}


ELEMENTWISE_TERMS = [
    "add",
    "sub",
    "mul",
    "div",
    "sqrt",
    "rsqrt",
    "relu",
    "gelu",
    "silu",
    "sigmoid",
    "tanh",
    "exp",
    "log",
    "sin",
    "cos",
    "abs",
    "clamp",
    "pow",
]


def _extract_description(instruction: str) -> str:
    match = re.search(
        r"Functional Description:\s*(.*?)\s*Wrapper Entry Information:",
        instruction,
        flags=re.DOTALL,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return re.sub(r"\s+", " ", instruction).strip()


def _extract_wrapper(instruction: str) -> str:
    match = re.search(r"Wrapper Entry Information:\s*(.*)", instruction, flags=re.DOTALL)
    if not match:
        return "unknown(*args, **kwargs)"

    wrapper = re.sub(r"\s+", " ", match.group(1)).strip()
    wrapper = re.sub(r"^def\s+", "", wrapper)
    wrapper = wrapper.split(" Args:")[0]
    wrapper = wrapper.split(" Keyword args:")[0]
    wrapper = wrapper.split("; Args:")[0]
    wrapper = wrapper.split("; input")[0]
    wrapper = wrapper.split(" -> ")[0]
    wrapper = wrapper.rstrip(":; ")
    return wrapper


def _function_name(wrapper: str) -> str:
    match = re.match(r"([A-Za-z_][A-Za-z0-9_\.]*)\s*\(", wrapper)
    if not match:
        return "unknown"
    return match.group(1).split(".")[-1]


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip("_"))
    return safe.lower() or "unknown"


def _has_keyword(text: str, keyword: str) -> bool:
    if len(keyword) <= 3 and keyword.isalpha():
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(keyword)}(?![A-Za-z0-9_])", text) is not None
    return keyword in text


def _family(name: str, description: str) -> str:
    haystack = f"{name} {description}".lower()
    for family, keywords in FAMILY_KEYWORDS.items():
        if any(_has_keyword(haystack, keyword) for keyword in keywords):
            return family
    if any(_has_keyword(haystack, term) for term in ELEMENTWISE_TERMS):
        if sum(_has_keyword(haystack, term) for term in ELEMENTWISE_TERMS) >= 2:
            return "elementwise_fusion"
        return "elementwise"
    return "elementwise"


def _semantic_groups(name: str, description: str, family: str) -> list[list[str]]:
    tokens = []
    haystack = f"{name} {description}".lower()
    for token in ELEMENTWISE_TERMS:
        if _has_keyword(haystack, token):
            tokens.append(token)
    for token in [
        "softmax",
        "log_softmax",
        "conv2d",
        "dropout",
        "batch_norm",
        "layer_norm",
        "matmul",
        "bmm",
        "linear",
        "argmax",
        "argmin",
        "mean",
        "sum",
        "max",
        "min",
        "index",
        "grid",
        "fill",
        "solve",
        "svd",
        "qr",
        "lu",
        "fft",
        "eig",
    ]:
        if _has_keyword(haystack, token) and token not in tokens:
            tokens.append(token)

    if not tokens:
        tokens.append(name.lower())

    groups = [[name], tokens[:8]]
    if family == "elementwise":
        groups.append(["tl.load", "tl.store"])
    elif family == "elementwise_fusion":
        groups.append(["tl.load", "tl.store"])
    elif family == "reduction":
        groups.append(["tl.sum", "tl.max", "tl.min", "argmax", "argmin"])
    elif family == "fusion_matmul":
        groups.append(["tl.dot", "matmul", "bmm", "linear"])
    elif family == "convolution_fusion":
        groups.append(["conv2d", "weight", "tl.load", "tl.store"])
    elif family == "indexing_interpolation":
        groups.append(["index", "grid", "gather", "scatter", "tl.load", "tl.store"])
    elif family == "complex_fallback":
        groups.append(["tl.load", "tl.store"])
    return groups


def _required_terms(family: str) -> list[str]:
    return ["import triton", "import triton.language as tl", "@triton.jit"]


def _wrapper_regex(name: str) -> str:
    return rf"def\s+{re.escape(name)}\s*\("


def _existing_ebnf(index: int) -> Path | None:
    matches = sorted(GRAMMAR_DIR.glob(f"tritonbench_t_{index:03d}_*.ebnf"))
    return matches[0] if matches else None


def _ebnf_text(task_id: str, name: str, wrapper: str, family: str, semantic_groups: list[list[str]]) -> str:
    semantic = ", ".join("/".join(group) for group in semantic_groups)
    example_id = task_id.removeprefix("tritonbench_t_")
    return f"""/* Auto-generated TritonBench-T contract for {task_id}: {name}.
   TritonBench-T example {example_id}.
   This is a structural grammar contract for constrained decoding and
   post-generation validation. TritonBench remains the source of functional
   correctness and speedup evidence.
   Family: {family}. Semantic hints: {semantic}.
*/

root ::= module

module ::= imports implementation wrapper
imports ::= "import torch" newline "import triton" newline "import triton.language as tl" newline
implementation ::= triton_kernel
triton_kernel ::= "@triton.jit" newline kernel_def kernel_body
kernel_def ::= "def " identifier "(" parameters ")" ":" newline
kernel_body ::= program_id offsets mask load_ops computation store
program_id ::= indent identifier "=" "tl.program_id(0)" newline
offsets ::= indent identifier "=" identifier "*" "BLOCK_SIZE" "+" "tl.arange(0, BLOCK_SIZE)" newline
mask ::= indent identifier "=" identifier "<" identifier newline
load_ops ::= load load?
load ::= indent identifier "=" "tl.load(" pointer_expr "," "mask=" identifier ")" newline
computation ::= indent identifier "=" triton_expr newline
store ::= indent "tl.store(" pointer_expr "," identifier "," "mask=" identifier ")" newline
triton_expr ::= identifier | identifier binary_op identifier | identifier binary_op number | "tl." triton_func "(" triton_args ")"
pointer_expr ::= identifier "+" identifier
triton_args ::= /[^\\n)]*/
triton_func ::= "abs" | "cos" | "exp" | "log" | "maximum" | "minimum" | "sigmoid" | "sqrt" | "sum" | "tanh" | "where"
binary_op ::= "+" | "-" | "*" | "/" | "&" | "|" | "^"
number ::= /[0-9]+(\\.[0-9]+)?/
wrapper ::= "def {wrapper}" ":" newline wrapper_body
wrapper_body ::= statement+
indent ::= "    "
identifier ::= /[a-zA-Z_][a-zA-Z0-9_]*/
parameters ::= /[^)]*/
statement ::= /[^\\n]+/
"""


def generate(source: Path, subset: Path, contracts_path: Path, preserve_existing: bool = True) -> dict:
    items = json.loads(source.read_text(encoding="utf-8"))
    if len(items) < 166:
        raise ValueError(f"Expected 166 TritonBench-T simple entries, found {len(items)}")

    GRAMMAR_DIR.mkdir(parents=True, exist_ok=True)
    subset.parent.mkdir(parents=True, exist_ok=True)
    contracts_path.parent.mkdir(parents=True, exist_ok=True)

    subset.write_text(json.dumps(items, indent=2), encoding="utf-8")

    contracts = {}
    for index, item in enumerate(items, start=1):
        task_id = f"tritonbench_t_{index:03d}"
        instruction = item["instruction"]
        description = _extract_description(instruction)
        wrapper = _extract_wrapper(instruction)
        name = _function_name(wrapper)
        family = _family(name, description)
        semantic_groups = _semantic_groups(name, description, family)

        existing = _existing_ebnf(index) if preserve_existing else None
        ebnf_path = existing or GRAMMAR_DIR / f"{task_id}_{_safe_name(name)}.ebnf"
        if not existing:
            ebnf_path.write_text(
                _ebnf_text(task_id, name, wrapper, family, semantic_groups),
                encoding="utf-8",
            )

        contracts[task_id] = {
            "name": name,
            "family": family,
            "ebnf": str(ebnf_path.relative_to(ROOT)),
            "wrapper": wrapper,
            "description": description,
            "wrapper_regex": _wrapper_regex(name),
            "required_terms": _required_terms(family),
            "semantic_terms_any": semantic_groups,
            "source_index": index,
        }

    contracts_path.write_text(json.dumps(contracts, indent=2), encoding="utf-8")
    return contracts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    contracts = generate(
        source=args.source,
        subset=args.subset,
        contracts_path=args.contracts,
        preserve_existing=not args.overwrite_existing,
    )
    print(f"Generated {len(contracts)} TritonBench-T contracts")
    print(f"Subset: {args.subset}")
    print(f"Contracts: {args.contracts}")


if __name__ == "__main__":
    main()
