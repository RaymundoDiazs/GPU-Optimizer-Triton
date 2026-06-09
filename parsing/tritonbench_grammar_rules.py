import json
import re
from dataclasses import dataclass
from pathlib import Path

from parsing.triton_grammar_rules import validate_triton_kernel


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_PATH = ROOT / "grammars" / "tritonbench_t_contracts.json"


@dataclass
class TritonBenchGrammarResult:
    valid: bool
    task_id: str
    errors: list[str]
    warnings: list[str]
    passed_rules: list[str]


def load_tritonbench_contracts() -> dict:
    """Load grammar contracts for the current TritonBench-T subset."""
    with CONTRACTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_python_code(text: str) -> str:
    """Extract Python source from a fenced LLM response when possible."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"^```(?:python|py)?\s*\n?", "", text.strip()).removesuffix("```").strip()


def _contains_any(clean_code: str, options: list[str]) -> bool:
    lowered = clean_code.lower()
    return any(option.lower() in lowered for option in options)


def _check_balanced_symbols(code: str) -> list[str]:
    errors = []
    if code.count("(") != code.count(")"):
        errors.append("Unbalanced parentheses")
    if code.count("[") != code.count("]"):
        errors.append("Unbalanced brackets")
    if code.count("{") != code.count("}"):
        errors.append("Unbalanced braces")
    return errors


FAMILY_REQUIRED_TERMS = {
    "elementwise": [
        ["@triton.jit"],
        ["tl.program_id"],
        ["tl.arange"],
        ["mask"],
        ["tl.load"],
        ["tl.store"],
    ],
    "elementwise_fusion": [
        ["@triton.jit"],
        ["tl.program_id"],
        ["tl.arange"],
        ["mask"],
        ["tl.load"],
        ["tl.store"],
    ],
    "reduction": [
        ["@triton.jit"],
        ["tl.load"],
        ["tl.store"],
        ["argmax", "max", "tl.sum", "tl.max", "tl.min"],
    ],
    "convolution_fusion": [
        ["@triton.jit"],
        ["tl.load"],
        ["tl.store"],
        ["weight", "WEIGHT"],
        ["sigmoid", "tl.exp", "exp"],
    ],
    "fusion_matmul": [
        ["@triton.jit"],
        ["tl.load"],
        ["tl.store"],
        ["tl.dot", "bmm", "matmul", "dot", "mv", "linear"],
        ["gelu", "relu", "tanh", "softmax", "log_softmax", "logsumexp", "dropout"],
    ],
    "complex_fallback": [
        ["@triton.jit"],
        ["tl.program_id"],
        ["tl.arange"],
        ["mask"],
        ["tl.load"],
        ["tl.store"],
    ],
    "indexing_interpolation": [
        ["@triton.jit"],
        ["tl.program_id"],
        ["tl.arange"],
        ["mask"],
        ["tl.load"],
        ["tl.store"],
        ["index", "gather", "scatter", "where", "tl.load", "tl.store"],
    ],
}


def validate_tritonbench_candidate(task_id: str, code: str) -> TritonBenchGrammarResult:
    """Validate one generated module against a TritonBench-T subset contract.

    This is post-generation validation. The EBNF files define the formal intent;
    this function implements practical checks that are easy to benchmark.
    """
    contracts = load_tritonbench_contracts()
    if task_id not in contracts:
        return TritonBenchGrammarResult(
            valid=False,
            task_id=task_id,
            errors=[f"No TritonBench-T grammar contract found for task_id={task_id}"],
            warnings=[],
            passed_rules=[],
        )

    contract = contracts[task_id]
    raw_code = code.strip()
    clean_code = extract_python_code(code)
    compact = re.sub(r"\s+", " ", clean_code)
    errors = []
    warnings = []
    passed_rules = []

    if "```" in raw_code:
        errors.append("Candidate contains markdown fences")

    if not clean_code:
        errors.append("Candidate code is empty")
    else:
        passed_rules.append("non_empty_code")

    for error in _check_balanced_symbols(clean_code):
        errors.append(error)
    if not any(error.startswith("Unbalanced") for error in errors):
        passed_rules.append("balanced_symbols")

    if re.search(contract["wrapper_regex"], compact):
        passed_rules.append("wrapper_signature")
    else:
        errors.append(f"Missing expected wrapper: {contract['name']}")

    kernel_result = validate_triton_kernel(clean_code)
    if kernel_result.valid:
        passed_rules.append("strict_triton_kernel")
    else:
        errors.extend(kernel_result.errors)
    warnings.extend(kernel_result.warnings)

    for term in contract.get("required_terms", []):
        if term in clean_code:
            passed_rules.append(f"required:{term}")
        else:
            errors.append(f"Missing required term: {term}")

    for group_index, options in enumerate(contract.get("semantic_terms_any", []), start=1):
        if _contains_any(clean_code, options):
            passed_rules.append(f"semantic_group_{group_index}")
        else:
            errors.append(f"Missing semantic evidence; expected one of: {', '.join(options)}")

    if "TODO" in clean_code or re.search(r"\bpass\b", clean_code):
        warnings.append("Candidate contains TODO/pass marker")

    return TritonBenchGrammarResult(
        valid=len(errors) == 0,
        task_id=task_id,
        errors=errors,
        warnings=warnings,
        passed_rules=passed_rules,
    )


def validate_tritonbench_family_candidate(task_id: str, code: str) -> TritonBenchGrammarResult:
    """Validate a candidate with the reusable family grammar rules."""
    contracts = load_tritonbench_contracts()
    if task_id not in contracts:
        return TritonBenchGrammarResult(
            valid=False,
            task_id=task_id,
            errors=[f"No TritonBench-T grammar contract found for task_id={task_id}"],
            warnings=[],
            passed_rules=[],
        )

    contract = contracts[task_id]
    family = contract.get("family", "")
    raw_code = code.strip()
    clean_code = extract_python_code(code)
    compact = re.sub(r"\s+", " ", clean_code)
    errors = []
    warnings = []
    passed_rules = []

    if "```" in raw_code:
        errors.append("Candidate contains markdown fences")

    if not clean_code:
        errors.append("Candidate code is empty")
    else:
        passed_rules.append("non_empty_code")

    for error in _check_balanced_symbols(clean_code):
        errors.append(error)
    if not any(error.startswith("Unbalanced") for error in errors):
        passed_rules.append("balanced_symbols")

    if re.search(contract["wrapper_regex"], compact):
        passed_rules.append("wrapper_signature")
    else:
        errors.append(f"Missing expected wrapper: {contract['name']}")

    kernel_result = validate_triton_kernel(clean_code)
    if kernel_result.valid:
        passed_rules.append("strict_triton_kernel")
    else:
        errors.extend(kernel_result.errors)
    warnings.extend(kernel_result.warnings)

    required_groups = FAMILY_REQUIRED_TERMS.get(family)
    if required_groups is None:
        errors.append(f"No family grammar rules found for family={family}")
    else:
        for group_index, options in enumerate(required_groups, start=1):
            if _contains_any(clean_code, options):
                passed_rules.append(f"family_group_{group_index}")
            else:
                errors.append(f"Missing family evidence; expected one of: {', '.join(options)}")

    if "TODO" in clean_code or re.search(r"\bpass\b", clean_code):
        warnings.append("Candidate contains TODO/pass marker")

    return TritonBenchGrammarResult(
        valid=len(errors) == 0,
        task_id=task_id,
        errors=errors,
        warnings=warnings,
        passed_rules=passed_rules,
    )
