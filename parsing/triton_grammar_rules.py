import re
from dataclasses import dataclass


@dataclass
class GrammarCheckResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


REQUIRED_PATTERNS = {
    "triton_jit": r"@triton\.jit",
    "kernel_def": r"def\s+\w+\s*\(",
    "program_id": r"tl\.program_id\s*\(",
    "load": r"tl\.load\s*\(",
    "store": r"tl\.store\s*\(",
}


def check_balanced_symbols(code: str) -> list[str]:
    errors = []

    if code.count("(") != code.count(")"):
        errors.append("Unbalanced parentheses")

    if code.count("[") != code.count("]"):
        errors.append("Unbalanced brackets")

    return errors


def validate_triton_kernel(code: str) -> GrammarCheckResult:
    errors = []
    warnings = []

    clean_code = code.strip()

    if not clean_code:
        return GrammarCheckResult(
            valid=False,
            errors=["Kernel code is empty"],
            warnings=[],
        )

    for rule_name, pattern in REQUIRED_PATTERNS.items():
        if not re.search(pattern, clean_code):
            errors.append(f"Missing required Triton structure: {rule_name}")

    errors.extend(check_balanced_symbols(clean_code))

    if "tl.arange" not in clean_code:
        warnings.append("Kernel does not use tl.arange; check if block offsets are needed")

    return GrammarCheckResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def get_valid_next_tokens(current_state: str) -> list[str]:
    """
    Small rule-based simulation of constrained decoding.
    Later this can be replaced by real XGrammar token restrictions.
    """

    token_rules = {
        "START": ["@triton.jit"],
        "@triton.jit": ["def"],
        "def": ["kernel_name"],
        "tl.load(": ["x_ptr", "y_ptr", "input_ptr", "offsets"],
        "tl.store(": ["output_ptr", "offsets"],
        "tl.program_id(": ["0"],
    }

    return token_rules.get(current_state, [])