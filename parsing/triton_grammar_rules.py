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
    "arange": r"tl\.arange\s*\(",
    "load": r"tl\.load\s*\(",
    "store": r"tl\.store\s*\(",
    "mask": r"mask\s*=",
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


def validate_vector_add_kernel(code: str) -> GrammarCheckResult:
    """
    Validate the current formal grammar target: a Triton vector-add kernel.

    This is a practical validator for the grammar contract in
    grammars/vector_add.ebnf. It checks the required structure after
    generation; token-level constrained decoding is a separate integration.
    """
    base_result = validate_triton_kernel(code)
    errors = list(base_result.errors)
    warnings = list(base_result.warnings)
    clean_code = re.sub(r"\s+", " ", code.strip())

    vector_add_patterns = {
        "vector_add_kernel_name": r"def\s+vector_add_kernel\s*\(",
        "block_size_constexpr": r"BLOCK_SIZE\s*:\s*tl\.constexpr",
        "n_constexpr": r"N\s*:\s*tl\.constexpr",
        "offsets": r"offsets\s*=\s*pid\s*\*\s*BLOCK_SIZE\s*\+\s*tl\.arange\s*\(\s*0\s*,\s*BLOCK_SIZE\s*\)",
        "bounds_mask": r"mask\s*=\s*offsets\s*<\s*N",
        "load_x": r"x\s*=\s*tl\.load\s*\(\s*X\s*\+\s*offsets\s*,\s*mask\s*=\s*mask\s*\)",
        "load_y": r"y\s*=\s*tl\.load\s*\(\s*Y\s*\+\s*offsets\s*,\s*mask\s*=\s*mask\s*\)",
        "store_add": r"tl\.store\s*\(\s*Z\s*\+\s*offsets\s*,\s*x\s*\+\s*y\s*,\s*mask\s*=\s*mask\s*\)",
    }

    for rule_name, pattern in vector_add_patterns.items():
        if not re.search(pattern, clean_code):
            errors.append(f"Missing vector-add grammar rule: {rule_name}")

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
