import ast
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

FORBIDDEN_KERNEL_NAME_ROOTS = {"torch", "np", "numpy", "math"}
FORBIDDEN_KERNEL_CALLS = {"isinstance", "range"}
FORBIDDEN_TRITON_JIT_CALLS = {
    "triton.jit.block_index",
    "triton.jit.get_block_index",
    "triton.jit.get_global_id",
}


def check_balanced_symbols(code: str) -> list[str]:
    errors = []

    if code.count("(") != code.count(")"):
        errors.append("Unbalanced parentheses")

    if code.count("[") != code.count("]"):
        errors.append("Unbalanced brackets")

    return errors


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _attribute_root(node: ast.AST) -> str:
    name = _attribute_name(node)
    return name.split(".", 1)[0] if name else ""


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _attribute_name(node.func)
    return _attribute_name(node)


def _is_triton_jit_kernel(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef) and any(
        _decorator_name(decorator) == "triton.jit"
        for decorator in node.decorator_list
    )


def _validate_kernel_ast(kernel: ast.FunctionDef) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(kernel):
        if node is kernel:
            continue
        if isinstance(node, (ast.For, ast.AsyncFor)):
            errors.append(
                f"Forbidden Python loop inside @triton.jit kernel {kernel.name}: for"
            )
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_KERNEL_NAME_ROOTS:
                errors.append(
                    f"Forbidden {node.id} use inside @triton.jit kernel {kernel.name}"
                )
        elif isinstance(node, ast.Attribute):
            root = _attribute_root(node)
            full_name = _attribute_name(node)
            if root in FORBIDDEN_KERNEL_NAME_ROOTS:
                errors.append(
                    f"Forbidden {root} use inside @triton.jit kernel {kernel.name}"
                )
            if full_name in FORBIDDEN_TRITON_JIT_CALLS:
                errors.append(
                    f"Invalid Triton API inside @triton.jit kernel {kernel.name}: {full_name}"
                )
        elif isinstance(node, ast.Call):
            call_name = _attribute_name(node.func)
            if call_name in FORBIDDEN_KERNEL_CALLS:
                errors.append(
                    f"Forbidden call inside @triton.jit kernel {kernel.name}: {call_name}"
                )
            if call_name in FORBIDDEN_TRITON_JIT_CALLS:
                errors.append(
                    f"Invalid Triton API inside @triton.jit kernel {kernel.name}: {call_name}"
                )

    return sorted(set(errors))


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

    try:
        tree = ast.parse(clean_code)
    except SyntaxError as exc:
        return GrammarCheckResult(
            valid=False,
            errors=[f"Python syntax error: {exc.msg}"],
            warnings=[],
        )

    kernels = [node for node in ast.walk(tree) if _is_triton_jit_kernel(node)]
    if not kernels:
        errors.append("Missing required Triton structure: triton_jit")

    for rule_name, pattern in REQUIRED_PATTERNS.items():
        if rule_name == "triton_jit":
            continue
        if not re.search(pattern, clean_code):
            errors.append(f"Missing required Triton structure: {rule_name}")

    for kernel in kernels:
        kernel_source = ast.get_source_segment(clean_code, kernel) or ""
        for rule_name, pattern in REQUIRED_PATTERNS.items():
            if rule_name in {"triton_jit", "kernel_def"}:
                continue
            if not re.search(pattern, kernel_source):
                errors.append(
                    f"Missing required Triton structure inside {kernel.name}: {rule_name}"
                )
        errors.extend(_validate_kernel_ast(kernel))

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
