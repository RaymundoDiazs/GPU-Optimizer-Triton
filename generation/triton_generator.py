from generation.kernel_templates import (
    ELEMENTWISE_TEMPLATE,
    REDUCTION_TEMPLATE,
    MATMUL_TEMPLATE,
    GENERIC_TEMPLATE,
)
from models.llm_decider import decide_strategy


def generate_kernel(problem_type: str, grammar: str, code: str = "", ast_repr: dict | None = None) -> str:
    decide_strategy(code, ast_repr or {}, grammar)

    if problem_type == "element_wise":
        return ELEMENTWISE_TEMPLATE
    if problem_type == "reduction":
        return REDUCTION_TEMPLATE
    if problem_type == "matrix_operation":
        return MATMUL_TEMPLATE

    return GENERIC_TEMPLATE.format(problem_type=problem_type, grammar=grammar)
