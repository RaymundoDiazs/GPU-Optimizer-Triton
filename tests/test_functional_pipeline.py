import pytest

from generation.triton_generator import generate_kernel
from main import run_pipeline
from models.classifier import classify_problem
from parsing.ast_parser import parse_code
from parsing.xgrammar_converter import convert_to_xgrammar
from validation.gpu_validator import _load_gpu_stack, validate_generated_kernel


PIPELINE_CASES = [
    ("C = A + B", "element_wise", "launch_elementwise_add"),
    ("output = torch.sum(x)", "reduction", "launch_reduction_sum"),
    ("C = A @ B", "matrix_operation", "launch_matmul"),
]


def _generate_from_code(code: str) -> tuple[str, str]:
    problem_type = classify_problem(code)
    ast_repr = parse_code(code)
    grammar = convert_to_xgrammar(ast_repr)
    return problem_type, generate_kernel(problem_type, grammar, code=code, ast_repr=ast_repr)


@pytest.mark.parametrize(("code", "expected_problem_type", "expected_wrapper"), PIPELINE_CASES)
def test_pipeline_generates_functional_wrapper_for_supported_examples(
    code: str,
    expected_problem_type: str,
    expected_wrapper: str,
) -> None:
    output = run_pipeline(code)

    assert f"Problem type: {expected_problem_type}" in output
    assert f"def {expected_wrapper}" in output
    assert "Generated Triton kernel:" in output


@pytest.mark.gpu
@pytest.mark.parametrize(("code", "expected_problem_type", "expected_wrapper"), PIPELINE_CASES)
def test_generated_wrapper_matches_pytorch_reference_on_gpu(
    code: str,
    expected_problem_type: str,
    expected_wrapper: str,
) -> None:
    _, reason = _load_gpu_stack()
    if reason:
        pytest.skip(reason)

    problem_type, kernel_code = _generate_from_code(code)
    result = validate_generated_kernel(problem_type, kernel_code, size=257, seed=123)

    assert problem_type == expected_problem_type
    assert result["status"] == "ok", result
    assert result["correct"] is True
    assert result["wrapper"] == expected_wrapper
    assert result["max_abs_error"] is not None


@pytest.mark.gpu
def test_generic_generated_wrapper_matches_pytorch_reference_on_gpu() -> None:
    _, reason = _load_gpu_stack()
    if reason:
        pytest.skip(reason)

    kernel_code = generate_kernel("generic", "XGRAMMAR\n  Module\n    Expr")
    result = validate_generated_kernel("generic", kernel_code, size=257, seed=123)

    assert result["status"] == "ok", result
    assert result["correct"] is True
    assert result["wrapper"] == "launch_generic"
