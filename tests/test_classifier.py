from models.classifier import classify_problem
from parsing.ast_parser import parse_code
from parsing.xgrammar_converter import convert_to_xgrammar
from generation.triton_generator import generate_kernel


def test_elementwise():
    assert classify_problem("A + B") == "element_wise"


def test_reduction():
    assert classify_problem("sum(x)") == "reduction"


def test_matrix_operation():
    assert classify_problem("A @ B") == "matrix_operation"


def test_parse_code_returns_structure():
    ast_repr = parse_code("C = A + B")
    assert isinstance(ast_repr, dict)
    assert "Module" in ast_repr


def test_convert_to_xgrammar_contains_module():
    ast_repr = parse_code("C = A + B")
    grammar = convert_to_xgrammar(ast_repr)
    assert grammar.startswith("XGRAMMAR")
    assert "Module" in grammar


def test_generate_kernel_elementwise():
    code = "C = A + B"
    problem_type = classify_problem(code)
    grammar = convert_to_xgrammar(parse_code(code))
    kernel = generate_kernel(problem_type, grammar)
    assert "elementwise_kernel" in kernel
