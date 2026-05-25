import ast

from generation.triton_generator import generate_kernel


def _assert_valid_python(code: str) -> None:
    ast.parse(code)


def test_elementwise_template_includes_launch_wrapper():
    kernel = generate_kernel("element_wise", "XGRAMMAR")

    _assert_valid_python(kernel)
    assert "def elementwise_kernel" in kernel
    assert "def launch_elementwise_add" in kernel
    assert "torch.empty_like" in kernel
    assert "elementwise_kernel[grid]" in kernel
    assert "BLOCK_SIZE=block_size" in kernel


def test_reduction_template_includes_launch_wrapper():
    kernel = generate_kernel("reduction", "XGRAMMAR")

    _assert_valid_python(kernel)
    assert "def reduction_kernel" in kernel
    assert "def launch_reduction_sum" in kernel
    assert "partials = torch.empty" in kernel
    assert "reduction_kernel[grid]" in kernel
    assert "return partials.sum()" in kernel


def test_matmul_template_includes_launch_wrapper():
    kernel = generate_kernel("matrix_operation", "XGRAMMAR")

    _assert_valid_python(kernel)
    assert "def matmul_kernel" in kernel
    assert "def launch_matmul" in kernel
    assert "tl.dot" in kernel
    assert "matmul_kernel[grid]" in kernel
    assert "BLOCK_M=block_m" in kernel


def test_generic_template_escapes_multiline_grammar_as_comments():
    kernel = generate_kernel("unknown", "XGRAMMAR\n  Module\n    Assign")

    _assert_valid_python(kernel)
    assert "# XGRAMMAR" in kernel
    assert "#   Module" in kernel
    assert "#     Assign" in kernel
    assert "def launch_generic" in kernel
