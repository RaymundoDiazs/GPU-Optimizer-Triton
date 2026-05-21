from generation.constrained_decoder import ConstrainedDecoder, constrain_model_output
from parsing.triton_grammar_rules import validate_triton_kernel


VALID_KERNEL = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x + y

    tl.store(output_ptr + offsets, result, mask=mask)
"""


INVALID_KERNEL_MISSING_STORE = """
import triton
import triton.language as tl

@triton.jit
def bad_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    result = x + y
"""


def test_valid_triton_kernel_passes():
    result = validate_triton_kernel(VALID_KERNEL)

    assert result.valid is True
    assert result.errors == []


def test_invalid_kernel_missing_store_fails():
    result = validate_triton_kernel(INVALID_KERNEL_MISSING_STORE)

    assert result.valid is False
    assert any("store" in error for error in result.errors)


def test_constrained_decoder_accepts_valid_kernel():
    output = constrain_model_output(VALID_KERNEL)

    assert output["accepted"] is True
    assert output["errors"] == []


def test_decoder_returns_valid_next_tokens():
    decoder = ConstrainedDecoder()
    tokens = decoder.allowed_tokens()

    assert "@triton.jit" in tokens