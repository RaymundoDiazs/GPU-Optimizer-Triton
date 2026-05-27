import json
from pathlib import Path

from evaluation.repair_generated_kernels import repair_code


def test_repair_code_adds_colon_and_renames_function(tmp_path: Path):
    input_code = 'import triton\n@triton.jit\ndef vector_add_kernel(X, Y, Z, N)\n    pass'
    output_code = repair_code(input_code)

    assert 'def add_vectors(' in output_code
    assert output_code.count(':') >= 1


def test_repair_code_adds_block_size_default():
    input_code = 'def vector_add_kernel(x, y, z, n, BLOCK_SIZE: tl.constexpr)'
    output_code = repair_code(input_code)

    assert 'BLOCK_SIZE: tl.constexpr = 1024' in output_code
