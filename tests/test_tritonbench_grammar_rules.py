from parsing.tritonbench_grammar_rules import (
    load_tritonbench_contracts,
    validate_tritonbench_candidate,
    validate_tritonbench_family_candidate,
)


DIV_CANDIDATE = """
import torch
import triton
import triton.language as tl

@triton.jit
def div_kernel(X, Y, Z, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    z = x / y
    tl.store(Z + offsets, z, mask=mask)

def div(input, other, *, rounding_mode=None, out=None):
    return out
"""


TANH_CANDIDATE = """
import torch
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(X, Z, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    z = tl.tanh(x)
    tl.store(Z + offsets, z, mask=mask)

def tanh(input, *, out=None):
    return out
"""


TANH_WITH_TORCH_WRAPPER = """
import torch
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(X, Z, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    z = tl.tanh(x)
    tl.store(Z + offsets, z, mask=mask)

def tanh(input, *, out=None):
    output = torch.empty_like(input) if out is None else out
    return output
"""


TANH_WITH_TORCH_IN_KERNEL = """
import torch
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(X, Z, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = torch.from_buffer(X)
    z = tl.tanh(x)
    tl.store(Z + offsets, z, mask=mask)

def tanh(input, *, out=None):
    return out
"""


def test_tritonbench_subset_has_all_simple_contracts():
    contracts = load_tritonbench_contracts()

    assert set(contracts) == {f"tritonbench_t_{index:03d}" for index in range(1, 167)}


def test_div_candidate_passes_subset_contract():
    result = validate_tritonbench_candidate("tritonbench_t_002", DIV_CANDIDATE)

    assert result.valid is True
    assert result.errors == []


def test_tanh_candidate_passes_subset_contract():
    result = validate_tritonbench_candidate("tritonbench_t_005", TANH_CANDIDATE)

    assert result.valid is True
    assert result.errors == []


def test_markdown_fences_fail_contract():
    result = validate_tritonbench_candidate(
        "tritonbench_t_005",
        f"```python\n{TANH_CANDIDATE}\n```",
    )

    assert result.valid is False
    assert "Candidate contains markdown fences" in result.errors


def test_torch_is_allowed_in_wrapper_but_not_kernel():
    wrapper_result = validate_tritonbench_candidate(
        "tritonbench_t_005",
        TANH_WITH_TORCH_WRAPPER,
    )
    kernel_result = validate_tritonbench_candidate(
        "tritonbench_t_005",
        TANH_WITH_TORCH_IN_KERNEL,
    )

    assert wrapper_result.valid is True
    assert kernel_result.valid is False
    assert any("Forbidden torch use inside @triton.jit" in error for error in kernel_result.errors)


def test_div_candidate_passes_family_contract():
    result = validate_tritonbench_family_candidate("tritonbench_t_002", DIV_CANDIDATE)

    assert result.valid is True
    assert result.errors == []


def test_tanh_candidate_matches_individual_and_family_contracts():
    individual = validate_tritonbench_candidate("tritonbench_t_005", TANH_CANDIDATE)
    family = validate_tritonbench_family_candidate("tritonbench_t_005", TANH_CANDIDATE)

    assert individual.valid is True
    assert family.valid is True


def test_empty_candidate_fails_contract():
    result = validate_tritonbench_candidate("tritonbench_t_005", "")

    assert result.valid is False
    assert "Candidate code is empty" in result.errors


def test_unknown_task_fails_contract():
    result = validate_tritonbench_candidate("tritonbench_t_999", TANH_CANDIDATE)

    assert result.valid is False
    assert any("No TritonBench-T grammar contract" in error for error in result.errors)
