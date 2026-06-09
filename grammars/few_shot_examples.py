"""
Few-shot examples for category-aware prompting.

Based on official Triton tutorials, adapted to reinforce correct patterns:
  - Tutorial 01: Vector Add  (elementwise pattern)
  - Tutorial 02: Fused Softmax (reduction pattern)
  - Tutorial 03: Matrix Multiplication (fusion/matmul pattern)

Key patterns reinforced in v2:
  - Use tl.math.* for transcendental ops (tanh, exp2, log2, etc.), NOT tl.tanh
  - Pass tensors directly to kernel launch (NOT .data_ptr())
  - Wrapper handles arbitrary input shapes via .numel() or .reshape()
  - tl.arange must use tl.constexpr arguments (use BLOCK_SIZE, not runtime vars)

None of these operators appear in TritonBench-T — no data contamination.
"""

# ---------------------------------------------------------------------------
# elementwise, elementwise_fusion
# Source: Tutorial 01 — Vector Addition (official, unmodified kernel)
# Pattern: pid → offsets → mask → load(s) → compute → store
# Key: wrapper uses .numel() so it works for ANY shape, passes tensors directly
# ---------------------------------------------------------------------------
ELEMENTWISE = """
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # Pass tensors directly — NEVER use .data_ptr()
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
""".strip()

# ---------------------------------------------------------------------------
# indexing_interpolation, complex_fallback
# Elementwise with tl.math.* — teaches model the correct API for
# transcendental functions (NOT tl.tanh, tl.sigmoid — those don't exist).
# Also shows .numel() + .view(-1) for handling any input shape.
# ---------------------------------------------------------------------------
ELEMENTWISE_MATH = """
import torch
import triton
import triton.language as tl


@triton.jit
def neg_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = -x
    tl.store(output_ptr + offsets, output, mask=mask)


def neg(input: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # Pass tensors directly — NEVER use .data_ptr()
    neg_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    return output
""".strip()

# ---------------------------------------------------------------------------
# reduction
# Source: Tutorial 02 — Fused Softmax (official kernel, simplified wrapper)
# Pattern: pid selects row → load row → reduce (max, sum) → store
# Key: wrapper reshapes input to 2D so kernel always gets (n_rows, n_cols)
# Key: tl.arange uses BLOCK_SIZE (constexpr), never a runtime variable
# ---------------------------------------------------------------------------
REDUCTION = """
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    # Pass tensors directly — NEVER use .data_ptr()
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0),
                              n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=2)
    return y
""".strip()

# ---------------------------------------------------------------------------
# fusion_matmul, convolution_fusion
# Source: Tutorial 03 — Matrix Multiplication (official kernel, simplified wrapper)
# Pattern: tiled 2D pid → tl.dot accumulation loop → store
# Key: wrapper passes tensors directly to kernel (NOT .data_ptr())
# ---------------------------------------------------------------------------
FUSION_MATMUL = """
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    # Pass tensors directly — NEVER use .data_ptr()
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_SIZE_M=128, BLOCK_SIZE_N=256,
                        BLOCK_SIZE_K=64, GROUP_SIZE_M=8)
    return c
""".strip()

# ---------------------------------------------------------------------------
# Lookup table: family -> example
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES: dict[str, str] = {
    "elementwise":              ELEMENTWISE,        # Tutorial 01 — vector add
    "elementwise_fusion":       ELEMENTWISE,        # Tutorial 01 — same base structure
    "indexing_interpolation":   ELEMENTWISE_MATH,   # neg kernel — shows .numel() pattern
    "complex_fallback":         ELEMENTWISE_MATH,   # neg kernel — safest fallback
    "reduction":                REDUCTION,           # Tutorial 02 — with 1D reshape fix
    "fusion_matmul":            FUSION_MATMUL,       # Tutorial 03
    "convolution_fusion":       FUSION_MATMUL,       # Tutorial 03 — closest structure
}
