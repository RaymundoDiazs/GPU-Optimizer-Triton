"""
Few-shot examples for category-aware prompting.

v3 — restructured for maximum pattern reinforcement:
  - 2 short examples per family instead of 1 long one
  - No comments in the code (model copies comments, ignores meaning)
  - Every wrapper passes tensors directly (never .data_ptr())
  - Every wrapper uses .numel() for element count
  - Kernel names always end with _kernel, wrapper names never do
  - tl.arange always uses BLOCK_SIZE (constexpr)

None of these operators appear in TritonBench-T — no data contamination.
"""

# ---------------------------------------------------------------------------
# elementwise, elementwise_fusion
# Two short examples: add (2 inputs) and neg (1 input)
# Both show: numel → grid → kernel[grid](tensor, tensor, ..., n, BLOCK_SIZE=)
# ---------------------------------------------------------------------------
ELEMENTWISE = """
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


@triton.jit
def neg_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, -x, mask=mask)


def neg(input: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    neg_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    return output
""".strip()

# ---------------------------------------------------------------------------
# indexing_interpolation, complex_fallback
# Same as elementwise — safest pattern for unknown operators
# ---------------------------------------------------------------------------
ELEMENTWISE_MATH = ELEMENTWISE

# ---------------------------------------------------------------------------
# reduction
# Two examples: softmax (row reduce) and sum (simple reduce)
# Both show: reshape to 2D → (n_rows,) grid → pass tensors + strides
# ---------------------------------------------------------------------------
REDUCTION = """
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    output_row_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_ptr + col_offsets, result, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton.jit
def sum_kernel(output_ptr, input_ptr, input_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
    result = tl.sum(row, axis=0)
    tl.store(output_ptr + row_idx, result)


def row_sum(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    sum_kernel[(n_rows,)](output, x, x.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output
""".strip()

# ---------------------------------------------------------------------------
# fusion_matmul, convolution_fusion
# One focused example: simple matmul with tiled 2D pid
# Shorter than Tutorial 03 to save tokens for the model's own output
# ---------------------------------------------------------------------------
FUSION_MATMUL = """
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    return c
""".strip()

# ---------------------------------------------------------------------------
# Lookup table: family -> example
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES: dict[str, str] = {
    "elementwise":              ELEMENTWISE,
    "elementwise_fusion":       ELEMENTWISE,
    "indexing_interpolation":   ELEMENTWISE_MATH,
    "complex_fallback":         ELEMENTWISE_MATH,
    "reduction":                REDUCTION,
    "fusion_matmul":            FUSION_MATMUL,
    "convolution_fusion":       FUSION_MATMUL,
}
